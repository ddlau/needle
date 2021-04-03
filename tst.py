
import os
import redis
import pickle

import numpy as np

os.environ[ 'path' ] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin;' + os.environ.get( 'path', str() )

import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.runners import traj_segment_generator
from stable_baselines.common.policies import MlpPolicy








PMT = '11.3f'
VMT = '11.3f'
AMT = '11.4f'


class Environment:
	def __init__( self, initial, start, stride, channel, unit, rate4ts, rate4tb, rate4ms, rate4mb, key='observations', host='localhost', port=6379, check=True ):
		self.check = check

		self.key = key
		self.redis = redis.StrictRedis( host=host, port=port )
		self.initial = initial

		self.start = start
		self.stride = stride
		self.channel = channel

		self.unit = unit
		self.rate4ts = rate4ts
		self.rate4tb = rate4tb
		self.rate4ms = rate4ms
		self.rate4mb = rate4mb

		self.b = None
		self.q = None

		self.idx = 0
		self.old = None
		self.new = None
		self.p_of_old = None
		self.p_of_new = None

		self.stop = True
		self.done = True

		self.observation_space = gym.spaces.Box( float( '-inf' ), float( '+inf' ), [127] )
		self.action_space = gym.spaces.Box( -1.0,+1.0, [ 1 ] )#gym.spaces.Box( float( '-inf' ), float( '+inf' ), [ 1 ] )


	def observe( self ):
		observation = self.redis.xread( { self.key: self.idx }, count=self.channel )
		if observation:
			observation = observation[0][1]
			if len(observation) == self.channel:
				self.stop = None

				self.idx = observation[ self.stride - 1 ][ 0 ]
				self.old = self.new
				self.new = list( pickle.loads( observation[ i ][ 1 ][ b'data' ] ) for i in range( self.channel ) )

				self.p_of_old = self.p_of_new
				self.p_of_new = self.new[ -1 ][ 'p' ]

				return

		self.stop = True

		self.idx = self.start( self.idx )
		self.old = None
		self.new = None

		self.p_of_old = None
		self.p_of_new = None

	def assemble( self ):
		assert self.b is not None
		assert self.q is not None

		asks = np.dstack( list( x[ 'asks' ] for x in self.new ) )
		bids = np.dstack( list( x[ 'bids' ] for x in self.new ) )

		d = np.concatenate( (asks, bids), axis=0 )

		seq_of_p = d[ :, 0, : ]
		seq_of_a = d[ :, 1, : ]

		s = np.vstack( list( x[ 's' ] for x in self.new ) )
		b = np.vstack( list( x[ 'b' ] for x in self.new ) )

		amt_of_s, pwr_of_s = np.sum( s, axis=0 )
		amt_of_b, pwr_of_b = np.sum( b, axis=0 )

		return seq_of_p, seq_of_a, np.asarray( [ amt_of_s, pwr_of_s, self.b, self.p_of_new, self.q, amt_of_b, pwr_of_b ] )

	def perform( self, a ):
		try:
			a = a[ 0 ]

			reward = 0

			q = self.q* abs(a)
			if a > 0 and q > self.p_of_old * self.unit:
				d = q
				b = 0
				for p, a in self.new[ -1 ][ 'asks' ][ ::-1 ]:
					a = min( a, d / p )

					b += a * (1 - self.rate4tb)
					d -= p * a

					if not d > 0:
						break

				reward -= q
				self.q -= q
				reward -= self.p_of_old * self.b
				self.b += b
				reward += self.p_of_new * self.b

				if self.check:
					print(
						f'{self.p_of_old:>{PMT}} => {self.p_of_new:<{PMT}} {"↑" if self.p_of_new > self.p_of_old else "↓" if self.p_of_new < self.p_of_old else "→"} '
						f'{reward:>+{VMT}} {self.b:>{AMT}} + {self.q:<{VMT}} = {self.q + self.p_of_new * self.b:<{VMT}} '
						f'B {q:>{VMT}} => {b:<{AMT}} '
					)

				return reward

			b = self.b * abs( a )
			if a < 0 and b > self.unit:
				d = b
				q = 0
				for p, a in self.new[ -1 ][ 'bids' ]:
					a = min( a, d )

					q += p * a * (1 - self.rate4ts)
					d -= a

					if not d > 0:
						break

				reward += q
				self.q += q
				reward -= self.p_of_old * self.b
				self.b -= b
				reward += self.p_of_new * self.b

				if self.check:
					print(
						f'{self.p_of_old:>{PMT}} => {self.p_of_new:<{PMT}} {"↑" if self.p_of_new > self.p_of_old else "↓" if self.p_of_new < self.p_of_old else "→"} '
						f'{reward:>+{VMT}} {self.b:>{AMT}} + {self.q:<{VMT}} = {self.q + self.p_of_new * self.b:<{VMT}} '
						f'S {b:>{AMT}} => {q:<{VMT}} '
					)

				return reward

			reward += (self.p_of_new - self.p_of_old) * self.b
			if self.check:
				print(
					f'{self.p_of_old:>{PMT}} => {self.p_of_new:<{PMT}} {"↑" if self.p_of_new > self.p_of_old else "↓" if self.p_of_new < self.p_of_old else "→"} '
					f'{reward:>+{VMT}} {self.b:>{AMT}} + {self.q:<{VMT}} = {self.q + self.p_of_new * self.b:<{VMT}} '
				)

			return reward

		finally:
			self.done = (self.b + self.q / self.p_of_new) < self.unit * 5

			if self.done:
				self.idx = self.start( self.idx )
				self.old = None
				self.new = None

				self.p_of_old = None
				self.p_of_new = None

	def reset( self ):
		assert self.stop or self.done
		self.stop = None
		self.done = None

		self.b, self.q = self.initial()

		while True:
			self.observe()

			if self.stop:
				continue

			return self.assemble()

	def step( self, a ):
		assert np.all( -1.0 <= a <= +1.0 )
		assert not self.stop and not self.done

		self.observe()
		if self.stop:
			return None, None, self.stop, self.done, dict()

		reward = self.perform( a )
		if self.done:
			return None, reward, self.stop, self.done, dict()

		return self.assemble(), reward, self.stop, self.done, dict()






class χEnvironment( gym.Env ):
	def render( self, mode='human' ):
		pass

	def __init__( self, initial, unit=0.0001, rate4ts=0.002, rate4tb=0.002, rate4ms=0.002, rate4mb=0.002, key='observations', host='localhost', port=6379 ):
		self.key = key
		self.redis = redis.StrictRedis( host=host, port=port )
		self.initial = initial

		self.unit = unit

		self.rate4ts = rate4ts
		self.rate4tb = rate4tb
		self.rate4ms = rate4ms
		self.rate4mb = rate4mb

		self.b = None
		self.q = None

		self.idx = 0
		self.old = None
		self.new = None

		self.stop = True
		self.done = True

		x = self.redis.xread( { key: 0 }, count=1 )[ 0 ][ 1 ][ 0 ][ 1 ][ b'data' ]
		x = pickle.loads( x )
		x = np.concatenate( (x[ 'asks' ].flatten(), x[ 's' ].flatten(), (self.b, x[ 'p' ], self.q), x[ 'b' ].flatten(), x[ 'bids' ].flatten()) ).shape

		self.log = print

	def assemble( self ):
		return np.concatenate( (
			self.new[ 'asks' ].flatten(),
			self.new[ 's' ].flatten(),
			(self.b, self.new[ 'p' ], self.q),
			self.new[ 'b' ].flatten(),
			self.new[ 'bids' ].flatten(),
		) )

	def observe( self ):
		observation = self.redis.xread( { self.key: self.idx }, count=1 )

		if observation:
			self.stop = None
			self.idx = observation[ 0 ][ 1 ][ 0 ][ 0 ]
			self.old = self.new
			self.new = pickle.loads( observation[ 0 ][ 1 ][ 0 ][ 1 ][ b'data' ] )
		else:
			self.stop = True
			self.idx = 0
			self.old = None
			self.new = None


	def perform( self, a ):
		try:
			assert np.all( -1.0 <= a <= +1.0)
			#a = np.clip(a,-1.0,+1.0)
			reward = 0

			q = self.q * abs( a )
			#print( q, self.q )
			if a > 0 and q > self.old[ 'p' ] * self.unit:
				d = q
				b = 0
				for p, a in self.old[ 'asks' ][ ::-1 ]:
					a = min( a, d / p )
					b += a * (1 - self.rate4tb)
					d -= p * a
					if not d > 0:
						break

				reward -= q
				self.q -= q
				reward -= self.old[ 'p' ] * self.b
				self.b += b
				reward += self.new[ 'p' ] * self.b

				self.log( f'P: {self.old["p"]:>16.3f}=>{self.new["p"]:<16.3f}B: {q:>16.3f}=>{b:<16.3f}state={self.b:>16.8f}/{self.q:<16.8f}R={reward:<8.3f}' )

				assert self.b > - 1e-7, f'b={self.b:>16.8f}, q={self.q:>16.8f}'
				assert self.q > - 1e-7, f'b={self.b:>16.8f}, q={self.q:>16.8f}'
				return reward

			b = self.b * abs( a )
			if a < 0 and b > self.unit:
				d = b
				q = 0
				for p, a in self.old[ 'bids' ]:
					a = min( a, d )
					q += p * a * (1 - self.rate4ts)
					d -= a
					if not d > 0:
						break

				reward += q
				self.q += q
				reward -= self.old[ 'p' ] * self.b
				self.b -= b
				reward += self.new[ 'p' ] * self.b

				self.log( f'P: {self.old["p"]:>16.3f}=>{self.new["p"]:<16.3f}S: {b:>16.3f}=>{q:<16.3f}state={self.b:>16.8f}/{self.q:<16.8f}R={reward:<8.3f}' )
				assert self.b > - 1e-7, f'b={self.b:>16.8f}, q={self.q:>16.8f}'
				assert self.q > - 1e-7, f'b={self.b:>16.8f}, q={self.q:>16.8f}'
				return reward

			reward += self.b * (self.new[ 'p' ] - self.old[ 'p' ])

			assert self.b > - 1e-7, f'b={self.b:>16.8f}, q={self.q:>16.8f}'
			assert self.q > - 1e-7, f'b={self.b:>16.8f}, q={self.q:>16.8f}'
			return reward

		finally:
			#print( f'{self.old["p"]}=>{self.new["p"]}: {reward}')

			self.done = ( self.b + self.q / self.new[ 'p' ] ) < self.unit * 10
			#print( '@'*128, ( self.b + self.q / self.new[ 'p' ] ), self.b, self.q)
			if self.done:
				print( '@'*128, ( self.b + self.q / self.new[ 'p' ] ), self.b, self.q)

	def reset( self ):
		#print( 'reset')
		assert self.stop or self.done
		self.stop = None
		self.done = None

		self.b, self.q = self.initial()

		while True:
			self.observe()

			if self.stop:
				continue

			return self.assemble()

	def step( self, a ):
		#print( 'step')
		a = a[0]
		# assert np.all( -1.0 <= a <= +1.0 )
		assert not self.stop and not self.done

		self.observe()
		if self.stop:
			return None, None, self.stop, self.done, dict()

		reward = self.perform( a )
		if self.done:
			return None, reward, self.stop, self.done, dict()

		return self.assemble(), reward, self.stop, self.done, dict()







def tst():
	env = Environment( lambda :(0.01, 1000) )
	print( env.observation_space)
	print( env.action_space )

	print( env.reset() )
	print( env.step(0))


	#check_env( env )

	#
	#
	# for x in traj_segment_generator( MlpPolicy, env, 1024 ):
	# 	print( x )
	#
	# pass


if __name__ == '__main__':
	tst()
	exit()

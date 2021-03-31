
import os
import redis
import pickle

import numpy as np

os.environ[ 'path' ] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin;' + os.environ.get( 'path', str() )

import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.runners import traj_segment_generator
from stable_baselines.common.policies import MlpPolicy















class Environment( gym.Env ):
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
		self.observation_space = gym.spaces.Box( float( '-inf' ), float( '+inf' ), x )
		self.action_space = gym.spaces.Box( -1.0,+1.0, [ 1 ] )#gym.spaces.Box( float( '-inf' ), float( '+inf' ), [ 1 ] )

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
			reward = 0

			q = self.q * abs( a )
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

				return reward

			reward += self.b * (self.new[ 'p' ] - self.old[ 'p' ])

			return reward

		finally:
			#print( f'{self.old["p"]}=>{self.new["p"]}: {reward}')

			self.done = self.b < self.unit and self.q / self.new[ 'p' ] < self.unit

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

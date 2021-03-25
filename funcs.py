import gym
import redis
import pickle

import numpy as np


class x1Environment( gym.Env ):
	def render( self, mode='human' ):
		pass

	def __init__( self, initial, reset_to_head_after_done=None, unit=0.0001, rate4ts=0.002, rate4tb=0.002, rate4ms=0.002, rate4mb=0.002, host='localhost', port=6379 ):
		self.redis = redis.StrictRedis( host=host, port=port )

		self.initial = initial

		self.reset_to_head_after_done = reset_to_head_after_done

		self.unit = unit

		self.rate4ts = rate4ts
		self.rate4tb = rate4tb
		self.rate4ms = rate4ms
		self.rate4mb = rate4mb

		self.b = None
		self.q = None

		self.idx = None
		self.old = None
		self.new = None

		self.stop = True
		self.done = True

		self.action_space = gym.spaces.Box( -1.0, +1.0, [1] )

		print( self.action_space)
		print( self.action_space.sample())

		x = pickle.loads(self.redis.xread(dict(observations=0),count=1)[0][1][0][1][b'data'] )
		x = np.concatenate( (
			x[ 'asks' ].flatten(),
			x[ 's' ].flatten(),
			(self.b, None, self.q),
			x[ 'b' ].flatten(),
			x[ 'bids' ].flatten(),
		) )
		print( len( x ) )

		#self.observation_space = gym.

	def assemble( self, x ):
		return np.concatenate( (
			x[ 'asks' ].flatten(),
			x[ 's' ].flatten(),
			(self.b, x[ 'p' ], self.q),
			x[ 'b' ].flatten(),
			x[ 'bids' ].flatten(),
		) )

	def reset( self ):
		assert self.stop or self.done

		if self.done and self.reset_to_head_after_done:
			self.idx = 0

		self.stop=None
		self.done = None

		self.b, self.q = self.initial()

		while True:
			observations = self.redis.xread( dict( observations = self.idx), count=2)
			if not observations or len( observations[0][1]) < 2:
				self.idx = 0
				continue

			self.idx = observations[0][1][1][0]
			self.old = pickle.loads( observations[0][1][0][1][b'data'])
			self.new = pickle.loads( observations[0][1][1][1][b'data'])

			return self.assemble( self.old )

	def step( self, a ):
		assert -1.0 <= a <= +1.0
		assert not self.stop and not self.done

		reward = 0

		if a > 0:
			q = self.q * a

			if q > self.old['p'] * self.unit:
				reward -= q
				self.q -= q

				b = 0
				for p, a in self.old['asks'][::-1]:
					a = min( a, q / p )

					b += a * ( 1 - self.rate4tb)
					q -= p * a

					if not q >0:
						break

				reward -= self.old['p'] * self.b
				self.b += b
				reward += self.new['p'] * self.b

			else:
				reward += self.b * ( self.new['p']-self.old['p'])

		if a < 0:
			b = self.b * a

			if b > self.unit:
				reward -= self.old['p'] * self.b

				q = 0
				for p, a in self.old['bids']:
					a = min( a, b )

					b -= a
					q += p * a * ( 1-self.rate4ts)

					if not b >= 0:
						break

				reward += q
				self.q += q

				self.b -=b
				reward += self.new['p'] * self.b

			else:
				reward += self.b * ( self.new['p'] - self.old['p'])

		observation = self.redis.xread( dict( observations=self.idx), count=1)

		self.stop = not observation
		self.done = self.b < self.unit and self.q / self.new['p'] < self.unit

		self.idx = 0 if self.stop else observation[0][1][0][0]
		self.old = self.new
		self.new = pickle.loads( observation[0][1][0][1][b'data'])

		return self.assemble( self.old), reward, self.stop, self.done, dict()




class x2Environment( gym.Env ):
	def render( self, mode='human' ):
		pass

	def __init__( self, initial, reset_to_head_after_done=None, unit=0.0001, rate4ts=0.002, rate4tb=0.002, rate4ms=0.002, rate4mb=0.002, host='localhost', port=6379 ):
		self.redis = redis.StrictRedis( host=host, port=port )

		self.initial = initial

		self.reset_to_head_after_done = reset_to_head_after_done

		self.unit = unit

		self.rate4ts = rate4ts
		self.rate4tb = rate4tb
		self.rate4ms = rate4ms
		self.rate4mb = rate4mb

		self.b = None
		self.q = None

		self.idx = None
		self.old = None
		self.new = None

		self.stop = True
		self.done = True

		self.action_space = gym.spaces.Box( -1.0, +1.0, [1] )

		print( self.action_space)
		print( self.action_space.sample())

		x = pickle.loads(self.redis.xread(dict(observations=0),count=1)[0][1][0][1][b'data'] )
		x = np.concatenate( (
			x[ 'asks' ].flatten(),
			x[ 's' ].flatten(),
			(self.b, None, self.q),
			x[ 'b' ].flatten(),
			x[ 'bids' ].flatten(),
		) )
		print( len( x ) )

	#self.observation_space = gym.

	def assemble( self, x ):
		return np.concatenate( (
			x[ 'asks' ].flatten(),
			x[ 's' ].flatten(),
			(self.b, x[ 'p' ], self.q),
			x[ 'b' ].flatten(),
			x[ 'bids' ].flatten(),
		) )

	def reset( self ):
		assert self.stop or self.done

		if self.stop or self.reset_to_head_after_done:
			self.idx = 0

		self.stop = None
		self.done = None

		self.b, self.q = self.initial()

		while True:
			observation = self.redis.xread( dict( observations=self.idx), count=1)
			if not observation:
				self.idx = 0
				continue

			self.idx = observation[0][1][0][0]
			self.new = pickle.loads( observation[0][1][0][1][b'data'])

			return self.assemble(self.new)

	def step( self, a ):
		assert -1.0 <=a<= +1.0
		assert not self.stop and not self.done

		observation =  self.redis.xread( dict( observations=self.idx), count=1)
		self.stop = not observation

		if self.stop:
			return None, None, self.stop, self.done, dict()

		self.idx = observation[0][1][0][0]
		self.old = self.new
		self.new = pickle.loads( observation[0][1][0][1][b'data'])

		reward = self.perform(a)

		self.done = ...





	def reset( self ):
		assert self.stop or self.done

		if self.done and self.reset_to_head_after_done:
			self.idx = 0

		self.stop=None
		self.done = None

		self.b, self.q = self.initial()

		while True:
			observations = self.redis.xread( dict( observations = self.idx), count=2)
			if not observations or len( observations[0][1]) < 2:
				self.idx = 0
				continue

			self.idx = observations[0][1][1][0]
			self.old = pickle.loads( observations[0][1][0][1][b'data'])
			self.new = pickle.loads( observations[0][1][1][1][b'data'])

			return self.assemble( self.old )

	def step( self, a ):
		assert -1.0 <= a <= +1.0
		assert not self.stop and not self.done

		reward = 0

		if a > 0:
			q = self.q * a

			if q > self.old['p'] * self.unit:
				reward -= q
				self.q -= q

				b = 0
				for p, a in self.old['asks'][::-1]:
					a = min( a, q / p )

					b += a * ( 1 - self.rate4tb)
					q -= p * a

					if not q >0:
						break

				reward -= self.old['p'] * self.b
				self.b += b
				reward += self.new['p'] * self.b

			else:
				reward += self.b * ( self.new['p']-self.old['p'])

		if a < 0:
			b = self.b * a

			if b > self.unit:
				reward -= self.old['p'] * self.b

				q = 0
				for p, a in self.old['bids']:
					a = min( a, b )

					b -= a
					q += p * a * ( 1-self.rate4ts)

					if not b >= 0:
						break

				reward += q
				self.q += q

				self.b -=b
				reward += self.new['p'] * self.b

			else:
				reward += self.b * ( self.new['p'] - self.old['p'])

		observation = self.redis.xread( dict( observations=self.idx), count=1)

		self.stop = not observation
		self.done = self.b < self.unit and self.q / self.new['p'] < self.unit

		self.idx = 0 if self.stop else observation[0][1][0][0]
		self.old = self.new
		self.new = pickle.loads( observation[0][1][0][1][b'data'])

		return self.assemble( self.old), reward, self.stop, self.done, dict()









	def tst( self ):
		x = self.redis.xread( dict( observations="16138851382799-0" ), count=2 )
		print( x )





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

		def func():
			x = self.redis.xread( { key: 0 }, count=1 )[ 0 ][ 1 ][ 0 ][ 1 ][ b'data' ]
			x = pickle.loads( x )
			x = np.concatenate( (x[ 'asks' ].flatten(), x[ 's' ].flatten(), (self.b, x[ 'p' ], self.q), x[ 'b' ].flatten(), x[ 'bids' ].flatten()) ).shape
			return gym.spaces.Box( float( '-inf' ), float( '+inf' ), [ 1 ] ), gym.spaces.Box( float( '-inf' ), float( '+inf' ), x )

		self.action_space, self.observation_space = func()

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

		self.stop = not observation

		self.idx = 0 if self.stop else observation[ 0 ][ 1 ][ 0 ][ 0 ]
		self.old = self.new
		self.new = pickle.loads( observation[ 0 ][ 1 ][ 0 ][ 1 ][ b'data' ] )

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
			self.done = self.b < self.unit and self.q / self.new[ 'p' ] < self.unit

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

if __name__ == '__main__':

	#print( np.concatenate( ( np.random.randn(5), ( None, 123, None), np.random.randn(5))).shape)
	#exit()

	environment = Environment( lambda: 0.01, 1000 )
	environment.tst()

	exit()


import os
import redis
import pickle

import numpy as np

os.environ[ 'path' ] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin;' + os.environ[ 'path' ]

import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.runners import traj_segment_generator
from stable_baselines.common.policies import MlpPolicy





class Environment( gym.Env ):

	def render( self, mode='human' ):
		pass

	def __init__( self,
				  rate_for_ts=0.002,
				  rate_for_tb=0.002,
				  b_at_first=0.01,
				  b_at_least=0.001,
				  q_at_first=1000,
				  q_at_least=100,
				  host='localhost',
				  port=6379
				  ):
		self.action_space = gym.spaces.Box( float( '-inf' ), float( '+inf' ), [ 1 ] )
		self.observation_space = gym.spaces.Box( float( '-inf' ), float( '+inf' ), [ 60 + 2 + 2 ] )

		self.redis = redis.StrictRedis( host=host, port=port )

		self.latest = 0

		self.rate_for_ts = rate_for_ts
		self.rate_for_tb = rate_for_tb

		self.b_at_first = b_at_first
		self.b_at_least = b_at_least
		self.q_at_first = q_at_first
		self.q_at_least = q_at_least

		self.b = b_at_first
		self.q = q_at_first

	def observation( self ):
		observation = self.redis.xread( dict( observations=self.latest ), count=1 )
		self.latest = observation[ 0 ][ 1 ][ 0 ][ 0 ]
		return pickle.loads( observation[ 0 ][ 1 ][ 0 ][ 1 ][ b'data' ] )

	def state( self, observation=None ):
		if not observation:
			observation = self.observation()
		return np.concatenate( (observation[ 'asks' ].flatten(), observation[ 's' ].flatten(), (self.b, self.q), observation[ 'b' ].flatten(), observation[ 'bids' ].flatten()) )

	def reset( self ):
		self.latest = 0
		self.b = self.b_at_first
		self.q = self.q_at_first
		return self.state()

	def step( self, a ):
		return self.observation(), ...


def tst():

	# print( redis.StrictRedis().xread( { 'observations' : "9913881416585-0" }, 1))
	# exit()
	#
	# print( np.concatenate( (np.random.randn( 3, 4 ).flatten(), (123, 321), np.random.randn( 4, 3 ).flatten()) ) )
	# exit()

	env = gym.make( 'CartPole-v1')
	# print( env.observation_space )
	# print( env.action_space )

	#env = Environment()

	#check_env( env )



	for x in traj_segment_generator( MlpPolicy, env, 1024 ):
		print( x )

	pass


if __name__ == '__main__':
	tst()
	exit()

import os
os.environ['path'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin;' + os.environ['path']

print(os.environ['path'])


import gym

from stable_baselines.common.runners import traj_segment_generator

def tst():

	env = gym.make( 'CartPole-v1')
	print( env.observation_space )
	print( env.action_space )

	pass

if __name__ == '__main__':
	tst()
	exit()
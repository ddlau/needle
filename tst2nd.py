from abc import ABC, abstractmethod
import typing
from typing import Union, Optional, Any

import gym
import numpy as np

from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import VecEnv

if typing.TYPE_CHECKING:
	from stable_baselines.common.base_class import BaseRLModel  # pytype: disable=pyi-error

from stable_baselines.common.policies import MlpPolicy


import time
from contextlib import contextmanager
from collections import deque

import gym
from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common import explained_variance, zipsame, dataset, fmt_row, colorize, ActorCriticRLModel, \
	SetVerbosity, TensorboardWriter
from stable_baselines import logger
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.misc_util import flatten_lists


def traj_segment_generator( policy, env, horizon, callback=None ):
	step = 0
	action = env.action_space.sample()  # not used, just so we have the datatype
	observation = env.reset()

	cur_ep_ret = 0  # return in current episode
	current_it_len = 0  # len of current iteration
	current_ep_len = 0  # len of current episode
	ep_rets = [ ]  # returns of completed episodes in this segment
	ep_lens = [ ]  # Episode lengths

	# Initialize history arrays
	observations = np.array( [ observation for _ in range( horizon ) ] )
	rewards = np.zeros( horizon, 'float32' )
	vpreds = np.zeros( horizon, 'float32' )
	episode_starts = np.zeros( horizon, 'bool' )
	dones = np.zeros( horizon, 'bool' )
	actions = np.array( [ action for _ in range( horizon ) ] )
	states = policy.initial_state
	episode_start = True  # marks if we're on first timestep of an episode
	done = False

	callback.on_rollout_start()

	while True:
		action, vpred, states, _ = policy.step( observation.reshape( -1, *observation.shape ), states, done )
		# Slight weirdness here because we need value function at time T
		# before returning segment [0, T-1] so we get the correct
		# terminal value

		if step > 0 and step % horizon == 0:
			callback.update_locals( locals() )
			callback.on_rollout_end()
			yield {
				"observations"     : observations,
				"rewards"          : rewards,
				"dones"            : dones,
				"episode_starts"   : episode_starts,
				"vpred"            : vpreds,
				"actions"          : actions,
				"nextvpred"        : vpred[ 0 ] * (1 - episode_start),
				"ep_rets"          : ep_rets,
				"ep_lens"          : ep_lens,
				"total_timestep"   : current_it_len,
				'continue_training': True
			}
			_, vpred, _, _ = policy.step( observation.reshape( -1, *observation.shape ) )
			# Be careful!!! if you change the downstream algorithm to aggregate
			# several of these batches, then be sure to do a deepcopy
			ep_rets = [ ]
			ep_lens = [ ]
			# Reset current iteration length
			current_it_len = 0
			callback.on_rollout_start()

		i = step % horizon
		observations[ i ] = observation
		vpreds[ i ] = vpred[ 0 ]
		actions[ i ] = action[ 0 ]
		episode_starts[ i ] = episode_start

		clipped_action = action
		# Clip the actions to avoid out of bound error
		if isinstance( env.action_space, gym.spaces.Box ):
			clipped_action = np.clip( action, env.action_space.low, env.action_space.high )

		observation, reward, done, info = env.step( clipped_action[ 0 ] )

		if callback is not None:
			callback.update_locals( locals() )
			if callback.on_step() is False:
				# We have to return everything so pytype does not complain
				yield {
					"observations"     : observations,
					"rewards"          : rewards,
					"dones"            : dones,
					"episode_starts"   : episode_starts,
					"vpred"            : vpreds,
					"actions"          : actions,
					"nextvpred"        : vpred[ 0 ] * (1 - episode_start),
					"ep_rets"          : ep_rets,
					"ep_lens"          : ep_lens,
					"total_timestep"   : current_it_len,
					'continue_training': False
				}
				return

		rewards[ i ] = reward
		dones[ i ] = done
		episode_start = done

		cur_ep_ret += reward
		current_it_len += 1
		current_ep_len += 1
		if done:
			# Retrieve unnormalized reward if using Monitor wrapper
			maybe_ep_info = info.get( 'episode' )
			if maybe_ep_info is not None:
				cur_ep_ret = maybe_ep_info[ 'r' ]

			ep_rets.append( cur_ep_ret )
			ep_lens.append( current_ep_len )
			cur_ep_ret = 0
			current_ep_len = 0
			if not isinstance( env, VecEnv ):
				observation = env.reset()
		step += 1


def add_vtarg_and_adv( seg, gamma, lam ):
	"""
	Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

	:param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
	:param gamma: (float) Discount factor
	:param lam: (float) GAE factor
	"""
	# last element is only used for last vtarg, but we already zeroed it if last new = 1
	episode_starts = np.append( seg[ "episode_starts" ], False )
	vpred = np.append( seg[ "vpred" ], seg[ "nextvpred" ] )
	rew_len = len( seg[ "rewards" ] )
	seg[ "adv" ] = np.empty( rew_len, 'float32' )
	rewards = seg[ "rewards" ]
	lastgaelam = 0
	for step in reversed( range( rew_len ) ):
		nonterminal = 1 - float( episode_starts[ step + 1 ] )
		delta = rewards[ step ] + gamma * vpred[ step + 1 ] * nonterminal - vpred[ step ]
		seg[ "adv" ][ step ] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
	seg[ "tdlamret" ] = seg[ "adv" ] + seg[ "vpred" ]


class TRPO( ActorCriticRLModel ):


	def __init__( self, policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
	              entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
	              _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
	              seed=None, n_cpu_tf_sess=1 ):
		super( TRPO, self ).__init__( policy=policy, env=env, verbose=verbose, requires_vec_env=False,
		                              _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
		                              seed=seed, n_cpu_tf_sess=n_cpu_tf_sess )

		self.timesteps_per_batch = timesteps_per_batch
		self.cg_iters = cg_iters
		self.cg_damping = cg_damping
		self.gamma = gamma
		self.lam = lam
		self.max_kl = max_kl
		self.vf_iters = vf_iters
		self.vf_stepsize = vf_stepsize
		self.entcoeff = entcoeff
		self.tensorboard_log = tensorboard_log
		self.full_tensorboard_log = full_tensorboard_log

		# GAIL Params
		self.hidden_size_adversary = 100
		self.adversary_entcoeff = 1e-3
		self.expert_dataset = None
		self.g_step = 1
		self.d_step = 1
		self.d_stepsize = 3e-4

		self.graph = None
		self.sess = None
		self.policy_pi = None
		self.loss_names = None
		self.assign_old_eq_new = None
		self.compute_losses = None
		self.compute_lossandgrad = None
		self.compute_fvp = None
		self.compute_vflossandgrad = None
		self.d_adam = None
		self.vfadam = None
		self.get_flat = None
		self.set_from_flat = None
		self.timed = None
		self.allmean = None
		self.nworkers = None
		self.rank = None
		self.reward_giver = None
		self.step = None
		self.proba_step = None
		self.initial_state = None
		self.params = None
		self.summary = None

		if _init_setup_model:
			self.setup_model()

	def _get_pretrain_placeholders( self ):
		policy = self.policy_pi
		action_ph = policy.pdtype.sample_placeholder( [ None ] )
		if isinstance( self.action_space, gym.spaces.Discrete ):
			return policy.obs_ph, action_ph, policy.policy
		return policy.obs_ph, action_ph, policy.deterministic_action

	def setup_model( self ):
		# prevent import loops
		from stable_baselines.gail.adversary import TransitionClassifier

		with SetVerbosity( self.verbose ):

			assert issubclass( self.policy, ActorCriticPolicy ), "Error: the input policy for the TRPO model must be " \
			                                                     "an instance of common.policies.ActorCriticPolicy."

			self.nworkers = MPI.COMM_WORLD.Get_size()
			self.rank = MPI.COMM_WORLD.Get_rank()
			np.set_printoptions( precision=3 )

			self.graph = tf.Graph()
			with self.graph.as_default():
				self.set_random_seed( self.seed )
				self.sess = tf_util.make_session( num_cpu=self.n_cpu_tf_sess, graph=self.graph )

				# Construct network for new policy
				self.policy_pi = self.policy( self.sess, self.observation_space, self.action_space, self.n_envs, 1,
				                              None, reuse=False, **self.policy_kwargs )

				# Network for old policy
				with tf.variable_scope( "oldpi", reuse=False ):
					old_policy = self.policy( self.sess, self.observation_space, self.action_space, self.n_envs, 1,
					                          None, reuse=False, **self.policy_kwargs )

				with tf.variable_scope( "loss", reuse=False ):
					atarg = tf.placeholder( dtype=tf.float32, shape=[ None ] )  # Target advantage function (if applicable)
					ret = tf.placeholder( dtype=tf.float32, shape=[ None ] )  # Empirical return

					observation = self.policy_pi.obs_ph
					action = self.policy_pi.pdtype.sample_placeholder( [ None ] )

					kloldnew = old_policy.proba_distribution.kl( self.policy_pi.proba_distribution )
					ent = self.policy_pi.proba_distribution.entropy()
					meankl = tf.reduce_mean( kloldnew )
					meanent = tf.reduce_mean( ent )
					entbonus = self.entcoeff * meanent

					vferr = tf.reduce_mean( tf.square( self.policy_pi.value_flat - ret ) )

					# advantage * pnew / pold
					ratio = tf.exp( self.policy_pi.proba_distribution.logp( action ) -
					                old_policy.proba_distribution.logp( action ) )
					surrgain = tf.reduce_mean( ratio * atarg )

					optimgain = surrgain + entbonus
					losses = [ optimgain, meankl, entbonus, surrgain, meanent ]
					self.loss_names = [ "optimgain", "meankl", "entloss", "surrgain", "entropy" ]

					dist = meankl

					all_var_list = tf_util.get_trainable_vars( "model" )
					var_list = [ v for v in all_var_list if "/vf" not in v.name and "/q/" not in v.name ]
					vf_var_list = [ v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name ]

					self.get_flat = tf_util.GetFlat( var_list, sess=self.sess )
					self.set_from_flat = tf_util.SetFromFlat( var_list, sess=self.sess )

					klgrads = tf.gradients( dist, var_list )
					flat_tangent = tf.placeholder( dtype=tf.float32, shape=[ None ], name="flat_tan" )
					shapes = [ var.get_shape().as_list() for var in var_list ]
					start = 0
					tangents = [ ]
					for shape in shapes:
						var_size = tf_util.intprod( shape )
						tangents.append( tf.reshape( flat_tangent[ start: start + var_size ], shape ) )
						start += var_size
					gvp = tf.add_n( [ tf.reduce_sum( grad * tangent )
					                  for (grad, tangent) in zipsame( klgrads, tangents ) ] )  # pylint: disable=E1111
					# Fisher vector products
					fvp = tf_util.flatgrad( gvp, var_list )

					tf.summary.scalar( 'entropy_loss', meanent )
					tf.summary.scalar( 'policy_gradient_loss', optimgain )
					tf.summary.scalar( 'value_function_loss', surrgain )
					tf.summary.scalar( 'approximate_kullback-leibler', meankl )
					tf.summary.scalar( 'loss', optimgain + meankl + entbonus + surrgain + meanent )

					self.assign_old_eq_new = \
						tf_util.function( [ ], [ ], updates=[ tf.assign( oldv, newv ) for (oldv, newv) in
						                                      zipsame( tf_util.get_globals_vars( "oldpi" ),
						                                               tf_util.get_globals_vars( "model" ) ) ] )
					self.compute_losses = tf_util.function( [ observation, old_policy.obs_ph, action, atarg ], losses )
					self.compute_fvp = tf_util.function( [ flat_tangent, observation, old_policy.obs_ph, action, atarg ],
					                                     fvp )
					self.compute_vflossandgrad = tf_util.function( [ observation, old_policy.obs_ph, ret ],
					                                               tf_util.flatgrad( vferr, vf_var_list ) )

					@contextmanager
					def timed( msg ):
						if self.rank == 0 and self.verbose >= 1:
							print( colorize( msg, color='magenta' ) )
							start_time = time.time()
							yield
							print( colorize( "done in {:.3f} seconds".format( (time.time() - start_time) ),
							                 color='magenta' ) )
						else:
							yield

					def allmean( arr ):
						assert isinstance( arr, np.ndarray )
						out = np.empty_like( arr )
						MPI.COMM_WORLD.Allreduce( arr, out, op=MPI.SUM )
						out /= self.nworkers
						return out

					tf_util.initialize( sess=self.sess )

					th_init = self.get_flat()
					MPI.COMM_WORLD.Bcast( th_init, root=0 )
					self.set_from_flat( th_init )

				with tf.variable_scope( "Adam_mpi", reuse=False ):
					self.vfadam = MpiAdam( vf_var_list, sess=self.sess )

					self.vfadam.sync()

				with tf.variable_scope( "input_info", reuse=False ):
					tf.summary.scalar( 'discounted_rewards', tf.reduce_mean( ret ) )
					tf.summary.scalar( 'learning_rate', tf.reduce_mean( self.vf_stepsize ) )
					tf.summary.scalar( 'advantage', tf.reduce_mean( atarg ) )
					tf.summary.scalar( 'kl_clip_range', tf.reduce_mean( self.max_kl ) )

					if self.full_tensorboard_log:
						tf.summary.histogram( 'discounted_rewards', ret )
						tf.summary.histogram( 'learning_rate', self.vf_stepsize )
						tf.summary.histogram( 'advantage', atarg )
						tf.summary.histogram( 'kl_clip_range', self.max_kl )
						if tf_util.is_image( self.observation_space ):
							tf.summary.image( 'observation', observation )
						else:
							tf.summary.histogram( 'observation', observation )

				self.timed = timed
				self.allmean = allmean

				self.step = self.policy_pi.step
				self.proba_step = self.policy_pi.proba_step
				self.initial_state = self.policy_pi.initial_state

				self.params = tf_util.get_trainable_vars( "model" ) + tf_util.get_trainable_vars( "oldpi" )

				self.summary = tf.summary.merge_all()

				self.compute_lossandgrad = \
					tf_util.function( [ observation, old_policy.obs_ph, action, atarg, ret ],
					                  [ self.summary, tf_util.flatgrad( optimgain, var_list ) ] + losses )



	def learn( self, total_timesteps, callback=None, log_interval=100, tb_log_name="TRPO",
	           reset_num_timesteps=True ):

		new_tb_log = self._init_num_timesteps( reset_num_timesteps )
		callback = self._init_callback( callback )

		with SetVerbosity( self.verbose ), TensorboardWriter( self.graph, self.tensorboard_log, tb_log_name, new_tb_log ) 				as writer:
			self._setup_learn()

			with self.sess.as_default():
				callback.on_training_start( locals(), globals() )

				seg_gen = traj_segment_generator( self.policy_pi, self.env, self.timesteps_per_batch,callback=callback )

				episodes_so_far = 0
				timesteps_so_far = 0
				iters_so_far = 0
				t_start = time.time()
				len_buffer = deque( maxlen=40 )  # rolling buffer for episode lengths
				reward_buffer = deque( maxlen=40 )  # rolling buffer for episode rewards

				while True:
					if timesteps_so_far >= total_timesteps:
						break

					logger.log( "********** Iteration %i ************" % iters_so_far )

					def fisher_vector_product( vec ):
						return self.allmean( self.compute_fvp( vec, *fvpargs, sess=self.sess ) ) + self.cg_damping * vec

					# ------------------ Update G ------------------
					logger.log( "Optimizing Policy..." )
					# g_step = 1 when not using GAIL
					mean_losses = None
					vpredbefore = None
					tdlamret = None
					observation = None
					action = None
					seg = None
					for k in range( self.g_step ):
						with self.timed( "sampling" ):
							seg = seg_gen.__next__()

						# Stop training early (triggered by the callback)
						if not seg.get( 'continue_training', True ):  # pytype: disable=attribute-error
							break

						add_vtarg_and_adv( seg, self.gamma, self.lam )
						# ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
						observation, action = seg[ "observations" ], seg[ "actions" ]
						atarg, tdlamret = seg[ "adv" ], seg[ "tdlamret" ]

						vpredbefore = seg[ "vpred" ]  # predicted value function before update
						atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-8)  # standardized advantage function estimate


						print( 'advantages: ', np.min(atarg), np.max(atarg), np.mean(atarg))
						# true_rew is the reward without discount
						if writer is not None:
							total_episode_reward_logger( self.episode_reward,
							                             seg[ "true_rewards" ].reshape(
								                             (self.n_envs, -1) ),
							                             seg[ "dones" ].reshape( (self.n_envs, -1) ),
							                             writer, self.num_timesteps )

						args = seg[ "observations" ], seg[ "observations" ], seg[ "actions" ], atarg
						# Subsampling: see p40-42 of John Schulman thesis
						# http://joschu.net/docs/thesis.pdf
						fvpargs = [ arr[ ::5 ] for arr in args ]

						self.assign_old_eq_new( sess=self.sess )

						with self.timed( "computegrad" ):
							steps = self.num_timesteps + (k + 1) * (seg[ "total_timestep" ] / self.g_step)
							run_options = tf.RunOptions( trace_level=tf.RunOptions.FULL_TRACE )
							run_metadata = tf.RunMetadata() if self.full_tensorboard_log else None





							_, grad, *lossbefore = self.compute_lossandgrad( *args, tdlamret, sess=self.sess,
								                                                 options=run_options,
								                                                 run_metadata=run_metadata )

						print( f'losses before', lossbefore)
						lossbefore = self.allmean( np.array( lossbefore ) )
						grad = self.allmean( grad )
						if np.allclose( grad, 0 ):
							logger.log( "Got zero gradient. not updating" )
						else:
							with self.timed( "conjugate_gradient" ):
								stepdir = conjugate_gradient( fisher_vector_product, grad, cg_iters=self.cg_iters,
								                              verbose=self.rank == 0 and self.verbose >= 1 )
							assert np.isfinite( stepdir ).all()
							shs = .5 * stepdir.dot( fisher_vector_product( stepdir ) )
							# abs(shs) to avoid taking square root of negative values
							lagrange_multiplier = np.sqrt( abs( shs ) / self.max_kl )
							# logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
							fullstep = stepdir / lagrange_multiplier
							expectedimprove = grad.dot( fullstep )
							surrbefore = lossbefore[ 0 ]
							stepsize = 1.0
							thbefore = self.get_flat()
							for _ in range( 10 ):
								thnew = thbefore + fullstep * stepsize
								self.set_from_flat( thnew )
								mean_losses = surr, kl_loss, *_ = self.allmean(
									np.array( self.compute_losses( *args, sess=self.sess ) ) )
								improve = surr - surrbefore
								logger.log( "Expected: %.3f Actual: %.3f" % (expectedimprove, improve) )
								if not np.isfinite( mean_losses ).all():
									logger.log( "Got non-finite value of losses -- bad!" )
								elif kl_loss > self.max_kl * 1.5:
									logger.log( "violated KL constraint. shrinking step." )
								elif improve < 0:
									logger.log( "surrogate didn't improve. shrinking step." )
								else:
									logger.log( "Stepsize OK!" )
									break
								stepsize *= .5
							else:
								logger.log( "couldn't compute a good step" )
								self.set_from_flat( thbefore )
							if self.nworkers > 1 and iters_so_far % 20 == 0:
								# list of tuples
								paramsums = MPI.COMM_WORLD.allgather( (thnew.sum(), self.vfadam.getflat().sum()) )
								assert all( np.allclose( ps, paramsums[ 0 ] ) for ps in paramsums[ 1: ] )

							for (loss_name, loss_val) in zip( self.loss_names, mean_losses ):
								logger.record_tabular( loss_name, loss_val )

						with self.timed( "vf" ):
							for _ in range( self.vf_iters ):
								# NOTE: for recurrent policies, use shuffle=False?
								for (mbob, mbret) in dataset.iterbatches( (seg[ "observations" ], seg[ "tdlamret" ]),
								                                          include_final_partial_batch=False,
								                                          batch_size=128,
								                                          shuffle=True ):
									grad = self.allmean( self.compute_vflossandgrad( mbob, mbob, mbret, sess=self.sess ) )
									self.vfadam.update( grad, self.vf_stepsize )

					# Stop training early (triggered by the callback)
					if not seg.get( 'continue_training', True ):  # pytype: disable=attribute-error
						break

					logger.record_tabular( "explained_variance_tdlam_before",
					                       explained_variance( vpredbefore, tdlamret ) )

					# lr: lengths and rewards
					lr_local = (seg[ "ep_lens" ], seg[ "ep_rets" ])  # local values
					list_lr_pairs = MPI.COMM_WORLD.allgather( lr_local )  # list of tuples
					lens, rews = map( flatten_lists, zip( *list_lr_pairs ) )

					len_buffer.extend( lens )
					reward_buffer.extend( rews )

					if len( len_buffer ) > 0:
						logger.record_tabular( "EpLenMean", np.mean( len_buffer ) )
						logger.record_tabular( "EpRewMean", np.mean( reward_buffer ) )

					logger.record_tabular( "EpThisIter", len( lens ) )
					episodes_so_far += len( lens )
					current_it_timesteps = MPI.COMM_WORLD.allreduce( seg[ "total_timestep" ] )
					timesteps_so_far += current_it_timesteps
					self.num_timesteps += current_it_timesteps
					iters_so_far += 1

					logger.record_tabular( "EpisodesSoFar", episodes_so_far )
					logger.record_tabular( "TimestepsSoFar", self.num_timesteps )
					logger.record_tabular( "TimeElapsed", time.time() - t_start )

					if self.verbose >= 1 and self.rank == 0:
						logger.dump_tabular()

		callback.on_training_end()
		return self

	def save( self, save_path, cloudpickle=False ):
		pass


if __name__ == '__main__':

	# from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution
	# np.random.seed(1)
	#
	# ppp = np.arange(1,1024*10000+1,1).reshape(10000,1024).astype(np.float32) #np.random.rand(10000,1024).astype(np.float32)
	# prb = DiagGaussianProbabilityDistribution( ppp)
	# x = prb.sample()
	# b = prb.logp(x)
	#
	# c =tf.reduce_mean(		prb.entropy())
	# d = tf.reduce_mean(-b)
	# e = tf.math.reduce_std(b)
	# with tf.Session() as session:
	# 	c = session.run(tf.reduce_mean(		prb.entropy()))
	# 	print( c)
	# 	d = session.run( d)
	# 	print( d)
	# 	print( c-d)
	# 	e= session.run(e)
	# 	print( e)
	#
	# exit()


	env = gym.make('CartPole-v1')
	env = gym.make( 'Pendulum-v0')

	model = TRPO(MlpPolicy, env, verbose=1)
	model.learn(total_timesteps=250000)
	#model.save("trpo_cartpole")
	#del model # remove to demonstrate saving and loading
	#model = TRPO.load("trpo_cartpole")

	obs = env.reset()
	steps=0
	while True:
		action, _states = model.predict(obs)
		steps += 1
		obs, rewards, dones, info = env.step(action)
		env.render()

	exit()

# 纯TRPO没GAIL
# class AbstractEnvRunner( ABC ):
# 	def __init__( self, *, env: Union[ gym.Env, VecEnv ], model: 'BaseRLModel', n_steps: int ):
# 		"""
# 		Collect experience by running `n_steps` in the environment.
# 		Note: if this is a `VecEnv`, the total number of steps will
# 		be `n_steps * n_envs`.
#
# 		:param env: (Union[gym.Env, VecEnv]) The environment to learn from
# 		:param model: (BaseRLModel) The model to learn
# 		:param n_steps: (int) The number of steps to run for each environment
# 		"""
# 		self.env = env
# 		self.model = model
# 		n_envs = env.num_envs
# 		self.batch_ob_shape = (n_envs * n_steps,) + env.observation_space.shape
# 		self.obs = np.zeros( (n_envs,) + env.observation_space.shape, dtype=env.observation_space.dtype.name )
# 		self.obs[ : ] = env.reset()
# 		self.n_steps = n_steps
# 		self.states = model.initial_state
# 		self.dones = [ False for _ in range( n_envs ) ]
# 		self.callback = None  # type: Optional[BaseCallback]
# 		self.continue_training = True
# 		self.n_envs = n_envs
#
# 	def run( self, callback: Optional[ BaseCallback ] = None ) -> Any:
# 		"""
# 		Collect experience.
#
# 		:param callback: (Optional[BaseCallback]) The callback that will be called
# 			at each environment step.
# 		"""
# 		self.callback = callback
# 		self.continue_training = True
# 		return self._run()
#
# 	@abstractmethod
# 	def _run( self ) -> Any:
# 		"""
# 		This method must be overwritten by child class.
# 		"""
# 		raise NotImplementedError


# , gail=False
# 	"""
# 	Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
# 	:param policy: (MLPPolicy) the policy
# 	:param env: (Gym Environment) the environment
# 	:param horizon: (int) the number of timesteps to run per batch
# 	:param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
# 	:param gail: (bool) Whether we are using this generator for standard trpo or with gail
# 	:param callback: (BaseCallback)
# 	:return: (dict) generator that returns a dict with the following keys:
# 		- observations: (np.ndarray) observations
# 		- rewards: (numpy float) rewards (if gail is used it is the predicted reward)
# 		- true_rewards: (numpy float) if gail is used it is the original reward
# 		- vpred: (numpy float) action logits
# 		- dones: (numpy bool) dones (is end of episode, used for logging)
# 		- episode_starts: (numpy bool)
# 			True if first timestep of an episode, used for GAE
# 		- actions: (np.ndarray) actions
# 		- nextvpred: (numpy float) next action logits
# 		- ep_rets: (float) cumulated current episode reward
# 		- ep_lens: (int) the length of the current episode
# 		- ep_true_rets: (float) the real environment reward
# 		- continue_training: (bool) Whether to continue training
# 			or stop early (triggered by the callback)
# 	"""
# 	# Check when using GAIL
# 	assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"
#


# 	if gail:
# 		reward = reward_giver.get_reward( observation, clipped_action[ 0 ] )
# 		observation, true_reward, done, info = env.step( clipped_action[ 0 ] )
# 	else:
#	if not gail: reward_giver=None,
# true_rewards = np.zeros( horizon, 'float32' )
#	"true_rewards"     : true_rewards,
#	"true_rewards"     : true_rewards,
#	true_reward = reward
#	true_rewards[ i ] = true_reward
#	cur_ep_true_ret = 0
#	ep_true_rets = [ ]
#	"ep_true_rets"     : ep_true_rets,
#	"ep_true_rets"     : ep_true_rets,
#	ep_true_rets = [ ]
#	ep_true_rets.append( cur_ep_true_ret )
#	cur_ep_true_ret = 0
#	cur_ep_true_ret = maybe_ep_info[ 'r' ]


#
#
# data = {
# 	"gamma": self.gamma,
# 	"timesteps_per_batch": self.timesteps_per_batch,
# 	"max_kl": self.max_kl,
# 	"cg_iters": self.cg_iters,
# 	"lam": self.lam,
# 	"entcoeff": self.entcoeff,
# 	"cg_damping": self.cg_damping,
# 	"vf_stepsize": self.vf_stepsize,
# 	"vf_iters": self.vf_iters,
# 	"hidden_size_adversary": self.hidden_size_adversary,
# 	"adversary_entcoeff": self.adversary_entcoeff,
# 	"expert_dataset": self.expert_dataset,
# 	"g_step": self.g_step,
# 	"d_step": self.d_step,
# 	"d_stepsize": self.d_stepsize,
# 	"using_gail": self.using_gail,
# 	"verbose": self.verbose,
# 	"policy": self.policy,
# 	"observation_space": self.observation_space,
# 	"action_space": self.action_space,
# 	"n_envs": self.n_envs,
# 	"n_cpu_tf_sess": self.n_cpu_tf_sess,
# 	"seed": self.seed,
# 	"_vectorize_action": self._vectorize_action,
# 	"policy_kwargs": self.policy_kwargs
# }
#
# params_to_save = self.get_parameters()
#
# self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

# from stable_baselines.common.runners import traj_segment_generator
# self.using_gail = False
# from stable_baselines.trpo_mpi.utils import add_vtarg_and_adv
#
# 	if self.using_gail:
# 		self.reward_giver = TransitionClassifier( self.observation_space, self.action_space, self.hidden_size_adversary, entcoeff=self.adversary_entcoeff )

#
# if self.using_gail:
# 	self.d_adam = MpiAdam( self.reward_giver.get_trainable_variables(), sess=self.sess )
# 	self.d_adam.sync()
#
# if self.using_gail:
# 	self.params.extend( self.reward_giver.get_trainable_variables() )


#	reward_giver=self.reward_giver,
#	gail=self.using_gail,


#
#
#
#
#
#
#
#
#
# true_reward_buffer = None
# if self.using_gail:
# 	true_reward_buffer = deque( maxlen=40 )
#
# 	self._initialize_dataloader()

#  Stats not used for now
# TODO: replace with normal tb logging
#  g_loss_stats = Stats(loss_names)
#  d_loss_stats = Stats(reward_giver.loss_name)
#  ep_stats = Stats(["True_rewards", "Rewards", "Episode_length"])
#
#
#
#
#
#
#
#
#

#
# if self.using_gail:
# 	logger.record_tabular( "EpTrueRewMean", np.mean( true_reward_buffer ) )
#
# 	if self.using_gail:
# 	# ------------------ Update D ------------------
# 	logger.log( "Optimizing Discriminator..." )
# 	logger.log( fmt_row( 13, self.reward_giver.loss_name ) )
# 	assert len( observation ) == self.timesteps_per_batch
# 	batch_size = self.timesteps_per_batch // self.d_step
#
# 	# NOTE: uses only the last g step for observation
# 	d_losses = [ ]  # list of tuples, each of which gives the loss for a minibatch
# 	# NOTE: for recurrent policies, use shuffle=False?
# 	for ob_batch, ac_batch in dataset.iterbatches( (observation, action),
# 	                                               include_final_partial_batch=False,
# 	                                               batch_size=batch_size,
# 	                                               shuffle=True ):
# 		ob_expert, ac_expert = self.expert_dataset.get_next_batch()
# 		# update running mean/std for reward_giver
# 		if self.reward_giver.normalize:
# 			self.reward_giver.obs_rms.update( np.concatenate( (ob_batch, ob_expert), 0 ) )
#
# 		# Reshape actions if needed when using discrete actions
# 		if isinstance( self.action_space, gym.spaces.Discrete ):
# 			if len( ac_batch.shape ) == 2:
# 				ac_batch = ac_batch[ :, 0 ]
# 			if len( ac_expert.shape ) == 2:
# 				ac_expert = ac_expert[ :, 0 ]
# 		*newlosses, grad = self.reward_giver.lossandgrad( ob_batch, ac_batch, ob_expert, ac_expert )
# 		self.d_adam.update( self.allmean( grad ), self.d_stepsize )
# 		d_losses.append( newlosses )
# 	logger.log( fmt_row( 13, np.mean( d_losses, axis=0 ) ) )
#
# 	# lr: lengths and rewards
# 	lr_local = (seg[ "ep_lens" ], seg[ "ep_rets" ], seg[ "ep_true_rets" ])  # local values
# 	list_lr_pairs = MPI.COMM_WORLD.allgather( lr_local )  # list of tuples
# 	lens, rews, true_rets = map( flatten_lists, zip( *list_lr_pairs ) )
# 	true_reward_buffer.extend( true_rets )
# else:
# 	def _initialize_dataloader( self ):
# 		"""Initialize dataloader."""
# 		batchsize = self.timesteps_per_batch // self.d_step
# 		self.expert_dataset.init_dataloader( batchsize )
# 	"""
# 	Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477)
#
# 	:param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
# 	:param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
# 	:param gamma: (float) the discount value
# 	:param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
# 	:param max_kl: (float) the Kullback-Leibler loss threshold
# 	:param cg_iters: (int) the number of iterations for the conjugate gradient calculation
# 	:param lam: (float) GAE factor
# 	:param entcoeff: (float) the weight for the entropy loss
# 	:param cg_damping: (float) the compute gradient dampening factor
# 	:param vf_stepsize: (float) the value function stepsize
# 	:param vf_iters: (int) the value function's number iterations for learning
# 	:param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
# 	:param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
# 	:param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
# 	:param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
# 	:param full_tensorboard_log: (bool) enable additional logging when using tensorboard
# 		WARNING: this logging can take a lot of space quickly
# 	:param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
# 		If None (default), use random seed. Note that if you want completely deterministic
# 		results, you must set `n_cpu_tf_sess` to 1.
# 	:param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
# 		If None, the number of cpu of the current machine will be used.
# 	"""

	#
	# # run loss backprop with summary, and save the metadata (memory, compute time, ...)
	# if writer is not None:
	# 	summary, grad, *lossbefore = self.compute_lossandgrad( *args, tdlamret, sess=self.sess,
	# 	                                                       options=run_options,
	# 	                                                       run_metadata=run_metadata )
	# 	if self.full_tensorboard_log:
	# 		writer.add_run_metadata( run_metadata, 'step%d' % steps )
	# 	writer.add_summary( summary, steps )
	# else:
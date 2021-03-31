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
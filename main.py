import logging
import tensorflow as tf
import numpy as np
import gym
import gflags
import sys
import time
from needle.agents import find_agent
from needle.adaptors import find_adaptor

gflags.DEFINE_string("mode", "infer", "inference or training (default infer)")
gflags.DEFINE_integer("save_step", 2000, "how many steps between saving")
gflags.DEFINE_float("gamma", 0.99, "value discount per step")
gflags.DEFINE_string("model_dir", "", "directory to save models")
gflags.DEFINE_string("log_dir", "", "directory to save logs")
gflags.DEFINE_boolean("train_without_init", False, "initialize all variables when training")
gflags.DEFINE_string("monitor", "", "path to save recordings")
gflags.DEFINE_integer("iterations", 100000, "# iterations to run")
gflags.DEFINE_float("learning_rate", 1e-3, "learning rate")
gflags.DEFINE_boolean("verbose", False, "to show all log")

FLAGS = gflags.FLAGS



FLAGS.mode='train'
FLAGS.env = 'CartPole-v0'
FLAGS.agent = 'TRPO'
#FLAGS.verbose = True
#FLAGS.iterations = 50

def main():
    np.set_printoptions(linewidth=1000)
    if FLAGS.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    env = gym.make(FLAGS.env)
    if FLAGS.monitor != "":
        env.monitor.start(FLAGS.monitor)
    # logging.warning("action space: %s, %s, %s" % (env.action_space, env.action_space.high, env.action_space.low))

    logging.warning("Making new agent: %s" % (FLAGS.agent,))
    adaptor = find_adaptor()(env)
    agent = find_agent()(adaptor.input_dim, adaptor.output_dim)

    op_rewards = tf.placeholder(tf.float32)
    tf.summary.scalar("rewards", op_rewards)

    saver = tf.train.Saver()
    if FLAGS.mode == "train" or not FLAGS.train_without_init or FLAGS.model_dir == "":
        logging.info("Initializing variables...")
        agent.init()
    else:
        logging.info("Restore variables...")
        saver.restore(tf.get_default_session(), FLAGS.model_dir)

    merged = tf.summary.merge_all()
    if FLAGS.log_dir != "" and tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir) #tf.train.SummaryWriter(FLAGS.log_dir)

    for iterations in range(FLAGS.iterations):
        print( 'iterations', iterations)
        if not FLAGS.verbose and iterations % FLAGS.batch_size == 0:
            logging.root.setLevel(logging.DEBUG)
        agent.reset()
        adaptor.reset()
        state = env.reset()

        model_state = adaptor.state(state)

        done = False
        total_rewards = 0
        steps = 0

        while not done and steps < env.spec.max_episode_steps:# .timestep_limit:
            steps += 1

            model_action = agent.action(model_state)
            action = adaptor.to_env(model_action)
            # logging.warning("action = %s" % (action))
            # if steps % 100 == 0:
            #     logging.warning(action[0])
            new_state, reward, done, info = env.step(action)
            # logging.debug("state = %s, action = %s, reward = %s" % (model_state, action, reward))
            if steps == env.spec.max_episode_steps:#.timestep_limit:
                done = False

            model_new_state = adaptor.state(new_state)
            agent.feedback(model_state, model_action, reward, done, model_new_state)
            model_state = model_new_state

            total_rewards += reward
            if iterations % 10 == 0 and steps % 1 == 0:# and FLAGS.mode == "infer":
                #time.sleep(1)
                env.render()
                # logging.warning("step: #%d, action = %.3f, reward = %.3f, iteration = %d" % (steps, action[0], reward, iterations))
            # if episode == 0:
            #     print observation, action, info

        # if iterations % args.batch_size == 0:
        if FLAGS.mode == "train":
            agent.train(done)

        summary = tf.get_default_session().run(merged, feed_dict={
            op_rewards: total_rewards,
        })
        summary_writer.add_summary(summary, iterations)

        # logging.info("iteration #%4d: total rewards = %.3f" % (iterations, total_rewards))
        if not FLAGS.verbose and iterations % FLAGS.batch_size == 0:
            logging.root.setLevel(logging.INFO)

        if iterations % FLAGS.save_step == 0 and FLAGS.model_dir != "":
            saver.save(tf.get_default_session(), FLAGS.model_dir)

    if FLAGS.monitor != "":
        env.monitor.close()











def tst():
    def _init_openmpi():
        """Pre-load libmpi.dll and register OpenMPI distribution."""
        import os
        import ctypes
        if os.name != 'nt' or 'OPENMPI_HOME' in os.environ:
            return
        try:
            openmpi_home = os.path.abspath(os.path.dirname(__file__))
            openmpi_bin = os.path.join(openmpi_home, 'bin')
            os.environ['OPENMPI_HOME'] = openmpi_home
            os.environ['PATH'] = ';'.join((openmpi_bin, os.environ['PATH']))
            ctypes.cdll.LoadLibrary(os.path.join(openmpi_bin, 'libmpi.dll'))
        except Exception:
            pass

    _init_openmpi()

    import gym

    from stable_baselines.common.policies import MlpPolicy, CnnPolicy
    from stable_baselines import TRPO

    env = gym.make('BreakoutNoFrameskip-v4')#'CartPole-v1')

    model = TRPO(CnnPolicy, env, timesteps_per_batch=1024, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("trpo_cartpole")

    del model # remove to demonstrate saving and loading

    model = TRPO.load("trpo_cartpole")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == "__main__":


    tst()
    exit()




    FLAGS(sys.argv)
    with tf.Session().as_default():
        main()

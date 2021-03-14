import gflags
import logging
import numpy as np

from needle.agents import BasicAgent, register_agent
from needle.agents.TNPG.net import Net
from needle.agents.TRPO.critic import Critic
from needle.helper.conjugate_gradient import conjugate_gradient
from needle.helper.OU_process import OUProcess
from needle.helper.softmax_sampler import SoftmaxSampler
from needle.helper.batcher import Batcher
from needle.helper.utils import decay_cumsum, softmax

gflags.DEFINE_float("GAE_lambda", 0.98, "GAE lambda")
gflags.DEFINE_float("critic_eps", 0.01, "critic's trust region")
gflags.DEFINE_float("line_search_decay", 0.7, "line search's decay factor")
FLAGS = gflags.FLAGS


@register_agent("TRPO")
class Agent(SoftmaxSampler, Batcher, BasicAgent):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.counter = 0

        self.net = Net(input_dim, output_dim)
        self.net.build_infer()
        self.net.build_train()

        self.critic = Critic(input_dim)
        self.critic.build_infer()
        self.critic.build_train()

        self.batch_mode = "step"

    def compute_advantage(self, values, mask, rewards):
        advantages = FLAGS.gamma * values[:, 1:] + rewards - values[:, :-1]
        advantages = decay_cumsum(advantages[:, ::-1], FLAGS.gamma * FLAGS.GAE_lambda)[:, ::-1]
        return advantages

    def train_actor(self, values, lengths, mask, states, choices, rewards):
        # advantages = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
        advantages = self.compute_advantage(values, mask, rewards)
        # logging.info("GAE = %s" % (advantages.shape,))


        old_logits = self.net.infer(states)
        old_actions = softmax(old_logits)
        feed_dict = self.net.get_dict(lengths, mask, states, choices, advantages, old_logits)
        gradient = self.net.gradient(feed_dict)

        natural_gradient, dot_prod = conjugate_gradient(
            lambda direction: self.net.fisher_vector_product(direction, feed_dict),
            gradient,
        )
        natural_gradient *= np.sqrt(2 * FLAGS.delta_KL / (dot_prod + 1e-8))
        variables = self.net.get_variables()

        old_loss, old_KL = self.net.test(feed_dict)
        logging.info("old loss = %s, old KL = %s" % (old_loss, np.mean(old_KL)))

        while True:
            self.net.apply_var(variables - natural_gradient)
            new_loss, new_KL = self.net.test(feed_dict)
            logging.info("new loss = %s, new KL = %s" % (new_loss, np.mean(new_KL)))
            if new_KL - old_KL <= FLAGS.delta_KL and new_loss <= old_loss:
                break
            natural_gradient *= FLAGS.line_search_decay

    def train_critic(self, lengths, values, states, rewards):
        total_rewards = decay_cumsum(rewards[:, ::-1], FLAGS.gamma)[:, ::-1]

        # since TF doesn't support computing Jacobian, we have to compute it one by one
        # shame on the following code
        gradients = []
        deltas = []
        for i in range(states.shape[0]):
            for j in range(lengths[i]):
                gradients.append(self.critic.gradient({self.critic.op_inputs: states[i:i + 1, j:j + 1]}))
                deltas.append(values[i, j] - total_rewards[i, j])
        jacobian = np.vstack(gradients) # (batchSize,thetaSize)
        deltas = np.array(deltas) # (batchSize,)
        rank = len(deltas) # batchSize

        logging.info("values = %s, rewards = %s max/min-total-rewards = %s %s" % (values[0, :3], total_rewards[0, :3], np.max(total_rewards), np.min(total_rewards)))
        # logging.info("variance: %s, deltas = %s" % (variance, deltas))
        # logging.info("rewards = %s" % (total_rewards,))
        variance = np.var(deltas)


        def fAx(direction):
            a = jacobian.dot(direction)
            b = jacobian.T.dot(a)
            c = b / variance / rank
            return c

        theB = jacobian.T.dot(deltas) / rank




        natural_gradient, dot_prod = conjugate_gradient(
            fAx, theB
            #lambda direction: jacobian.T.dot(jacobian.dot(direction)) / variance / rank,
            #jacobian.T.dot(deltas) / rank,
        )

        beta = np.sqrt(2 * FLAGS.critic_eps / (dot_prod + 1e-8))
        print( 'the beta', beta)
        natural_gradient *= beta
        self.critic.apply_grad(natural_gradient)

    def train_batch(self, lengths, mask, states, choices, rewards, new_states):
        # logging.info("lengths = %s, new states = %s" % (lengths, new_states))
        # logging.info("concat: %s" % (np.concatenate([states, new_states[:, lengths - 1, :]], axis=1),))
        values = self.critic.infer(states)
        # logging.info("states = %s" % (states,))

        batch_size = values.shape[0]
        values = np.concatenate([values * mask, np.zeros((batch_size, 1))], axis=1)

        self.train_actor(values, lengths, mask, states, choices, rewards)
        self.train_critic(lengths, values, states, rewards)

    def action(self, inputs):
        return self.softmax_action(
            self.net.infer(np.array([inputs])),
            noise=self.noise,
        )

    def reset(self):
        self.noise = OUProcess()
        self.net.reset()
        self.counter = 0

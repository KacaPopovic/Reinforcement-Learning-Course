import numpy as np
from gridworld import GridWorldEnv
from util import value_function_policy_plot


class TDAgent:
    def __init__(self, env, discount_factor, learning_rate):
        self.env = env
        self.g = discount_factor
        self.lr = learning_rate

        self.num_actions = env.action_space.n

        # V[y, x] is value for grid position y, x, initialize to all zeros
        self.V = np.zeros(env.observation_space.high, dtype=np.float32)

        # uniform random policy[y, x, z], i.e. probability of action z when in grid position y, x is 1 / num_actions
        self.policy = np.ones((*env.observation_space.high, self.num_actions), dtype=np.float32) / self.num_actions

    def action(self, s):
        # Sample action following the policy
        action_probabilities = self.policy[s[0], s[1]]
        a = np.random.choice(np.arange(self.num_actions), p=action_probabilities)
        return a

    def learn(self, n_timesteps=50000):
        s, _ = self.env.reset()

        for i in range(n_timesteps):
            a = self.action(s)
            s_, r, done, _, _ = self.env.step(a)
            if done:
                s_, _ = self.env.reset()
            self.update(s, r, s_)
            s = s_

    def update(self, s, r, s_):
        # TD estimation update rule
        self.V[s[0], s[1]] += self.lr * (r + self.g * self.V[s_[0], s_[1]] - self.V[s[0], s[1]])

        # Policy update to be greedy with respect to the value function
        best_action = np.argmax([self.V[s_[0], s_[1]] for _ in range(self.num_actions)])
        self.policy[s[0], s[1]] = np.eye(self.num_actions)[best_action]


if __name__ == "__main__":
    # Create Agent and environment
    td_agent = TDAgent(GridWorldEnv(), discount_factor=0.9, learning_rate=0.01)

    # Learn the state-value function for 100000 steps
    td_agent.learn(n_timesteps=100000)

    # Visualize V
    value_function_policy_plot(td_agent.V, td_agent.policy, td_agent.env.map)

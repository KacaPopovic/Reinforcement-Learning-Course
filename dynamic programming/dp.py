import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy

def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    policy[x][y] is the probability of action with y for state x.
    mdp is GridworlsMDP shape : list of int
    mdp.P[s][a] contains a list of tuples (p, s', r, is_terminal) with:
    """
    # Init value function array
    V_new = init_value(mdp)
    for state in range(mdp.num_states):
        V_new[state] = sum(
            policy[state][action] * sum(p * (r + discount * V[s_new]) for p, s_new, r, _ in mdp.P[state][action]) \
            for action in range(mdp.num_actions))
    return V_new

def policy_evaluation(mdp, policy, discount=0.99, theta=0.01):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here
    delta = np.inf
    while delta > theta:
        delta = 0
        V_new = policy_evaluation_one_step(mdp, V, policy, discount)
        delta = max(delta, np.amax(np.abs(V_new - V)))
        V = V_new
    return V

def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros_like(random_policy(mdp))
    policy_stable = True

    # TODO: Write your implementation here
    for state in range(mdp.num_states):
        max_action = np.argmax([sum(p * (r + discount * V[s_new]) for p, s_new, r, _ in mdp.P[state][action]) \
                                   for action in range(mdp.num_actions)])
        policy[state] = np.zeros(mdp.num_actions)
        policy[state, max_action] = 1
    return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """

    # Start from random policy
    policy = random_policy(mdp)
    # This is only here for the skeleton to run.
    V = init_value(mdp)

    # TODO: Write your implementation here
    policy_stable = False
    while policy_stable == False:
        V = policy_evaluation(mdp, policy, discount, theta)
        policy_stable = True
        new_policy = policy_improvement(mdp, V, discount)
        if not np.array_equal(new_policy, policy):
            policy_stable = False
        policy = new_policy
    return V, policy

def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    policy = random_policy(mdp)
    # This is only here for the skeleton to run.
    V = init_value(mdp)

    V = policy_evaluation(mdp, policy, discount, theta)

    policy = policy_improvement(mdp, V, discount)
    
    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    # Do one step of policy evaluation and print
    print('----------------------------------------------')
    print('One step of policy evaluation (random policy):')
    print('----------------------------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)
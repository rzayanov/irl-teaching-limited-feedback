"""
Evaluate the optimal policy of an MDP.
"""
import copy
from typing import Optional, Tuple

from const import *
from dimensions import *
from env import Env

CUMUL_ITERS = [0, 0, 0]


def value_iteration(env: Env, reward: Reward, init_V: Vs, eps: float = 1e-6) -> Tuple[Vs, Qs, Policy]:
    """
    @brief: Calculate optimal policy and corresponding optimal value function.
    @param env: environment class
    @param eps: threshold
    @return policy: optimal policy
    @return V: state value function
    @return Q: state-action value function
    """
    V = init_V  # np.zeros(env.n_states)
    Q = np.zeros((env.n_states, env.n_actions))

    iteration = 0
    V_prev = np.zeros(env.n_states)
    while True:
        np.copyto(V_prev, V)
        for a in range(env.n_actions):
            Q[:, a] = reward + env.gamma * env.T[a].dot(V)

        V = np.max(Q, axis=1)
        if (abs(np.linalg.norm(V - V_prev, np.inf)) < eps):
            break
        iteration += 1

    CUMUL_ITERS[0] += iteration
    # print(f"value iters: {iteration}")

    policy = Q - np.max(Q, axis=1)[:, None]
    policy[np.where((-1e-12 <= policy) & (policy <= 1e-12))] = 1
    policy[np.where(policy <= 0)] = 0
    policy = policy / policy.sum(axis=1)[:, None]

    return V, Q, policy


def value_iteration_soft(env: Env, reward: Reward, init_V: Vs, eps: float = 1e-6) -> Tuple[Vs, Qs, Policy]:
    """
    @brief: soft-value-iteration function.
    """
    V = init_V  # np.zeros(env.n_states)
    Q = np.zeros((env.n_states, env.n_actions))

    iteration = 0
    V_prev = np.zeros(env.n_states)
    while True:
        np.copyto(V_prev, V)
        for a in range(env.n_actions):
            Q[:, a] = reward + env.gamma * env.T[a].dot(V)

        V = softmax(Q, env.n_states)
        if (abs(np.linalg.norm(V - V_prev, np.inf)) < eps):
            break
        iteration += 1

    CUMUL_ITERS[1] += iteration
    # print(f"soft value iters: {iteration}")

    # Q_copy = Q.copy()
    # Q_copy -= Q.max(axis=1).reshape((env.n_states, 1))  # For numerical stability
    # policy = np.exp(ALPHA_I * Q_copy) / np.exp(ALPHA_I * Q_copy).sum(axis=1).reshape((env.n_states, 1))

    policy = np.exp(ALPHA_I * (Q - V.reshape((env.n_states, 1))))

    return V, Q, policy


def softmax(Q: Qs, states: int) -> NDArray['S']:
    Amax = Q.max(axis=1)
    Qmod = Q - Amax.reshape((states, 1))  # For numerical stability
    return Amax + ALPHA * np.log(np.exp(ALPHA_I * Qmod).sum(axis=1))


def generate_episode(
        env: Env, policy: Policy, len_episode: int, init_state: Optional[int] = None
) -> Episode:

    state = init_state
    if state is None:
        state = int(np.random.choice(env.initial_states, 1))

    episode = list()
    trans_probas_buf = np.zeros((1, env.n_states))
    for t in range(len_episode):
        action = sample_action(policy, state)
        episode.append((state, action))

        # trans_probas = env.T_dense[action, state, :]
        trans_probas_sparse = env.T[action][state, :]
        trans_probas_sparse.toarray(out=trans_probas_buf)
        trans_probas = trans_probas_buf.reshape(env.n_states)

        state = int(np.random.choice(np.arange(env.n_states), p=trans_probas))
    return episode


def calc_episode_rho(env: Env, episode: Episode) -> Rho:
    state_visitation = np.zeros(env.n_states)
    val = 1
    for t in range(len(episode)):
        s, a = episode[t]
        state_visitation[s] += val
        val *= env.gamma
    return state_visitation


def sample_action(policy: Policy, state: int) -> int:
    prob = policy[state]
    action = np.random.choice(len(prob), p=prob)
    return action


def compute_exp_rho_sampling(env: Env, policy: Policy, num_episode: int, len_episode: int, init_state: int) -> Rho:
    """
    @brief: Compute feature expectations using monte-carlo sampling.
            Initial state can be specified otherwise is randomly picked.
    """
    rho_s = np.zeros(env.n_states)
    for i in range(num_episode):
        ep = generate_episode(env, policy, len_episode, init_state)
        state_visitation = calc_episode_rho(env, ep)
        rho_s += state_visitation

    rho_s /= num_episode
    return rho_s


def compute_exp_rho_bellman(env: Env, policy: Policy, init_dist: Optional[P0] = None, eps: float = 1e-6) -> Rho:
    """
    compute expected rho
    @brief: Compute state-action visitation freq and feature expectation
            for given policy using Bellman's equation.
    """
    T_pi = env.policy_transition_matrix(policy)
    if init_dist is None:
        init_dist = env.D_init

    rho_s = np.zeros(env.n_states)
    rho_old = np.zeros(env.n_states)
    while True:
        np.copyto(rho_old, rho_s)
        rho_s = init_dist + T_pi.dot(env.gamma * rho_s)
        if np.linalg.norm(rho_s - rho_old, np.inf) < eps:
            break

    if not USE_OW:
        # todo figure out the situation with the terminal state
        rho_s[env.n_states - 1] = 0

    return rho_s


def compute_policy_value(env: Env, r: Reward, policy: Policy, init_V: Vs, eps=1e-6) -> Vs:
    T_pi = env.policy_transition_matrix(policy).transpose()
    V = init_V  # np.zeros(env.n_states)
    iteration = 0
    V_old = np.zeros(env.n_states)
    # Bellman Equation
    while True:
        np.copyto(V_old, V)
        V = r + env.gamma * T_pi.dot(V_old)
        if abs(np.linalg.norm(V - V_old, np.inf)) < eps:
            break
        iteration += 1
    CUMUL_ITERS[2] += iteration
    # print(f"policy iters: {iteration}")
    return V


def compute_policy_soft_value(env: Env, r: Reward, policy: Policy, init_V: Vs, eps=1e-6) -> Vs:
    T_pi = env.policy_transition_matrix(policy).transpose()
    V = init_V  # np.zeros(env.n_states)
    policy_for_log = policy.copy()
    policy_for_log[policy == 0] = 1
    entropy = (np.log(policy_for_log) * policy).sum(axis=1) * -1
    iteration = 0
    V_old = np.zeros(env.n_states)
    # Bellman Equation
    while True:
        np.copyto(V_old, V)
        V = r + ALPHA * entropy + env.gamma * T_pi.dot(V_old)
        if abs(np.linalg.norm(V - V_old, np.inf)) < eps:
            break
        iteration += 1
    CUMUL_ITERS[2] += iteration
    # print(f"policy iters: {iteration}")
    return V

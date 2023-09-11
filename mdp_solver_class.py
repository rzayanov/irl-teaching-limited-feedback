"""
Class definition for dynamic programming MDP solver.
"""

from typing import Optional, Tuple

import dp_solver_utils
from const import *
from dimensions import *
from env import Env
from np_utils import lock

D_INITS = {}


class solver:

    def __init__(self, env: Env, reward: Reward, dp_type: str, init_V: Vs, policy: Policy, ce_w: Weights) -> None:
        self.env: Env = env
        self.reward: Reward = reward
        lock(self.reward)
        if policy is not None:
            self.policy: Policy = policy
        else:
            if dp_type == "stochastic":
                self.V, self.Q, self.policy = self.value_iteration_soft(init_V)
            elif dp_type == "ce":
                self.V, self.Q = np.zeros(self.env.n_states), np.zeros((self.env.n_states, self.env.n_actions))
                self.policy = self.get_ce_policy(ce_w)
            else:
                self.V, self.Q, self.policy = self.value_iteration(init_V)
        lock(self.V)
        lock(self.Q)
        lock(self.policy)
        self.V_disp = self.env.get_state_array_disp(self.V) if CALC_DISP_DATA else None
        self.policy_disp = self.env.get_policy_disp(self.policy) if CALC_DISP_DATA else None

    def get_ce_policy(self, ce_w: Weights) -> Policy:
        H: NDArray['A,S'] = np.dot(self.env.action_features, ce_w[:self.env.n_features])
        if LEARNER_QUAD_MUL:
            H += np.dot(self.env.action_features, ce_w[self.env.n_features:]) ** 2
        H_max = H.max(axis=0)
        H_mod = H - H_max.reshape((1, self.env.n_states)) # For numerical stability
        H_exp: NDArray['A,S'] = np.exp(ALPHA_I * H_mod)
        H_sums: NDArray['S'] = H_exp.sum(axis=0)
        p = (H_exp / H_sums).transpose()
        return p

    def value_iteration_soft(self, init_V: Vs) -> Tuple[Vs, Qs, Policy]:
        return dp_solver_utils.value_iteration_soft(self.env, self.reward, init_V, EPS)

    def value_iteration(self, init_V: Vs) -> Tuple[Vs, Qs, Policy]:
        return dp_solver_utils.value_iteration(self.env, self.reward, init_V, EPS)

    def sample_trajectory_from_state(self, len_episode: int, state: int) -> Tuple[Episode, None]:
        # rho not currently used
        return dp_solver_utils.generate_episode(self.env, self.policy, len_episode, state), None

    def compute_exp_rho_bellman_state(self, state: int) -> Rho:
        if state in D_INITS:
            D_init = D_INITS[state]
        else:
            D_init = np.zeros(self.env.n_states)
            D_init[state] = 1
            D_INITS[state] = D_init
        return self.compute_exp_rho_bellman(D_init)

    def compute_exp_rho_bellman(self, init_dist: Optional[P0] = None) -> Rho:
        return dp_solver_utils.compute_exp_rho_bellman(self.env, self.policy, init_dist, EPS)

    def compute_exp_rho_sampling(self, num_episode: int, len_episode: int, state: int) -> Rho:
        return dp_solver_utils.compute_exp_rho_sampling(self.env, self.policy, num_episode, len_episode, state)

    def compute_policy_value(self) -> Vs:
        """ unused """
        return dp_solver_utils.compute_policy_value(self.env, self.reward, self.policy, self.V, EPS)

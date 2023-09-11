from typing import List

from scipy.sparse import csr_matrix

from dimensions import *


class Env:
    # all constant
    n_states: int
    n_actions: int
    gamma: float
    # T_dense: NDArray['A,S,S']
    T: List[csr_matrix]
    action_features: NDArray['A,S,F']
    D_init: P0
    initial_states: List[int]
    n_features: int
    feature_matrix: FMtx
    true_reward: Reward
    true_reward_disp: NDArray
    n_tasks: int

    ideal_lin_w: Weights
    ideal_nonlin_w: Weights

    # used to compute rho for car when sampling is off
    n_lanes_per_task: int
    road_length: int

    def policy_transition_matrix(self, policy: Policy) -> csr_matrix:
        pass

    def reward_for_rho(self, rho: Rho) -> float:
        pass

    def state_to_task_instance(self, state: int) -> int:
        """
        used only in teacher_curr_teacher
        state is initial
        """
        pass

    def state_to_task(self, state: int) -> int:
        """
        used only in reporting
        state is initial
        """
        pass

    def D_init_for_init_tasks(self, n_init_tasks: int) -> P0:
        """ used only in pre-training """
        pass

    def D_init_for_task(self, task: int) -> P0:
        """ unused """
        pass

    def get_policy_disp(self, policy: Policy, do_print: bool = False) -> NDArray['']:
        pass

    def get_state_array_disp(self, arr: NDArray['S']) -> NDArray['']:
        pass

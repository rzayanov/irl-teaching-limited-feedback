import copy
from typing import Tuple, Dict

from scipy import sparse
from scipy.sparse.csr import csr_matrix

from const import *
from dimensions import *
from env import Env


class car_simulator(Env):

    def __init__(self, length: int = 10, n_tasks: int = 8, n_lanes_per_task: int = 5, gamma: float = .99) -> None:
        self.actions: Dict = {0: "straight", 1: "left", 2: "right"}
        self.n_actions: int = len(self.actions)

        self.road_length: int = length
        self.n_tasks: int = n_tasks
        assert n_tasks in (5, 7, 8), "Improper n_tasks configuration"
        self.n_lanes_per_task: int = n_lanes_per_task
        self.n_states: int = self.road_length * 2 * self.n_tasks * self.n_lanes_per_task + 1
        self.gamma: float = gamma

        self.feature_names: List[str] = ["stone", "grass", "car", "ped", "HOV", "car-in-f", "ped-in-f", "police"]
        self.n_features: int = len(self.feature_names)
        self.W: NDArray['F'] = np.array([-1, -0.5, -5, -10, 1, -2, -5, 0])
        self.ideal_lin_w = self.W
        self.ideal_nonlin_w = np.array([-1, -0.5, -5, -10, 0, -2, -5, -9, 0, 0, 0, 0, 1, 0, 0, -3])

        self.feature_vectors: List[NDArray['F']] = self.feature_vectors()
        self.sample_features()
        self.true_reward: Reward = self.compute_reward()
        self.true_reward_disp = None

        self.D_init: P0 = np.zeros((self.n_states))  # P_0, uniform
        self.initial_states: List[int] = []
        for i in range(self.n_tasks * self.n_lanes_per_task):
            self.initial_states.append(i * 2 * self.road_length)
            self.D_init[i * 2 * self.road_length] += (1 / (self.n_tasks * self.n_lanes_per_task))

        t = self._transition_matrix()
        self.T: List[csr_matrix] = t[0]
        # self.T_dense: NDArray['A,S,S'] = t[1]

        self.compute_action_feature_matrix(t[1])

    def feature_vectors(self) -> List[NDArray['F']]:
        """
        @brief: create one-hot encoded feature vectors
        """
        feature_vectors = [np.zeros(self.n_features)]
        for i in range(self.n_features - 1):
            feature_vectors.append(np.zeros(self.n_features))
            feature_vectors[-1][i] = 1

        return feature_vectors

    def state_to_task_instance(self, state: int) -> int:
        return state // (2 * self.road_length)

    def state_to_task(self, state: int) -> int:
        """
          T0:   0 160 320 480 640
          T1:  20 180 340 500 660
          ...
          T6: 120 280 440 600 760
          T7: 140 300 460 620 780
        """
        return self.state_to_task_instance(state) % self.n_tasks

    def previous_state(self, state: int) -> int:
        prev = state - 2
        return prev

    def start_state(self, state):
        if state % (self.road_length * 2) == 0:
            return True
        return False

    def first_cells(self, state: int) -> bool:
        if state % (self.road_length * 2) < 2:
            return True
        return False

    def terminal_state(self, state: int) -> bool:
        if (state + 2) % (self.road_length * 2) < 2 or state == self.n_states - 1:
            return True
        return False

    def right_lane(self, state: int) -> bool:
        if state % 2 == 1:
            return True
        return False

    def blocked_left_lane(self, state):
        """ has car or ped to the left """
        if self.right_lane(state) and (
                self.feature_matrix[state - 1, F2_CAR] == 1 or self.feature_matrix[state - 1, F3_PED] == 1):
            return True
        return False

    def sample_features(self) -> None:
        """
        only HOV and police can be combined
        possible combinations:
          car-car, stone-car, stone-stone, grass-car, grass-grass, grass-ped, ped-ped, police-hovPolice
        possible state features with in-front:
          carf+car, carf+stone, carf+grass, pedf+ped, pedf+grass
        @brief: Sampling lanes.
        """
        self.feature_matrix: FMtx = np.zeros((self.n_states, self.n_features))

        for i in range(self.feature_matrix.shape[0] - 1):

            # if state is in the first lane type
            if ((i // (self.road_length * 2)) % self.n_tasks == T0):
                # feature index
                # [.96, 0, 0, 0.04, 0, 0, 0, 0]
                idx = np.random.choice(len(self.feature_vectors), 1, p=[1, 0, 0, 0, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]

            elif ((i // (self.road_length * 2)) % self.n_tasks == T1):
                idx = np.random.choice(len(self.feature_vectors), 1, p=[0.75, 0, 0, 0.25, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]

            elif ((i // (self.road_length * 2)) % self.n_tasks == T2):
                if self.right_lane(i):
                    self.feature_matrix[i] = self.feature_vectors[FV1_STONE]
                else:
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[1, 0, 0, 0, 0, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]

            elif ((i // (self.road_length * 2)) % self.n_tasks == T3):
                idx = np.random.choice(len(self.feature_vectors), 1, p=[0.6, 0.2, 0, 0.2, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]

            elif ((i // (self.road_length * 2)) % self.n_tasks == T4):
                if self.right_lane(i):
                    self.feature_matrix[i] = self.feature_vectors[FV2_GRASS]
                else:
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[1, 0, 0, 0, 0, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]

            elif ((i // (self.road_length * 2)) % self.n_tasks == T5):
                idx = np.random.choice(len(self.feature_vectors), 1, p=[0.6, 0, 0.2, 0.2, 0, 0, 0, 0])[0]
                self.feature_matrix[i] = self.feature_vectors[idx]

            elif ((i // (self.road_length * 2)) % self.n_tasks == T6):
                if self.right_lane(i):
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[0, 0, 0.95, 0, 0.05, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]
                else:
                    idx = np.random.choice(len(self.feature_vectors), 1, p=[0.95, 0, 0, 0, 0.05, 0, 0, 0])[0]
                    self.feature_matrix[i] = self.feature_vectors[idx]

            elif ((i // (self.road_length * 2)) % self.n_tasks == T7):
                if self.right_lane(i):
                    self.feature_matrix[i] += self.feature_vectors[FV5_HOV]
                if i % (self.road_length * 2) == 4 or i % (self.road_length * 2) == 12:
                    self.feature_matrix[i][F7A_POLICE] = 1
                    self.feature_matrix[i + 1][F7A_POLICE] = 1

            # add in-f features
            if not self.first_cells(i):
                if self.feature_matrix[i][F2_CAR] == 1:
                    self.feature_matrix[self.previous_state(i)][F5_CAR_IN_F] = 1
                if self.feature_matrix[i][F3_PED] == 1:
                    self.feature_matrix[self.previous_state(i)][F6_PED_IN_F] = 1

        # Add pedestrian if absent in T6
        for n in range(self.n_lanes_per_task):
            flag_ped = False
            start = (T6 + n * self.n_tasks) * 2 * self.road_length
            for step in range(2 * self.road_length):
                if (self.feature_matrix[start + step][F3_PED] == 1):
                    flag_ped = True
                    break
            if not flag_ped:
                self.feature_matrix[start + self.road_length - 1] = self.feature_vectors[FV4_PED]
                # todo add ped-in-f

        return

    def _transition_matrix(self) -> Tuple[List[csr_matrix], NDArray['A,S,S']]:

        transitions = np.zeros((self.n_actions, self.n_states, self.n_states))

        for a in range(self.n_actions):
            for s in range(self.n_states - 1):
                front, left, right = self.next_states(s)

                if not self.terminal_state(s):
                    if a == 0:
                        transitions[a, s, front] = 1

                    elif self.right_lane(s):
                        if a == 1:
                            transitions[a, s, left] = 1
                        else:
                            transitions[a, s, left] = 0.5
                            transitions[a, s, right] = 0.5

                    else:
                        if a == 1:
                            transitions[a, s, left] = 0.5
                            transitions[a, s, right] = 0.5
                        else:
                            transitions[a, s, right] = 1

                else:
                    transitions[a, s, self.n_states - 1] = 1
            transitions[a, self.n_states - 1, self.n_states - 1] = 1

        T = []
        for a in range(self.n_actions):
            T.append(sparse.csr_matrix(transitions[a]))

        return T, transitions

    def next_states(self, state: int) -> Tuple[int, int, int]:
        front = state + 2
        if self.right_lane(state):
            left = state + 1
            right = front
        else:
            left = front
            right = front + 1

        return front, left, right

    def compute_reward(self) -> Reward:
        """ private """
        reward = np.zeros((self.n_states))

        adjust = 0

        for s in range(self.n_states - 1):
            if not ENV_LINEAR and self.feature_matrix[s][F7A_POLICE] == 1 and self.feature_matrix[s][F4_HOV] == 1:
                reward[s] = -5 + adjust
            else:
                reward[s] = np.dot(self.W, self.feature_matrix[s]) + adjust
        return reward

    def policy_transition_matrix(self, policy: Policy) -> csr_matrix:
        T_pi = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                T_pi[s] += policy[s, a] * self.T[a][s, :]
        if EXTRA_ASSERT: assert np.allclose(T_pi.sum(axis=1), 1)
        return sparse.csr_matrix(np.transpose(T_pi))

    def reward_for_rho(self, rho: Rho) -> float:
        return np.dot(self.true_reward, rho)

    def print_lane(self, lane):
        """
        Print the given lane to understand what we have.
        """
        mask = np.array([1, 2, 3, 4, 0, 5])

        masked_features = np.dot(self.feature_matrix, mask)

        print(masked_features[self.road_length * 2 * lane: self.road_length * 2 * (lane + 1)].reshape(self.road_length,
                                                                                                      2))

    def compute_action_feature_matrix(self, t: NDArray['A,S,S']) -> None:
        """
        Given s and a, compute expected features of s'
        """
        self.action_features: NDArray['A,S,F'] = np.zeros((self.n_actions, self.n_states, self.n_features))
        for a in range(self.n_actions):
            self.action_features[a] = np.matmul(t[a], self.feature_matrix)

    def compute_exp_rho_bellman(self, policy, init_dist=None, eps=1e-6):
        """
        unused
        @brief: Compute teacher's SVF using Bellman's equation.
        """
        T_pi = self.policy_transition_matrix(policy)
        rho_s = np.zeros((self.n_states))

        if init_dist is None:
            init_dist = self.D_init

        while True:
            rho_old = copy.deepcopy(rho_s)
            rho_s = init_dist + T_pi.dot(self.gamma * rho_s)
            if np.linalg.norm(rho_s - rho_old, np.inf) < eps:
                break

        return rho_s

    def states_for_tasks(self, tasks):
        """ unused """
        states_list = list()

        for i in range(self.n_lanes_per_task):
            for l in range(tasks):
                states_list.append((l + i * self.n_tasks) * 2 * self.road_length)

        return states_list

    def D_init_for_init_tasks(self, n_init_tasks: int) -> P0:
        """ generate uniform P0 only for given initial tasks """
        D_init = np.zeros((self.n_states))
        for j in range(self.n_lanes_per_task):
            for i in range(n_init_tasks):
                D_init[(i + j * self.n_tasks) * 2 * self.road_length] += (1 / (n_init_tasks * self.n_lanes_per_task))

        return D_init

    def D_init_for_task(self, task: int) -> P0:
        D_init = np.zeros((self.n_states))
        for i in range(self.n_lanes_per_task):
            D_init[(task + i * self.n_tasks) * 2 * self.road_length] = 1 / self.n_lanes_per_task

        return D_init

    def get_policy_disp(self, policy: Policy, do_print: bool = False) -> NDArray['']:
        return None

    def get_state_array_disp(self, arr: NDArray['S']) -> NDArray['']:
        return None

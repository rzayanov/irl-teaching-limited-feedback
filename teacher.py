import collections
import copy
import math
import os.path
import pickle
from typing import Dict

import numpy as np

import dp_solver_utils
from birl_adapter import BirlAdapter, WData
from const import *
from dimensions import *
from env import Env
from mdp_solver_class import solver
from np_utils import lock
from quadratic_learner import Learner


class Teacher:

    def __init__(self, env: Env, max_iters: int = 200, num_trajectories: int = 10, num_episode: int = 10, len_episode: int = 10,
                 eta: float = .5, optimal_model_iter_count: int = 500, policy: None = None, init_states: List[int] = None) -> None:
        self.env: Env = env
        self.num_trajectories: int = num_trajectories
        self.num_episode: int = num_episode
        self.len_episode: int = len_episode
        self.batch_size: int = 1
        self.max_iters: int = max_iters
        self.b: float = 0.
        self.a: float = 0.8

        self.expert: solver = solver(self.env, self.env.true_reward, "deterministic", np.zeros(env.n_states), policy, None)
        self.reward_disp = self.env.true_reward_disp
        self.V_disp = self.expert.V_disp
        self.policy_disp = self.expert.policy_disp

        self.init_states: List[int] = init_states
        if init_states is None:
            self.init_states = self.env.initial_states

        self.expert_rho: Rho = self.expert.compute_exp_rho_bellman()
        lock(self.expert_rho)
        self.expert_reward: float = self.env.reward_for_rho(self.expert_rho)

        # unused
        # self.task_reward = self.per_task_reward()
        print("Optimal reward of expert = {}".format(self.expert_reward))

        if TEACH_OMN or (MCMC_USE_DEMOS and not LEARNER_QUAD_MUL):
            # todo driving:
            #   why Pf is positive?
            #   why weaker than others? because of full batch updates?
            self.optimal_learner: Learner = Learner(self.env, eta, num_episode, len_episode, False, None)
            self.train_optimal_model(optimal_model_iter_count)

        if TEACH_CUR or TEACH_CUR_TL or TEACH_VAR or TEACH_NOE or TEACH_RANDOM or TEACH_REP or TEACH_SCOT2:
            # used for curriculum, 10 trajectories per every init state
            self.teacher_demonstrations: Dict[int, List[Trajectory]] = self.collect_trajectories()
            ## Teacher-Curr ##
            self.seen_array: NDArray['I'] = np.zeros(len(self.init_states))

        # just for performance
        self.dict_rho_state: Dict[int, Rho] = dict()

        self.birl: BirlAdapter = BirlAdapter(env, self)
        self.used_VaR_states: NDArray['S'] = np.zeros(self.env.n_states)
        initial_assumption = -self.env.ideal_nonlin_w if LEARNER_QUAD_MUL else -self.env.ideal_lin_w
        if not PESSIMISTIC_INITIAL:
            initial_assumption = np.zeros_like(initial_assumption)
        self.inferred_wd: WData = self.birl.make_wd_soft(initial_assumption)
        self.use_easing = True
        self.use_discard = False

        self.dummy_learner: Learner = None
        self.debug_infinity_reached: bool = False
        self.debug_learner: Learner = None

    def train_optimal_model(self, total_iterations: int) -> None:
        f_name = "results/pickles/model_opt.pkl"

        if LOAD_PICKLES and os.path.exists(f_name):
            with open(f_name, "rb") as f:
                self.optimal_learner = pickle.load(f)
            print("Loaded optimal learner")
        else:
            print("Obtaining optimal w parameter.")
            for iteration in range(total_iterations):
                self.optimal_learner.update_step(self.expert_rho, "exp")
                # self.optimal_learner.scale_eta(np.sqrt(iteration + 2) / np.sqrt(iteration + 1))
                report_freq = 50
                if iteration % report_freq == 0:
                    print("Iteration [{}/{}] : Reward diff = {}, SVF diff = {}".format(
                        iteration,
                        total_iterations,
                        self.expert_reward - self.optimal_learner.exp_reward,
                        np.linalg.norm(self.expert_rho - self.optimal_learner.rho))
                    )
            if SAVE_PICKLES:
                with open(f_name, "wb") as f:
                    pickle.dump(self.optimal_learner, f)

        print("Final reward diff = {}".format(self.expert_reward - self.optimal_learner.exp_reward))
        return

    def rho_for_init_tasks(self, n_tasks: int) -> Rho:
        """ used to pre-train initial tasks """
        D_init = self.env.D_init_for_init_tasks(n_tasks)
        rho = self.expert.compute_exp_rho_bellman(D_init)
        return rho

    def rho_task(self, task):
        """ unused """
        if (task + 1) in self.dict_rho_state:
            return self.dict_rho_state[task + 1]

        D_init = np.zeros((self.env.n_states))
        for j in range(self.env.n_lanes_per_task):
            D_init[(task + j * self.env.n_tasks) * 2 * self.env.road_length] += (1 / (self.env.n_lanes_per_task))

        rho = self.expert.compute_exp_rho_bellman(D_init)

        self.dict_rho_state[task + 1] = rho
        return rho

    def per_task_reward(self) -> NDArray:
        """ unused """
        reward_list = np.zeros((self.env.n_tasks))
        for l in range(self.env.n_tasks):
            D_init = self.env.D_init_for_task(l)
            rho = self.expert.compute_exp_rho_bellman(D_init)
            reward_list[l] = self.env.reward_for_rho(rho)

        return reward_list

    def value_task(self, task):
        """ unused """
        D_init = np.zeros((self.env.n_states))
        for k in range(self.env.n_lanes_per_task):
            D_init[(task + k * self.env.n_tasks) * 2 * self.env.road_length] = (1 / self.env.n_lanes_per_task)

        V_task = np.dot(D_init, self.expert.V)
        return V_task

    def collect_trajectories(self) -> Dict[int, List[Trajectory]]:
        """
        used by cur
        for every initial state, return a list of 10 trajectories with rhos
        """
        demonstrations = collections.defaultdict(list)

        for s in self.init_states:
            t = 0
            while t < self.num_trajectories:
                # rho not really used
                episode, rho = self.expert.sample_trajectory_from_state(self.len_episode, s)
                demonstrations[s].append((rho, episode))
                t += 1

        return demonstrations

    def compute_exp_rho_state(self, state: int) -> Rho:
        if state in self.dict_rho_state:
            return self.dict_rho_state[state]

        if RHO_STATE_SAMPLING:
            rho_s = self.expert.compute_exp_rho_sampling(self.num_episode, self.len_episode, state)
        elif RHO_STATE_BELLMAN:
            rho_s = self.expert.compute_exp_rho_bellman_state(state)
        else:
            end_state = state + (2 * self.env.road_length)
            mask = np.zeros((self.env.n_states))
            mask[np.arange(state, end_state)] = self.env.n_tasks * self.env.n_lanes_per_task
            # mask[self.env.n_states - 1] = 1
            rho_s = mask * self.expert_rho

        self.dict_rho_state[state] = rho_s
        return rho_s

    def random_teacher(self) -> Tuple[Rho, List[int]]:
        rho = np.zeros(self.env.n_states)
        states = list()

        for i in range(self.batch_size):
            random_state = np.random.choice(self.init_states)
            # not used
            ep, _ = self.expert.sample_trajectory_from_state(self.len_episode, random_state)
            if not LONG_DEMOS:
                rho_sample = dp_solver_utils.calc_episode_rho(self.env, ep)
                rho += rho_sample
            states.append(random_state)
        return (rho / self.batch_size), states

    def imt_teacher(self, learner: Learner, iteration: int = -1, mode: str = "state") -> Tuple[Rho, int]:
        trajectory_cost = []

        # brute force minimization of (6) from Kamalaruban et al.
        for state in self.init_states:
            cost = self.compute_imt_cost(learner, state, mode)
            trajectory_cost.append([cost, state])

        trajectory_cost.sort(key=lambda l: l[0])

        index = self.randomizer(iteration)
        state = trajectory_cost[index][1]
        rho = self.compute_exp_rho_state(state)

        return (rho / self.batch_size), state

    def compute_imt_cost(self, learner: Learner, trajectory: int, mode: str) -> float:
        """
        @brief: Compute the IMT minimization objective.
        """
        lambda_diff: NDArray['W'] = learner.w_t - self.optimal_learner.w_t
        mu_diff: NDArray['W']
        if mode == "exp":
            # unused
            mu_diff = np.dot(learner.rho - trajectory[0], learner._reward_gradient())
        elif mode == "task":
            # unused
            task = self.env.state_to_task(trajectory[1][0][0])
            mu_diff = np.dot(learner.rho_task(task) - self.rho_task(task), learner._reward_gradient())
        else:
            mu_diff = np.dot(learner.rho_from_state(trajectory) - self.compute_exp_rho_state(trajectory),
                             learner._reward_gradient())
        objective = (np.square(learner.eta) * np.linalg.norm(mu_diff) ** 2) - (
                2 * learner.eta * np.dot(lambda_diff, mu_diff))
        return objective

    def importance_sampling_score(self, trajectory: Trajectory, learner_p: Policy) -> float:
        ratio = 1.
        for s, a in trajectory[1]:
            ratio *= (self.expert.policy[s, a] / learner_p[s, a])
        if math.isinf(ratio):
            self.debug_infinity_reached = True
            # raise Exception("Deterministic policy adopted. Re-train agent.")
        return ratio

    def teacher_importance_score(self, trajectory: Trajectory) -> float:
        """ does not consider transition probs """
        log_ratio = 0.
        for s, a in trajectory[1]:
            log_ratio += np.log(self.expert.policy[s, a])
        return log_ratio

    def learner_importance_score(self, trajectory: Trajectory,
                                 learner: Learner) -> float:
        """ does not consider transition probs """
        log_ratio = 0.
        for s, a in trajectory[1]:
            log_ratio -= np.log(learner.solver.policy[s, a])
        return log_ratio

    def blackbox_state_teacher(self, learner: Learner, iteration: int = -1) -> Tuple[Rho, int]:
        blackbox_objective = list()

        for s in self.init_states:
            rho_s = learner.rho_from_state(s)
            diff = np.dot(rho_s - self.compute_exp_rho_state(s), self.env.true_reward)
            blackbox_objective.append([abs(diff), s])

        blackbox_objective.sort(key=lambda l: l[0], reverse=True)
        index = self.randomizer(iteration)
        bbox_state = blackbox_objective[index][1]
        if LONG_DEMOS:
            bbox_state_rho = self.compute_exp_rho_state(bbox_state)
            return bbox_state_rho, bbox_state
        episode, _ = self.expert.sample_trajectory_from_state(self.len_episode, bbox_state)
        bbox_traj_rho = dp_solver_utils.calc_episode_rho(self.env, episode)
        return bbox_traj_rho, bbox_state

    def cur_teacher(self, learner: Learner, iteration: int = -1) -> Tuple[Rho, int, Episode]:
        if LONG_DEMOS:
            return self.cur_state_teacher(learner.solver.policy, iteration)
        return self.cur_traj_teacher(learner.solver.policy)

    def cur_state_teacher(self, learner_p: Policy, iteration: int = -1) -> Tuple[Rho, int, Episode]:
        state_list = list()

        for s, trajectories in self.teacher_demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.importance_sampling_score(trajectory, learner_p)

            state_list.append([cost, s])
        state_list.sort(key=lambda l: l[0], reverse=True)

        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        return self.compute_exp_rho_state(opt_state), opt_state, None

    def cur_traj_teacher(self, learner_p: Policy) -> Tuple[Rho, int, Episode]:
        traj_list: List[Tuple[float, Trajectory]] = []
        inf_traj_list: List[Trajectory] = []
        for s, trajectories in self.teacher_demonstrations.items():
            for trajectory in trajectories:
                cost = self.importance_sampling_score(trajectory, learner_p)
                traj_list.append((cost, trajectory))
                if math.isinf(cost):
                    inf_traj_list.append(trajectory)

        inf_traj_len = len(inf_traj_list)
        if INF_IMPORTANCE_RANDOM and inf_traj_len:
            print('inf imp traj count', inf_traj_len)
            opt_traj_idx = np.random.randint(inf_traj_len)
            opt_traj: Trajectory = inf_traj_list[opt_traj_idx]
        else:
            traj_list.sort(key=lambda l: l[0], reverse=True)
            opt_traj: Trajectory = traj_list[0][1]

        opt_state: int = opt_traj[1][0][0]
        return dp_solver_utils.calc_episode_rho(self.env, opt_traj[1]), opt_state, opt_traj[1]

    def teacher_curr_teacher(self, iteration: int = -1) -> Tuple[Rho, int]:
        state_list = list()
        if iteration == -1:
            if np.sum(self.seen_array) == len(self.init_states):
                self.seen_array[:] = 0

        for s, trajectories in self.teacher_demonstrations.items():
            if iteration == -1:
                index = self.env.state_to_task_instance(s)
                if self.seen_array[index] == 1:
                    continue
            cost = 0
            for trajectory in trajectories:
                cost += self.teacher_importance_score(trajectory)
            state_list.append([cost, s])

        state_list.sort(key=lambda l: l[0], reverse=True)
        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        if iteration == -1:
            self.seen_array[self.env.state_to_task_instance(opt_state)] = 1
        return self.compute_exp_rho_state(opt_state), opt_state

    def learner_curr_teacher(self, learner: Learner, iteration: int = -1) -> Tuple[Rho, int]:
        state_list = list()

        for s, trajectories in self.teacher_demonstrations.items():
            cost = 0
            for trajectory in trajectories:
                cost += self.learner_importance_score(trajectory, learner)

            state_list.append([cost, s])
        state_list.sort(key=lambda l: l[0], reverse=True)
        index = self.randomizer(iteration)
        opt_state = state_list[index][1]
        return self.compute_exp_rho_state(opt_state), opt_state

    def randomizer_step(self, iteration):
        """ unused """
        if iteration == -1:
            return 0

        else:
            if iteration < self.max_iters // 4:
                return np.random.randint(len(self.env.initial_states) // 4)
            elif iteration < self.max_iters // 2:
                return np.random.randint(len(self.env.initial_states) // 2)
            elif iteration < 3 * self.max_iters // 4:
                return np.random.randint(3 * len(self.env.initial_states) // 4)
            return np.random.randint(len(self.env.initial_states))

    def randomizer(self, iteration: int) -> int:
        """ not used """
        if iteration == -1: return 0

        cutoff = self.b * len(self.env.initial_states) + min(1, (iteration / (self.a * self.max_iters))) * (
                1 - self.b) * len(self.env.initial_states)
        return np.random.randint(max(1, int(cutoff)))

    def mu_diff_teacher(self, learner: Learner):
        """ unused """
        opt_objective = -1
        opt_state = -1

        for s in self.init_states:
            rho_diff: Rho = self.compute_exp_rho_state(s) - learner.rho_from_state(s)
            objective: float = np.linalg.norm(np.dot(rho_diff, self.env.feature_matrix))
            if objective > opt_objective:
                opt_objective = objective
                opt_state = s

        return self.compute_exp_rho_state(opt_state), opt_state

    def anticurriculum_teacher(self, learner):
        """ unused """
        opt_objective = np.inf
        opt_state = -1

        for s in self.init_states:
            objective = 0
            for trajectory in self.teacher_demonstrations[s]:
                objective += self.importance_sampling_function(trajectory, learner)
            if objective < opt_objective:
                opt_objective = objective
                opt_state = s

        return self.compute_exp_rho_state(opt_state), opt_state

    def batch_teacher_2(self) -> List[int]:
        pass

    def batch_teacher(self) -> List[int]:
        """
        @brief: SCOT teaching algorithm.
        operates on mu(s0)
        """

        # all uncovered mu(s0)
        U: Dict[int, NDArray['F']] = dict()
        for s in self.init_states:
            U[s] = np.dot(self.compute_exp_rho_state(s), self.env.feature_matrix)

        batch = list()
        while len(U) > 0:
            max_count = -1
            # greedily choosing a state that covers the maximum MUs
            for s in self.init_states:
                if s in batch: continue
                mu_s = np.dot(self.compute_exp_rho_state(s), self.env.feature_matrix)
                states = list()
                count = 0
                for state, mu in U.items():
                    if np.all(mu == mu_s):
                        count += 1
                        states.append(state)
                if count > max_count:
                    max_count = count
                    max_state = s
                    states_to_remove = states
            batch.append(max_state)
            for s in states_to_remove:
                del U[s]

        return batch

    def run_mce(self) -> None:
        """
        Implementation of Interactive-MCE
        """
        w = self.inferred_wd.w  # previous or initial estimate of the learner's weights
        wd = self.birl.make_wd_soft(w)  # contains MCE policy corresponding to w
        BirlAdapter.set_learner_values(self.dummy_learner, wd)
        eta_orig = self.dummy_learner.eta
        self.dummy_learner.eta *= TEACHER_MCE_ETA_MUL

        start_exam_idx = 0
        if self.use_easing and TEACHER_MCE_EASING:
            start_exam_idx = max(0, len(self.birl.exams) + len(self.birl.long_exams) - TEACHER_MCE_EASING)
        rho = np.zeros(self.env.n_states)
        if LONG_EXAM_SZ:  # not used
            long_exams: List[List[Tuple[int, NDArray['A']]]] = self.birl.long_exams[start_exam_idx:]
            many_states: List[int] = [le[0][0] for le in long_exams]
            for r in self.birl.long_exam_rhos[start_exam_idx:]:
                rho += r / len(long_exams)
        else:
            exams: List[Episode] = []
            for exam_idx, exam in enumerate(self.birl.exams[start_exam_idx:]):
                if exam_idx in self.birl.discarded_exam_idxs: continue
                exams.append(exam)
            many_states: List[int] = [ep[0][0] for ep in exams]
            for ep in exams:
                rho += dp_solver_utils.calc_episode_rho(self.env, ep) / len(exams)

        for iteration in range(TEACHER_MCE_ITERS):
            self.dummy_learner.update_step(rho, "many_states", many_states=many_states, calc_rho=(not RHO_STATE_SAMPLING and not RHO_STATE_BELLMAN))
            if TEACHER_MCE_ETA_SCALE:
                self.dummy_learner.scale_eta(np.sqrt(iteration + 2) / np.sqrt(iteration + 1))
            if False:
                print(np.sum(np.abs(self.debug_learner.w_t - self.dummy_learner.w_t)))

        self.dummy_learner.eta = eta_orig
        self.inferred_wd = self.birl.make_wd_soft(self.dummy_learner.w_t)

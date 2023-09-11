import copy
import os
import pickle
import random
from collections import defaultdict
from typing import List, Dict

import numpy as np

import mdp_solver_class
from birl_adapter import WData, BirlAdapter
from const import *
from dimensions import Weights, Rho, Episode
from dp_solver_utils import sample_action, compute_policy_value, compute_policy_soft_value
from env import Env
from env_car_simulator_nonlin import car_simulator
from env_ow import EnvOW
from learner_idx import *
from quadratic_learner import Learner
from scot import Scot2
from teacher import Teacher
from timer import print_timer, start_timer

N_LANES_PER_TASK = 5  # number of variations of each lane type
ETA_TEACHER = 0.5
ETA_LEARNER = 0.17
CUR_N_TRAJ = 10  # for curriculum, number of trajectories per state in teacher demonstrations
UPDATE_MODE = "state"
OMN_OPTIMAL_ITER_COUNT = 10 if FAST_PREPS else 150
INIT_TRAIN_ITER_COUNT = 10 if FAST_PREPS else 150


def do_run_experiment(experiment_id: int):
    runner = Runner(experiment_id)
    runner.run_experiment()


class Runner:
    def __init__(self, experiment_id: int):
        self.experiment_id: int = experiment_id
        seed = SEED
        if not USE_SAME_ENV:
            seed += self.experiment_id * 100
            if not ENV_LINEAR:
                # 60
                nonlin_seeds = [1001, 1003, 1004, 1009, 1010, 1012, 1015, 1024,
                                1025, 1026, 1034, 1038, 1042, 1043, 1048, 1055,
                                1056, 1057, 1058, 1059, 1062, 1065]
                seed = nonlin_seeds[self.experiment_id]
        np.random.seed(seed)
        random.seed(seed)

        if USE_OW:
            env: Env = EnvOW(N_TASKS, GAMMA)
        else:
            env: Env = car_simulator(10, N_TASKS, N_LANES_PER_TASK, GAMMA)
        self.env: Env = env

        # self.reward_curves = np.full((len(self.learner_names), TEACHING_N_ITERS + 1), np.nan)
        # self.curriculum_curves = np.full((len(self.learner_names), self.env.n_tasks, TEACHING_N_ITERS), np.nan, dtype=int)
        # the mean distance is .6 * RANDOM_W_L1_NORM
        self.learner_names_2 = CODE_NAMES
        n_learners = len(self.learner_names_2)
        self.stopped_learners = np.full(n_learners, False)
        self.teacher_w_curves = np.full((n_learners, TEACHING_N_ITERS), np.nan)
        self.teacher_p_curves = np.full((n_learners, TEACHING_N_ITERS), np.nan)
        self.teacher_loss_curves = np.full((n_learners, TEACHING_N_ITERS), np.nan)
        self.teacher_soft_loss_curves = np.full((n_learners, TEACHING_N_ITERS), np.nan)
        self.loss_curves = np.full((n_learners, TEACHING_N_ITERS), np.nan)
        self.demo_states = np.full((n_learners, self.env.n_tasks, TEACHING_N_ITERS), np.nan, dtype=int)
        self.exam_states = np.full((n_learners, self.env.n_tasks, TEACHING_N_ITERS), np.nan, dtype=int)

        self.teachers: List[Teacher] = []
        t: Teacher = Teacher(self.env, TEACHING_N_ITERS, CUR_N_TRAJ, SAMPLING_N_TRAJ, TRAJ_LEN,
                               ETA_TEACHER, OMN_OPTIMAL_ITER_COUNT)
        for i in range(len(self.learner_names_2)):
            t_copy = copy.deepcopy(t)
            self.teachers.append(t_copy)
        self.teachers[NOE_IDX].use_easing = False

        self.scot2 = Scot2(self.env, self.teachers[SCO_IDX]) if TEACH_SCOT2 else None

        w_count = Learner.get_w_count(self.env)
        init_w_t = None
        if LEARNER_RANDOM_W:
            if LEARNER_RANDOM_W_MAX:
                init_w_t = np.random.uniform(-LEARNER_RANDOM_W_MAX, LEARNER_RANDOM_W_MAX, w_count)
                print("learner w l1 norm:", np.sum(np.abs(init_w_t)).round(2))
            else:
                init_w_t = RANDOM_W_L1_NORM * Learner.l1_normalize(np.random.uniform(-1, 1, w_count))
        l = Learner(self.env, ETA_TEACHER, SAMPLING_N_TRAJ, TRAJ_LEN, CROSS_ENT, init_w_t)
        print(f"learner initial weights: {l.w_t.round(2)}")

        if SAVE_PICKLES:
            with open('results/pickles/model_ini.pkl', "wb") as f:
                pickle.dump(l, f)
        print("Environment and agents created.")
        l = self.train_initial_tasks(l, N_INIT_TASKS, INIT_TRAIN_ITER_COUNT)
        print("Learner knowledge initialized.")
        l.update_eta(ETA_LEARNER)

        self.learners: List[Learner] = []
        # self.reward_curves[:, 0] = l.exp_reward
        for i in range(len(self.learner_names_2)):
            l_copy = copy.deepcopy(l)
            self.learners.append(l_copy)
        dl = Learner(self.env, ETA_TEACHER, SAMPLING_N_TRAJ, TRAJ_LEN, False, init_w_t)
        for i in range(len(self.learner_names_2)):
            self.teachers[i].birl.dummy_learner = copy.deepcopy(dl)
            self.teachers[i].dummy_learner = copy.deepcopy(dl)
            # self.rnd_teacher.birl.debug_w_override = l.w_t.copy()
            self.teachers[i].debug_learner = self.learners[i]
            if TEACHER_MCE_ITERS and TEACHER_PREPARED:
                self.teachers[i].inferred_wd = self.teachers[i].birl.make_wd_soft(l.w_t)

        # used in SCOT
        self.scot_batch_states: List[int] = []
        if TEACH_SCOT:
            self.scot_batch_states = self.teachers[SCO_IDX].batch_teacher()

        self.iteration: int = 0

    def run_experiment(self) -> None:
        if USE_SAME_ENV:
            np.random.rand(self.experiment_id * 100)
        print("Begin teacher-learner interaction.")
        for self.iteration in range(TEACHING_N_ITERS):
            self.run_iter()
            if np.all(self.stopped_learners):
                break
        print("Finished!")

    def train_initial_tasks(self, learner: Learner, n_init_tasks: int, total_steps: int) -> Learner:
        if n_init_tasks == 0: return learner

        print("Training on initial tasks.")

        if CROSS_ENT:
            ideal_w = self.env.ideal_nonlin_w if LEARNER_QUAD_MUL else self.env.ideal_lin_w
            idxs = [0, 2, 5, 8, 10, 13] if LEARNER_QUAD_MUL else [0, 2, 5]
            learner.w_t[idxs] = ideal_w[idxs]
            return learner

        f_name = "results/pickles/model_pre.pkl"

        rho_exp = self.teachers[0].rho_for_init_tasks(n_init_tasks)
        print("Initial reward diff = {}".format(
            self.env.reward_for_rho(rho_exp) - self.env.reward_for_rho(learner.rho_for_init_tasks(n_init_tasks))))

        if LOAD_PICKLES and os.path.exists(f_name):
            with open(f_name, "rb") as f:
                learner = pickle.load(f)
            print("Loaded pretrained learner")
        else:
            for step in range(total_steps):
                learner.update_step(rho_exp, "init_tasks", n_init_tasks=n_init_tasks)
                learner.scale_eta(np.sqrt(step + 2) / np.sqrt(step + 1))
                report_freq = 50
                if step == 0 or ((step + 1) % report_freq == 0):
                    rho_learner = learner.rho_for_init_tasks(n_init_tasks)
                    print("Iteration [{}/{}] : Reward diff = {}, SVF diff = {}".format(
                        step + 1, total_steps,
                        self.env.reward_for_rho(rho_exp) - self.env.reward_for_rho(rho_learner),
                        np.linalg.norm(rho_exp - rho_learner))
                    )
            if SAVE_PICKLES:
                with open(f_name, "wb") as f:
                    pickle.dump(learner, f)

        final_r_diff = self.env.reward_for_rho(rho_exp) - self.env.reward_for_rho(learner.rho_for_init_tasks(n_init_tasks))
        print(f"Final reward diff = {final_r_diff}")
        print_timer('main')
        return learner

    def run_iter(self) -> None:
        report_freq = 1
        if self.iteration % report_freq == 0:
            print('-------')
            print(f"E{self.experiment_id} iter {self.iteration}/{TEACHING_N_ITERS}")
            print_timer("main")

        if TEACH_OMN and not self.stopped_learners[OMN_IDX]:
            self.advance_omn()

        if (TEACH_SCOT or TEACH_SCOT2) and not self.stopped_learners[SCO_IDX]:
            self.advance_scot()

        if TEACH_CUR and not self.stopped_learners[CUR_IDX]:
            self.advance_cur()

        # 4: Cur-T teacher
        # if TEACH_CUR_TL:
        #     rho_curriculum, state = self.teacher.teacher_curr_teacher()
        #     self.learners[None].update_step(rho_curriculum, UPDATE_MODE, state)
            # self.curriculum_curves[4, self.env.state_to_task(state), self.iteration] = 1
        # 5: Cur-L teacher
        # if TEACH_CUR_TL:
        #     rho_curriculum, state = self.teacher.learner_curr_teacher(self.learners[None])
        #     self.learners[None].update_step(rho_curriculum, UPDATE_MODE, state)
            # self.curriculum_curves[5, self.env.state_to_task(state), self.iteration] = 1

        if TEACH_BBOX and not self.stopped_learners[None]:
            self.advance_bbox()

        random_exam_s = np.random.choice(self.env.initial_states) if EXAM_AS_TRAJ else np.random.randint(self.env.n_states)

        if TEACH_REP and not self.stopped_learners[REP_IDX]:
            self.advance_repeater(random_exam_s)

        if LEARN_VAR and not self.stopped_learners[VAR_IDX]:
            self.advance_VaR(random_exam_s, VAR_IDX, TEACH_VAR)

        if TEACH_NOE and not self.stopped_learners[NOE_IDX]:
            self.advance_VaR(random_exam_s, NOE_IDX, True)

        if LEARN_RANDOM and not self.stopped_learners[RND_IDX]:
            self.advance_random(random_exam_s)

        if TEACH_AGN and not self.stopped_learners[AGN_IDX]:
            self.advance_agn()

        # for i in range(len(self.learners)):
        #     self.reward_curves[i, self.iteration + 1] = self.learners[i].exp_reward

        self.save_progress()

    def advance_omn(self):
        teacher = self.teachers[OMN_IDX]
        loss = teacher.expert_reward - self.learners[OMN_IDX].exp_reward
        self.loss_curves[OMN_IDX, self.iteration] = loss
        print(f"Omn {loss=:.5f}")
        if loss <= STOP_LOSS:
            self.stopped_learners[OMN_IDX] = True
        rho_imt, state = teacher.imt_teacher(self.learners[OMN_IDX])
        self.learners[OMN_IDX].update_step(rho_imt, UPDATE_MODE, state)
        # self.curriculum_curves[1, self.env.state_to_task(state), self.iteration] = 1
        self.demo_states[OMN_IDX, self.env.state_to_task(state), self.iteration] = 1

    def advance_scot(self):
        loss = self.teachers[SCO_IDX].expert_reward - self.learners[SCO_IDX].exp_reward
        self.loss_curves[SCO_IDX, self.iteration] = loss
        print(f"Sco {loss=:.5f}")
        if loss <= STOP_LOSS:
            self.stopped_learners[SCO_IDX] = True
        if TEACH_SCOT:
            if self.iteration < len(self.scot_batch_states):
                state = self.scot_batch_states[self.iteration]
            else:
                state = np.random.choice(self.env.initial_states)
            rho = self.teachers[SCO_IDX].compute_exp_rho_state(state)
            ep = None
        else:
            rho, state, ep = self.scot2.get_next(self.iteration)
        self.learners[SCO_IDX].update_step(rho, UPDATE_MODE, state, ep=ep)
        # self.curriculum_curves[2, self.env.state_to_task(state), self.iteration] = 1
        self.demo_states[SCO_IDX, self.env.state_to_task(state), self.iteration] = 1

    def advance_cur(self):
        teacher = self.teachers[CUR_IDX]
        loss = teacher.expert_reward - self.learners[CUR_IDX].exp_reward
        self.loss_curves[CUR_IDX, self.iteration] = loss
        print(f"Cur {loss=:.5f}")
        if loss <= STOP_LOSS:
            self.stopped_learners[CUR_IDX] = True
        rho_curriculum, state, ep = teacher.cur_teacher(self.learners[CUR_IDX])
        if teacher.debug_infinity_reached:
            print('exp', self.experiment_id, '- cur teacher reached infinite importance')
            teacher.debug_infinity_reached = False
        self.learners[CUR_IDX].update_step(rho_curriculum, UPDATE_MODE, state, ep=ep)
        # self.curriculum_curves[3, self.env.state_to_task(state), self.iteration] = 1
        self.demo_states[CUR_IDX, self.env.state_to_task(state), self.iteration] = 1

    def advance_bbox(self):
        teacher = self.teachers[None]
        loss = teacher.expert_reward - self.learners[None].exp_reward
        self.loss_curves[None, self.iteration] = loss
        print(f"Box {loss=:.5f}")
        if loss <= STOP_LOSS:
            self.stopped_learners[None] = True
        rho_bbox, state = teacher.blackbox_state_teacher(self.learners[None], -1)
        self.learners[None].update_step(rho_bbox, UPDATE_MODE, state)
        # self.curriculum_curves[6, self.env.state_to_task(state), self.iteration] = 1
        self.demo_states[None, self.env.state_to_task(state), self.iteration] = 1

    def advance_agn(self):
        teacher = self.teachers[AGN_IDX]
        loss = teacher.expert_reward - self.learners[AGN_IDX].exp_reward
        self.loss_curves[AGN_IDX, self.iteration] = loss
        print(f"Agn {loss=:.5f}")
        if loss <= STOP_LOSS:
            self.stopped_learners[AGN_IDX] = True
        rho_random, states = teacher.random_teacher()
        if LONG_DEMOS:
            rho_random = teacher.compute_exp_rho_state(states[0])
        ep = teacher.expert.sample_trajectory_from_state(TRAJ_LEN, states[0])[0] if CROSS_ENT else None
        self.learners[AGN_IDX].update_step(rho_random, UPDATE_MODE, states[0], ep=ep)
        # self.curriculum_curves[0, self.env.state_to_task(states[0]), self.iteration] = 1
        self.demo_states[AGN_IDX, self.env.state_to_task(states[0]), self.iteration] = 1

    def advance_cycler(self) -> None:
        useful_task_idx = self.iteration % 3
        task_idx = [3, 6, 7][useful_task_idx]
        candidates = []
        for s in self.env.initial_states:
            if task_idx == self.env.state_to_task(s):
                candidates.append(s)
        exam_s = np.random.choice(candidates)

    def advance_VaR(self, random_exam_s: int, idx: int, should_teach: bool) -> None:
        learner: Learner = self.learners[idx]
        teacher: Teacher = self.teachers[idx]

        # AL
        if self.iteration:
            exam_s = -1
            VaR_states, VaR_values = teacher.birl.get_VaR_states(teacher.inferred_wd)
            usable_state_count = len(self.env.initial_states) if EXAM_AS_TRAJ else self.env.n_states
            if teacher.used_VaR_states.sum() == usable_state_count * EXAM_REUSE_COUNT:
                teacher.used_VaR_states[:] = 0
            for s in VaR_states:
                if EXAM_AS_TRAJ and s not in self.env.initial_states:
                    continue
                if teacher.used_VaR_states[s] == EXAM_REUSE_COUNT:
                    continue
                teacher.used_VaR_states[s] += 1
                exam_s = s
                break
            assert exam_s >= 0
        else:
            exam_s = random_exam_s
        self.add_exam_answers(teacher, learner, exam_s)

        # IRL
        if TEACHER_MCE_ITERS:
            teacher.run_mce()
        if BIRL_NORM_MODE != 'const':
            w = teacher.inferred_wd.w if BIRL_NORM_MODE == 'inferred' else learner.w_t
            birl_norm = np.sum(np.abs(w))
            teacher.birl.set_norm(birl_norm)
        teacher.birl.run_inference()
        if not TEACHER_MCE_ITERS:
            teacher.inferred_wd = teacher.birl.map_wd

        self.save_metrics(idx, learner, teacher, exam_s)

        # MT
        if should_teach:
            self.do_teach(idx, teacher, learner)

    def advance_random(self, random_exam_s: int) -> None:
        learner: Learner = self.learners[RND_IDX]
        teacher: Teacher = self.teachers[RND_IDX]

        # AL
        exam_s = random_exam_s
        self.add_exam_answers(teacher, learner, exam_s)

        # IRL
        if TEACHER_MCE_ITERS:
            teacher.run_mce()
        else:
            if BIRL_NORM_MODE == 'learner':
                teacher.birl.set_norm(np.sum(np.abs(learner.w_t)))
            teacher.birl.run_inference()
            teacher.inferred_wd = teacher.birl.map_wd

        self.save_mcmc_debug(learner, teacher)
        self.save_metrics(RND_IDX, learner, teacher, exam_s)

        # MT
        if TEACH_RANDOM:
            self.do_teach(RND_IDX, teacher, learner)

    def advance_repeater(self, random_exam_s: int) -> None:
        learner: Learner = self.learners[REP_IDX]
        teacher = self.teachers[REP_IDX]

        # AL
        if EXAM_REUSE_COUNT < 100: raise NotImplementedError
        if self.iteration:
            exam_s = teacher.birl.demos[-1][1]
        else:
            exam_s = random_exam_s
        self.add_exam_answers(teacher, learner, exam_s)

        # IRL
        if TEACHER_MCE_ITERS:
            teacher.run_mce()
        else:
            if BIRL_NORM_MODE == 'learner':
                teacher.birl.set_norm(np.sum(np.abs(learner.w_t)))
            teacher.birl.run_inference()
            teacher.inferred_wd = teacher.birl.map_wd

        self.save_metrics(REP_IDX, learner, teacher, exam_s)

        # MT
        self.do_teach(REP_IDX, teacher, learner)

    def add_exam_answers(self, teacher: Teacher, learner: Learner, exam_s: int) -> None:
        if DEBUG_ALL_EXAM_STATES:
            all_exam_states = self.env.initial_states if EXAM_AS_TRAJ else [*range(self.env.n_states)]
            for exam_s in all_exam_states:
                self.add_exam_answers_0(teacher, learner, exam_s)
            return
        self.add_exam_answers_0(teacher, learner, exam_s)

    def add_exam_answers_0(self, teacher: Teacher, learner: Learner, exam_s: int) -> None:
        for e in range(EXAMS_PER_ITER):
            if LONG_EXAM_SZ:
                if EXAM_AS_TRAJ:
                    if USE_OW:
                        exam_a_dists = [(s, learner.solver.policy[s]) for s in range(self.env.n_states) if s != exam_s]
                        exam_a_dists.insert(0, (exam_s, learner.solver.policy[exam_s]))
                    else:
                        exam_a_dists = [(s, learner.solver.policy[s]) for s in range(exam_s, exam_s + self.env.road_length * 2)]
                    teacher.birl.add_long_exam(exam_a_dists)
                    teacher.birl.long_exam_rhos.append(learner.rho_from_state(exam_s))
                else:
                    exam_a_dist = learner.solver.policy[exam_s]
                    teacher.birl.add_long_exam([(exam_s, exam_a_dist)])
            else:
                if EXAM_AS_TRAJ:
                    exam_ep, _ = learner.solver.sample_trajectory_from_state(TRAJ_LEN, exam_s)
                    teacher.birl.add_exam(exam_ep)
                else:
                    exam_a = sample_action(learner.solver.policy, exam_s)
                    teacher.birl.add_exam([(exam_s, exam_a)])

    def do_teach(self, learner_idx: int, teacher: Teacher, learner: Learner) -> None:
        wds: List[WData] = teacher.birl.resample if TEACH_WITH_CHAIN else [teacher.inferred_wd]
        if MCMC_USE_DEMOS and not LEARNER_QUAD_MUL:
            func = teacher.imt_teacher
        elif MCMC_SOFT_Q:
            func = teacher.cur_teacher
        else:
            func = teacher.blackbox_state_teacher
        state_counter = defaultdict(int)
        state_rho_dict: Dict[int, Rho] = {}
        state_ep_dict: Dict[int, Episode] = {}
        wd_state_dict: Dict[bytes, int] = {}
        for wd in wds:
            if wd.key in wd_state_dict:
                state = wd_state_dict[wd.key]
            else:
                BirlAdapter.set_learner_values(teacher.dummy_learner, wd)
                rho, state, ep = func(teacher.dummy_learner)
                if teacher.debug_infinity_reached:
                    print('exp', self.experiment_id, '- var/agn teacher reached infinite importance')
                    teacher.debug_infinity_reached = False
                wd_state_dict[wd.key] = state
                state_rho_dict[state] = rho
                state_ep_dict[state] = ep
            state_counter[state] += 1
        state_counter = sorted(state_counter.items(), key=lambda i: i[1], reverse=True)
        demo_state = state_counter[0][0]
        demo_rho = state_rho_dict[demo_state]
        demo_ep = state_ep_dict[demo_state]

        if TEACH_WITH_CHAIN:
            inferred_key = teacher.inferred_wd.key
            if inferred_key in wd_state_dict:
                i_state = wd_state_dict[inferred_key]
            else:
                BirlAdapter.set_learner_values(teacher.dummy_learner, teacher.inferred_wd)
                _, i_state = func(teacher.dummy_learner)
            if i_state != demo_state:
                print(f'teaching with chain: {demo_state=}, {i_state=}')

        teacher.birl.add_demo(demo_rho, demo_state)
        learner.update_step(demo_rho, UPDATE_MODE, demo_state, ep=demo_ep)
        # self.curriculum_curves[7 + learner_idx, self.env.state_to_task(demo_state), self.iteration] = 1
        self.demo_states[learner_idx, self.env.state_to_task(demo_state), self.iteration] = 1

    def save_mcmc_debug(self, learner: Learner, teacher: Teacher) -> None:
        if not SAVE_MCMC_DEBUG:
            return
        w_pre_demo: Weights = learner.w_t
        solver_pre_demo: mdp_solver_class = learner.solver
        w_diffs, p_diffs, posteriors, resample_posteriors, _, _ = teacher.birl.debug_collect_data(w_pre_demo, solver_pre_demo.policy)
        os.makedirs("results/mcmc_w_diffs", exist_ok=True)
        os.makedirs("results/mcmc_p_diffs", exist_ok=True)
        os.makedirs("results/mcmc_posteriors", exist_ok=True)
        os.makedirs("results/mcmc_resample_posteriors", exist_ok=True)
        os.makedirs("results/mcmc_switches", exist_ok=True)
        os.makedirs("results/mcmc_switch_probas", exist_ok=True)
        np.save(f"results/mcmc_w_diffs/{self.iteration}", w_diffs)
        np.save(f"results/mcmc_p_diffs/{self.iteration}", p_diffs)
        np.save(f"results/mcmc_posteriors/{self.iteration}", posteriors)
        np.save(f"results/mcmc_resample_posteriors/{self.iteration}", resample_posteriors)
        np.save(f"results/mcmc_switches/{self.iteration}", teacher.birl.switches)
        np.save(f"results/mcmc_switch_probas/{self.iteration}", teacher.birl.switch_probas)

    def save_metrics(
            self, learner_idx: int,
            learner: Learner,
            teacher: Teacher,
            exam_s: int
    ) -> None:
        inferred_wd = teacher.inferred_wd
        teacher_w_diff = np.linalg.norm(learner.w_t - inferred_wd.w)
        teacher_p_diff = np.sum(np.abs(learner.solver.policy - inferred_wd.policy))

        inferred_lr_V = compute_policy_value(self.env, learner.solver.reward, inferred_wd.policy, inferred_wd.V, EPS)
        learner_lr_V = compute_policy_value(self.env, learner.solver.reward, learner.solver.policy, learner.solver.V, EPS)
        teacher_loss = np.dot(learner_lr_V - inferred_lr_V, self.env.D_init)

        inferred_lr_soft_V = compute_policy_soft_value(self.env, learner.solver.reward, inferred_wd.policy, inferred_wd.V, EPS)
        learner_lr_soft_V = compute_policy_soft_value(self.env, learner.solver.reward, learner.solver.policy, learner.solver.V, EPS)
        teacher_soft_loss = np.dot(learner_lr_soft_V - inferred_lr_soft_V, self.env.D_init)

        loss = teacher.expert_reward - learner.exp_reward

        name = self.learner_names_2[learner_idx]
        print(f"{name} {loss=:.5f}, {teacher_loss=:.5f}, posterior={inferred_wd.posterior:.5f}")
        if loss <= STOP_LOSS:
            self.stopped_learners[learner_idx] = True
        self.teacher_w_curves[learner_idx, self.iteration] = teacher_w_diff
        self.teacher_p_curves[learner_idx, self.iteration] = teacher_p_diff
        self.teacher_loss_curves[learner_idx, self.iteration] = teacher_loss
        self.teacher_soft_loss_curves[learner_idx, self.iteration] = teacher_soft_loss
        self.loss_curves[learner_idx, self.iteration] = loss
        if EXAM_AS_TRAJ:
            self.exam_states[learner_idx, self.env.state_to_task(exam_s), self.iteration] = 1

    def save_progress(self) -> None:
        prefix = "results/"
        reward_path = prefix + "reward/"
        curriculum_path = prefix + "curriculum/"
        w_path = prefix + "w_diffs/"
        p_path = prefix + "p_diffs/"
        teacher_loss_path = prefix + "teacher_losses/"
        teacher_soft_loss_path = prefix + "teacher_soft_losses/"
        loss_path = prefix + "losses/"
        ds_path = prefix + "demo_states/"
        es_path = prefix + "exam_states/"
        os.makedirs(reward_path, exist_ok=True)
        os.makedirs(curriculum_path, exist_ok=True)
        os.makedirs(w_path, exist_ok=True)
        os.makedirs(p_path, exist_ok=True)
        os.makedirs(teacher_loss_path, exist_ok=True)
        os.makedirs(teacher_soft_loss_path, exist_ok=True)
        os.makedirs(loss_path, exist_ok=True)
        os.makedirs(ds_path, exist_ok=True)
        os.makedirs(es_path, exist_ok=True)
        # np.save(reward_path + f"array_{self.experiment_id}", self.reward_curves)
        # np.save(curriculum_path + f"array_{self.experiment_id}", self.curriculum_curves)
        np.save(w_path + f"array_{self.experiment_id}", self.teacher_w_curves)
        np.save(p_path + f"array_{self.experiment_id}", self.teacher_p_curves)
        np.save(teacher_loss_path + f"array_{self.experiment_id}", self.teacher_loss_curves)
        np.save(teacher_soft_loss_path + f"array_{self.experiment_id}", self.teacher_soft_loss_curves)
        np.save(loss_path + f"array_{self.experiment_id}", self.loss_curves)
        np.save(ds_path + f"array_{self.experiment_id}", self.demo_states)
        np.save(es_path + f"array_{self.experiment_id}", self.exam_states)

        if SAVE_PICKLES:
            for i in range(len(self.learners)):
                with open(f"results/pickles/model_{i}.pkl", "wb") as f:
                    pickle.dump(self.learners[i], f)

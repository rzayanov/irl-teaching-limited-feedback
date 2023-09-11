import math
from collections import defaultdict
from typing import Dict, Set

import scipy.special

import dp_solver_utils
from const import *
from dimensions import *
from env import Env
from np_utils import lock
from quadratic_learner import Learner
from timer import print_timer, start_timer


VaR = 0.95


class WData:
    def __init__(
            self, key: bytes, bins: NDArray['W'], w: Weights, reward: Reward, reward_disp, V: Vs, V_disp, qs: Qs,
            policy: Policy, policy_disp
    ):
        self.key: bytes = key
        self.bins: NDArray['W'] = bins
        lock(self.bins)
        self.w: Weights = w
        self.reward: Reward = reward
        lock(self.reward)
        self.reward_disp = reward_disp
        self.policy: Policy = policy
        lock(self.policy)
        self.policy_disp = policy_disp
        self.V: Vs = V
        lock(self.V)
        self.V_disp = V_disp
        self.Q: Qs = qs
        lock(self.Q)
        self.posterior: float = 0
        self.applied_exam_count: int = 0


W_CACHE: Dict[bytes, WData] = {}
EVD_CACHE: Dict[bytes, Dict[bytes, Vs]] = defaultdict(dict)


class BirlAdapter:
    """
    Implements VaR-based Active Learning algorithms
    """

    def __init__(self, env: Env, teacher) -> None:
        from teacher import Teacher
        self.iteration: int = -1
        self.env: Env = env
        self.teacher: Teacher = teacher
        self.dummy_learner: Learner = None
        self.exams: List[Episode] = []
        self.long_exams: List[List[Tuple[int, NDArray['A']]]] = []
        self.long_exam_rhos: List[Rho] = []
        self.demos: List[Tuple[Rho, int]] = []
        self.chain: List[WData] = []
        self.resample: List[WData] = []
        self.map_wd: WData = None
        self.Z_buf: NDArray['A'] = np.zeros(self.env.n_actions)
        self.switches: NDArray[''] = None
        self.switch_probas: NDArray[''] = None
        self.w_cache_hits: int = 0
        self.evd_cache_hits: int = 0
        self.debug_w_override: Weights = None
        self.w_norm = 0
        self.w_max = 0
        self.bin_norm = 0
        self.set_norm(RANDOM_W_L1_NORM)
        self.discarded_exam_idxs: Set[int] = set()

    def set_norm(self, w_norm):
        print("BIRL norm", w_norm)
        self.w_norm = w_norm
        self.w_max = self.w_norm
        self.bin_norm = int(self.w_norm / R_STEP_SIZE)

    def add_exam(self, ep: Episode) -> None:
        self.exams.append(ep)

    def add_long_exam(self, a_dists: List[Tuple[int, NDArray['A']]]) -> None:
        self.long_exams.append(a_dists)

    def add_demo(self, rho: Rho, s: int):
        self.demos.append((rho, s))

        if self.teacher.use_discard:
            hack_affected_map: Dict[int, List[int]] = {
                0: [],
                1: [1, 3, 5],
                2: [2, 3],
                3: [1, 2, 3, 5],
                4: [4, 5, 6],
                5: [1, 3, 4, 5, 6],
                6: [4, 5, 6],
                7: [7],
            }
            demo_task = self.env.state_to_task(s)
            print('discarding due to', demo_task)
            affected_tasks = hack_affected_map[demo_task]
            for exam_idx, exam in enumerate(self.exams):
                if exam_idx in self.discarded_exam_idxs: continue
                exam_task = self.env.state_to_task(exam[0][0])
                if exam_task in affected_tasks:
                    self.discarded_exam_idxs.add(exam_idx)
                    print('discarded', exam_task)
            print('left', len(self.exams) - len(self.discarded_exam_idxs))

    def run_inference(self) -> None:
        """
        The first part of the Active-VaR algorithm: Bayesian IRL
        (Ramachandran, Deepak, and Eyal Amir. "Bayesian Inverse Reinforcement Learning." IJCAI. Vol. 7. 2007.)
        In our final experiment, instead of using MCMC to collect samples,
        we used uniform samples and resamples the set according to the probabilities of the samples.
        """
        assert len(self.exams) or len(self.long_exams)
        assert not (len(self.exams) and len(self.long_exams))
        if MCMC_USE_DEMOS:
            assert len(self.demos) == (len(self.exams) + len(self.long_exams)) / EXAMS_PER_ITER - 1
        self.iteration += 1

        if BIRL_NORM_MODE == 'const' and self._is_resampling_enabled() and self.iteration:
            self._update_chain_posteriors()
        else:
            self._build_chain()

        map_posterior = np.NINF
        self.map_wd = None
        for wd in self.chain:
            if wd.posterior > map_posterior:
                map_posterior = wd.posterior
                self.map_wd = wd

        self._build_resample()

    def _build_chain(self) -> None:
        self.chain.clear()
        self.w_cache_hits = 0

        reject_cnt = 0
        start_timer("MCMC")

        long_chain_len = MCMC_N_INITS * MCMC_BURN_IN + MCMC_CHAIN_LENGTH * MCMC_CHAIN_SKIP
        long_subchain_len = long_chain_len // MCMC_N_INITS
        self.switches = np.full(long_chain_len, np.nan)
        self.switch_probas = np.full(long_chain_len, np.nan)
        wd: WData = None
        for itr in range(long_chain_len):
            if False and itr % (long_chain_len / 10) == 0:
                print(f"MCMC iter {itr}")
                print_timer("MCMC")
                print(f"MCMC total value iters: {dp_solver_utils.CUMUL_ITERS[0]}")

            if itr % long_subchain_len == 0:
                wd = self._reuse_map() if MCMC_REUSE_MAP and self.iteration and not itr else self._init_w()
                self.switches[itr] = 1
                self.switch_probas[itr] = 1
            else:
                temp_wd = self._modify_w(wd)
                if temp_wd.posterior >= wd.posterior:
                    do_switch = True
                    self.switch_probas[itr] = 1
                else:
                    switch_proba = math.exp(temp_wd.posterior - wd.posterior)
                    do_switch = np.random.rand() < switch_proba
                    self.switch_probas[itr] = switch_proba
                if do_switch:
                    wd = temp_wd
                else:
                    reject_cnt += 1
                self.switches[itr] = 1 if do_switch else 0
            if itr % long_subchain_len >= MCMC_BURN_IN and itr % MCMC_CHAIN_SKIP == 0:
                self.chain.append(wd)
            if MCMC_N_INITS != MCMC_CHAIN_LENGTH and (itr + 1) % long_subchain_len == 0:
                print('switch rate:', np.sum(self.switches[itr-99:itr+1]))
        assert len(self.chain) == MCMC_CHAIN_LENGTH

        if USE_W_CACHE:
            print('w cache ratio:', 100 * self.w_cache_hits // long_chain_len, '%')

    def _reuse_map(self) -> WData:
        wd = self.map_wd
        self._update_posterior(wd)
        return wd

    def _init_w(self) -> WData:
        w_count = Learner.get_w_count(self.env)
        bins_new = np.zeros(w_count)
        abs_sum = 0
        for bin_idx in range(w_count):
            z = np.random.rand()
            if z < 0.5:
                bin_val = np.log(2.0 * z)
            else:
                bin_val = -np.log(2.0 - 2.0 * z)
            bins_new[bin_idx] = bin_val
            abs_sum += abs(bin_val)
        bins_new *= self.bin_norm / abs_sum
        bins_new = bins_new.round().astype(int)

        curr_bin_norm = np.sum(np.abs(bins_new))
        while curr_bin_norm < self.bin_norm:
            bin_idx = np.random.randint(w_count)
            bin_val = bins_new[bin_idx]
            bins_new[bin_idx] += 1 if bin_val > 0 else -1
            curr_bin_norm += 1
        while self.bin_norm < curr_bin_norm:
            bin_idx = 0
            bin_val = 0
            while bin_val == 0:
                bin_idx = np.random.randint(w_count)
                bin_val = bins_new[bin_idx]
            bins_new[bin_idx] -= 1 if bin_val > 0 else -1
            curr_bin_norm -= 1
        return self._get_or_make_wd(bins_new, np.zeros(self.env.n_states))

    def _modify_w(self, wd: WData) -> WData:
        bins_new = wd.bins.copy()
        iter_count = 0
        for st in range(R_STEP_COUNT):
            bin_idx, bin_val, bin_idx2, bin_val2 = 0, 0, 0, 0
            while not bin_val and not bin_val2:
                bin_idx = np.random.randint(len(bins_new))
                bin_idx2 = np.random.randint(len(bins_new))
                while bin_idx == bin_idx2:
                    bin_idx2 = np.random.randint(len(bins_new))
                bin_val = bins_new[bin_idx]
                bin_val2 = bins_new[bin_idx2]
                iter_count += 1
            if not bin_val:
                dirr = np.random.choice((-1, 1))
                dirr2 = -1
            elif not bin_val2:
                dirr2 = np.random.choice((-1, 1))
                dirr = -1
            else:
                dirr = np.random.choice((-1, 1))
                dirr2 = -dirr
            bins_new[bin_idx] += dirr if bin_val > 0 else -dirr
            bins_new[bin_idx2] += dirr2 if bin_val2 > 0 else -dirr2
        return self._get_or_make_wd(bins_new, wd.V)

    def _get_or_make_wd(self, bins_new: NDArray['W'], init_V: Vs) -> WData:
        w_new_round = bins_new * R_STEP_SIZE
        if self.debug_w_override is not None:
            w_new_round[:] = self.debug_w_override
        if EXTRA_ASSERT: assert np.isclose(np.sum(np.abs(w_new_round)), self.w_norm, EPS)
        if EXTRA_ASSERT: assert np.max(np.abs(w_new_round)) < self.w_max + EPS

        w_key = bins_new.tobytes()
        if w_key in W_CACHE:
            self.w_cache_hits += 1
            wd = W_CACHE[w_key]
            self._update_posterior(wd)
            return wd
        else:
            wd = self.make_wd(w_key, bins_new, w_new_round, init_V, True, MCMC_SOFT_Q)
            if USE_W_CACHE:
                W_CACHE[w_key] = wd
            return wd

    def make_wd_soft(self, w: Weights) -> WData:
        return self.make_wd(None, None, w.copy(), np.zeros(self.env.n_states), False, True)

    def make_wd(self, key: bytes, bins: NDArray['W'], w: Weights, init_V: Vs, calc_posterior, soft_q) -> WData:
        reward = Learner.calc_reward(w, self.env.feature_matrix)
        reward_disp = self.env.get_state_array_disp(reward) if CALC_DISP_DATA else None
        iter_func = dp_solver_utils.value_iteration_soft if soft_q else dp_solver_utils.value_iteration
        V, Q, policy = iter_func(self.env, reward, init_V, EPS)
        p_disp = self.env.get_policy_disp(policy) if CALC_DISP_DATA else None
        V_disp = self.env.get_state_array_disp(V) if CALC_DISP_DATA else None
        wd = WData(key, bins, w, reward, reward_disp, V, V_disp, Q, policy, p_disp)
        if calc_posterior:
            self._update_posterior(wd)
        return wd

    def _update_posterior(self, wd: WData) -> None:
        """
        returns log(Z*P(R|pi)) = log(P(demos|R)) = log(P(a1|s1,R)) + ...
        P(a1|s1,R) = exp(Qs1a1)/sum(exp(Qs1))
        """

        if MCMC_USE_DEMOS:
            start_exam_idx = wd.applied_exam_count
            self._init_dummy_learner(wd)
        elif self.teacher.use_easing and MCMC_EASING:
            wd.posterior = 0
            start_exam_idx = max(0, len(self.exams) + len(self.long_exams) - MCMC_EASING)
        else:
            start_exam_idx = wd.applied_exam_count

        for exam_idx in range(start_exam_idx, len(self.exams)):
            if exam_idx in self.discarded_exam_idxs: continue
            if MCMC_USE_DEMOS:
                self._advance_dummy_learner(exam_idx, wd)
            if self.teacher.use_easing and (len(self.exams) - exam_idx) % EXAMS_PER_ITER == 0:
                wd.posterior *= MCMC_EASING_MUL
            ep = self.exams[exam_idx]
            for s, a in ep:
                if MCMC_SOFT_Q:
                    wd.posterior += math.log(wd.policy[s, a])
                else:
                    self.Z_buf[:] = MCMC_BETA * wd.Q[s, :]
                    wd.posterior += MCMC_BETA * wd.Q[s, a] - scipy.special.logsumexp(self.Z_buf)
            wd.applied_exam_count += 1
        for long_exam_idx in range(start_exam_idx, len(self.long_exams)):
            if MCMC_USE_DEMOS:
                self._advance_dummy_learner(long_exam_idx, wd)
            if self.teacher.use_easing:
                wd.posterior *= MCMC_EASING_MUL
            dists = self.long_exams[long_exam_idx]
            for s, a_dist in dists:
                if MCMC_SOFT_Q:
                    wd.posterior += LONG_EXAM_SZ * np.dot(a_dist, np.log(wd.policy[s, :]))
                else:
                    raise NotImplementedError
            wd.applied_exam_count += 1

    def _init_dummy_learner(self, wd: WData) -> None:
        if not len(self.demos):
            return
        BirlAdapter.set_learner_values(self.dummy_learner, wd)

    @staticmethod
    def set_learner_values(l: Learner, wd: WData) -> None:
        s = l.solver

        l.w_t[:] = wd.w
        l.reward = None  # l._quadratic_reward()
        s.reward = None  # wd.reward
        s.policy = wd.policy
        s.V = wd.V
        s.Q = None  # wd.Q
        l.rho = None if RHO_STATE_SAMPLING or RHO_STATE_BELLMAN else l.solver.compute_exp_rho_bellman()
        lock(l.rho)
        l.exp_reward = None  # self.env.reward_for_rho(l.rho)

    def _advance_dummy_learner(self, exam_idx: int, wd: WData) -> None:
        demo_idx = exam_idx - 1
        if demo_idx < 0:
            return

        l = self.dummy_learner
        demo_rho, demo_s = self.demos[demo_idx]
        l.update_step(demo_rho, "state", demo_s, calc_rho=(not RHO_STATE_SAMPLING and not RHO_STATE_BELLMAN))
        s = l.solver

        wd.w[:] = l.w_t
        wd.reward = l.reward
        lock(wd.reward)
        wd.policy = s.policy
        lock(wd.policy)
        wd.V = s.V
        lock(wd.V)
        wd.Q = s.Q
        lock(wd.Q)

    def _is_resampling_enabled(self) -> bool:
        return MCMC_N_INITS == MCMC_CHAIN_LENGTH

    def _build_resample(self) -> None:
        if not self._is_resampling_enabled():
            self.resample = self.chain
            return
        probs = np.zeros(MCMC_CHAIN_LENGTH)
        for wd_idx in range(len(self.chain)):
            probs[wd_idx] = self.chain[wd_idx].posterior - self.map_wd.posterior
        np.exp(probs, out=probs)
        z = np.sum(probs)
        probs /= z
        resample = np.random.choice(self.chain, MCMC_CHAIN_LENGTH, p=probs).tolist()
        unique_counts = defaultdict(int)
        for wd in resample:
            unique_counts[id(wd)] += 1
        unique_counts = sorted(unique_counts.values(), reverse=True)
        print('resample unique count:', len(unique_counts), '/', unique_counts[:10])
        self.resample = resample

    def _update_chain_posteriors(self) -> None:
        for wd in self.chain:
            self._update_posterior(wd)

    def get_VaR_states(self, inferred_wd) -> Tuple[NDArray['S'], NDArray['S']]:
        """
        computes VaR of the inferred weights for all states and returns the list of all states ordered by VaR, along with VaR values
        """
        assert len(self.chain)

        start_timer("VaR")
        evds = np.zeros((self.env.n_states, MCMC_CHAIN_LENGTH))
        evd_dict = EVD_CACHE[inferred_wd.key] if USE_EVD_CACHE else {}
        self.evd_cache_hits = 0
        value_func = dp_solver_utils.compute_policy_soft_value if SOFT_EVD else dp_solver_utils.compute_policy_value
        for i in range(MCMC_CHAIN_LENGTH):
            if False and i % (MCMC_CHAIN_LENGTH / 10) == 0:
                print(f"VaR iter {i}")
                print_timer("VaR")
                print(f"VaR total policy iters: {dp_solver_utils.CUMUL_ITERS[2]}")
            wd = self.resample[i]
            if wd.key in evd_dict:
                evds[:, i] = evd_dict[wd.key]
                self.evd_cache_hits += 1
            else:
                inferred_V: Vs = value_func(self.env, wd.reward, inferred_wd.policy, wd.V, EPS)
                V = wd.V if MCMC_SOFT_Q == SOFT_EVD else value_func(self.env, wd.reward, wd.policy, wd.V, EPS)
                evd = V - inferred_V
                evd_dict[wd.key] = evd
                evds[:, i] = evd

        print('evd cache ratio:', 100 * self.evd_cache_hits // MCMC_CHAIN_LENGTH, '%')

        evds.sort(axis=1)
        VaR_index = int(MCMC_CHAIN_LENGTH * VaR)
        eval_VaR = evds[:, VaR_index]
        VaR_states = np.argsort(-eval_VaR)
        VaR_values = eval_VaR[VaR_states]
        return VaR_states, VaR_values

    def debug_collect_data(self, true_w: Weights, true_policy: Policy) -> Tuple[List[Weights], List[Policy], List[float], List[float], WData, float]:
        w_diffs, p_diffs, posteriors, resample_posteriors = [], [], [], []
        smallest_diff = np.PINF
        nearest_wd = None
        for wd in self.chain:
            w_diff = np.linalg.norm(true_w - wd.w)
            w_diffs.append(w_diff)
            p_diff = np.sum(np.abs(true_policy - wd.policy))
            p_diffs.append(p_diff)
            posteriors.append(wd.posterior)
            if w_diff < smallest_diff:
                smallest_diff = w_diff
                nearest_wd = wd
        for wd in self.resample:
            resample_posteriors.append(wd.posterior)
        return w_diffs, p_diffs, posteriors, resample_posteriors, nearest_wd, smallest_diff

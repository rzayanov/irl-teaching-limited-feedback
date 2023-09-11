import dp_solver_utils
import mdp_solver_class as mdp_solver
from const import *
from dimensions import *
from env import Env
from np_utils import lock


class Learner:

    def __init__(self, env: Env, eta: float, num_episode: int, len_episode: int, cross_ent: bool, init_w_t: Weights) -> None:
        self.env: Env = env
        self.eta: float = eta
        w_count = Learner.get_w_count(env)
        if init_w_t is not None:
            w_t = init_w_t
        else:
            w_t = np.ones(w_count)
        self.w_t: Weights = w_t

        self.num_episode: int = num_episode
        self.len_episode: int = len_episode
        self.cross_ent = cross_ent
        # unused
        self.radius: float = 10 * np.sqrt(env.n_features)

        self._quadratic_reward()
        self.solver: mdp_solver.solver = mdp_solver.solver(self.env, self.reward, "ce" if self.cross_ent else "stochastic", np.zeros(env.n_states), None, self.w_t)
        self.V_disp = self.solver.V_disp
        self.policy_disp = self.solver.policy_disp
        # state visitation frequency
        self.rho: Rho = self.solver.compute_exp_rho_bellman()
        lock(self.rho)
        # real value of the policy
        self.exp_reward: float = self.env.reward_for_rho(self.rho)

        if False:
            V_soft = dp_solver_utils.compute_policy_soft_value(self.env, self.reward, self.solver.policy, self.solver.V, EPS)
            assert np.allclose(V_soft, self.solver.V, EPS)

            V_real = dp_solver_utils.compute_policy_value(self.env, self.env.true_reward, self.solver.policy, self.solver.V, EPS)
            exp_reward2 = np.dot(V_real, self.env.D_init)
            assert self.exp_reward == exp_reward2

    @staticmethod
    def l1_normalize(w: Weights) -> Weights:
        """ l1-normalizes an array in-place """
        norm = np.sum(np.abs(w))
        w /= norm
        return w

    @staticmethod
    def get_w_count(env: Env):
        w_count = env.n_features
        if LEARNER_QUAD_MUL:
            w_count *= 2
        return w_count

    @staticmethod
    def calc_reward(w_t: Weights, features: FMtx) -> Reward:
        """ R(s) = sum(w_i * f_i(s)) + sum(w_j * f_i(s))^2 """
        n_features = features.shape[1]
        r_lin = np.dot(features, w_t[:n_features])
        r = r_lin
        if LEARNER_QUAD_MUL:
            r_quad = np.dot(features, w_t[n_features:]) ** 2 * LEARNER_QUAD_MUL
            r = r_lin + r_quad
        return r

    def _quadratic_reward(self) -> None:
        # used only for solver
        self.reward: Reward = Learner.calc_reward(self.w_t, self.env.feature_matrix)
        lock(self.reward)
        self.reward_disp = self.env.get_state_array_disp(self.reward) if CALC_DISP_DATA else None

    def _reward_gradient(self) -> NDArray['S,W']:
        """ Gradient of state reward relative to w """
        features = self.env.feature_matrix
        grad_lin = features
        grad = grad_lin
        if LEARNER_QUAD_MUL:
            ig = np.dot(features, self.w_t[self.env.n_features:])
            grad_quad = 2 * np.transpose(ig * np.transpose(features)) * LEARNER_QUAD_MUL
            grad = np.append(grad_lin, grad_quad, axis=1)
        return grad

    def update_step(self, rho_exp: Rho, algo, state: int = 0, n_init_tasks: int = 0, many_states: List[int] = None,
                    calc_rho: bool = True, ep: Episode = None) -> None:

        if self.cross_ent:
            grad_diff: NDArray['W'] = np.zeros_like(self.w_t)
            for s, a in ep:
                g_learner_lin = np.dot(self.solver.policy[s], self.env.action_features[:, s])
                g_expert_lin = self.env.action_features[a, s]
                g_learner = g_learner_lin
                g_expert = g_expert_lin
                if LEARNER_QUAD_MUL:
                    ig: NDArray['A'] = np.dot(self.env.action_features[:, s], self.w_t[self.env.n_features:])
                    ig2: NDArray['A,F'] = np.transpose(ig * np.transpose(self.env.action_features[:, s]))
                    g_learner_quad = 2 * np.dot(self.solver.policy[s], ig2) * LEARNER_QUAD_MUL
                    g_expert_quad = 2 * np.dot(self.w_t[self.env.n_features:], self.env.action_features[a,s]) * self.env.action_features[a,s]  * LEARNER_QUAD_MUL
                    g_learner = np.append(g_learner_lin, g_learner_quad)
                    g_expert = np.append(g_expert_lin, g_expert_quad)
                grad_diff += g_learner - g_expert
        else:
            rho_learner: Rho
            if algo == "exp":
                # used in omni to find optimal w
                rho_learner = self.rho
            elif algo == "init_tasks":
                # used to pre-train initial tasks
                rho_learner = self.rho_for_init_tasks(n_init_tasks)
            elif algo == "task":
                # not used
                rho_learner = self.rho_task(self.env.state_to_task(state))
            elif algo == "state":
                # used for actual training
                rho_learner = self.rho_from_state(state)
            elif algo == "many_states":
                rho_learner = self.rho_from_many_states(many_states)
            else:
                print("Improper update algorithm!")
                raise

            grad = self._reward_gradient()
            grad_learner: NDArray['W'] = np.dot(rho_learner, grad)
            grad_expert: NDArray['W'] = np.dot(rho_exp, grad)
            grad_diff = grad_learner - grad_expert

        self.w_t -= self.eta * ETA_MUL * grad_diff

        if CROSS_ENT:
            max_radius = 100 * np.sqrt(self.env.n_features * 2)
            curr_norm = np.linalg.norm(self.w_t)
            if max_radius < curr_norm:
                self.w_t *= max_radius / curr_norm

        # print(self.w_t.reshape(2, 4))
        # print('learner norm', np.sum(np.abs(self.w_t)).round(2))

        # reward function based on new weight
        self._quadratic_reward()
        # policy for the reward
        self.solver = mdp_solver.solver(self.env, self.reward, "ce" if self.cross_ent else "stochastic", self.solver.V, None, self.w_t)
        # expected rho of the policy
        self.rho = self.solver.compute_exp_rho_bellman() if calc_rho else None
        lock(self.rho)
        # real policy value
        self.exp_reward = self.env.reward_for_rho(self.rho) if calc_rho else np.nan

    def rho_from_state(self, state: int, sampling: bool = False) -> Rho:
        if sampling or RHO_STATE_SAMPLING:
            # not used
            rho_s = self.solver.compute_exp_rho_sampling(self.num_episode, self.len_episode, state)
        elif RHO_STATE_BELLMAN:
            rho_s = self.solver.compute_exp_rho_bellman_state(state)
        else:
            # state must be the first state of a task
            end_state = state + (2 * self.env.road_length)
            mask = np.zeros(self.env.n_states)
            mask[np.arange(state, end_state)] = self.env.n_tasks * self.env.n_lanes_per_task
            # mask[self.env.n_states - 1] = 1
            rho_s = mask * self.rho

        return rho_s

    def rho_from_many_states(self, many_states: List[int]) -> Rho:
        if RHO_STATE_SAMPLING:
            rho_sampling = np.zeros(self.env.n_states)
            for state in many_states:
                rho_s = self.solver.compute_exp_rho_sampling(self.num_episode, self.len_episode, state)
                rho_sampling += rho_s / len(many_states)
            return rho_sampling
        elif RHO_STATE_BELLMAN:
            D_init = np.zeros(self.env.n_states)
            for state in many_states:
                D_init[state] += 1 / len(many_states)
            rho_bellman = self.solver.compute_exp_rho_bellman(D_init)
            return rho_bellman
        else:
            mask = np.zeros(self.env.n_states)
            mul = self.env.n_tasks * self.env.n_lanes_per_task / len(many_states)
            for state in many_states:
                end_state = state + (2 * self.env.road_length)
                mask[np.arange(state, end_state)] += mul
            # mask[self.env.n_states - 1] = 1
            rho_fast = mask * self.rho
            return rho_fast

    def _calc_sampling_error(self, state: int) -> None:
        rho_s = self.solver.compute_exp_rho_bellman_state(state)
        vals = (5, 10, 20, 40, 80, 160, 320)
        dists = np.zeros((len(vals), len(vals)))
        ne_idx = 0
        for ne in vals:
            print(f'Calculating sampling error for {ne} samples...')
            le_idx = 0
            for le in vals:
                rho_s2 = self.solver.compute_exp_rho_sampling(ne, le, state)
                dist = np.sum(np.abs(rho_s - rho_s2))
                dists[ne_idx, le_idx] = dist
                le_idx += 1
            ne_idx += 1
        # length is more important
        print(dists.round())

    def rho_task(self, task: int) -> Rho:
        """ unused """
        D_init = np.zeros((self.env.n_states))
        for j in range(self.env.n_lanes_per_task):
            D_init[(task + j * self.env.n_tasks) * 2 * self.env.road_length] += (1 / (self.env.n_lanes_per_task))

        rho = self.solver.compute_exp_rho_bellman(D_init)
        return rho

    def rho_for_init_tasks(self, n_tasks: int) -> Rho:
        """ used to pre-train initial tasks """
        D_init = self.env.D_init_for_init_tasks(n_tasks)
        rho = self.solver.compute_exp_rho_bellman(D_init)
        return rho

    def update_eta(self, eta: float) -> None:
        self.eta = eta

    def scale_eta(self, factor: float) -> None:
        self.eta /= factor

    def per_task_reward(self):
        """ unused """
        reward_list = np.zeros(self.env.n_tasks)
        for l in range(self.env.n_tasks):
            D_init = self.env.D_init_for_task(l)
            rho = self.solver.compute_exp_rho_bellman(D_init)
            reward_list[l] = self.env.reward_for_rho(rho)

        return reward_list

    def value_task(self, task):
        """ unused, average value of a task """
        D_init = np.zeros((self.env.n_states))
        for k in range(self.env.n_lanes_per_task):
            D_init[(task + k * self.env.n_tasks) * 2 * self.env.road_length] = (1 / self.env.n_lanes_per_task)

        V_task = np.dot(D_init, self.solver.V)
        return V_task

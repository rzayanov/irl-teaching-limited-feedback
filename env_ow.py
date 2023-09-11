import numpy.random as rn
from scipy import sparse
from scipy.sparse.csr import csr_matrix

from const import *
from dimensions import *
from env import Env
from ow2 import Objectworld
from timer import start_timer, print_timer


class EnvOW(Env):
    """
    y axis points downward

    actions: R, D, L, U, S

    colours:
        0 is red, 1 is green

    features:
        inner red dist, outer red dist, inner green dist, outer green dist

    non-linear reward:
        close to inner green: good
        close to inner red: ok
        equidistant: bad
    """

    def __init__(self, n_tasks: int = 8, gamma: float = .99) -> None:
        if ENV_LINEAR:
            n_colours = 0
            n_objects = 0
            if OW_MANUAL_ONE_HOT:
                manual_map = np.random.randint(0, OW_MANUAL_FEATURES, OW_GRID_SZ ** 2)
                if OW_SEP:
                    manual_map = np.zeros(OW_GRID_SZ ** 2, int)
                    for i in range(OW_MANUAL_FEATURES):
                        left = bool(i % 2)
                        top = bool(i // 2 % 2)
                        x = np.random.randint(OW_GRID_SZ // 2) + (0 if left else OW_GRID_SZ // 2)
                        y = np.random.randint(OW_GRID_SZ // 2) + (0 if top else OW_GRID_SZ // 2)
                        manual_map[y * OW_GRID_SZ + x] = i
                # manual_map = np.array((
                #     0, 0, 0, 0, 0,
                #     0, 0, 1, 0, 0,
                #     0, 0, 1, 0, 2,
                #     2, 0, 1, 0, 0,
                #     0, 0, 1, 0, 0,
                # ))
                manual_matrix = EnvOW.make_manual_matrix(OW_MANUAL_FEATURES, manual_map)
            else:
                manual_map = np.zeros((OW_GRID_SZ ** 2))
                manual_matrix = np.random.random_sample((OW_GRID_SZ ** 2, OW_MANUAL_FEATURES))
            manual_w = np.arange(-5, 5, 10 / OW_MANUAL_FEATURES)
            assert len(manual_w) == OW_MANUAL_FEATURES
        else:
            n_colours = 2
            n_objects = OW_GRID_OBJS
            manual_map = np.zeros((OW_GRID_SZ ** 2))
            manual_matrix = np.zeros((OW_GRID_SZ ** 2, 0))
            manual_w = np.zeros(0)
        ow = Objectworld(OW_GRID_SZ, n_objects, n_colours, manual_matrix, manual_w, OW_WIND)

        self.n_tasks = n_tasks
        self.n_states = ow.n_states
        self.grid_size = ow.grid_size
        self.n_actions = ow.n_actions
        self.gamma = gamma
        T_dense = np.swapaxes(ow.transition_probability, 0, 1)
        # self.T_dense = T_dense
        self.T = []
        for a in range(self.n_actions):
            self.T.append(sparse.csr_matrix(T_dense[a]))

        self.objects = ow.objects
        self.objects_disp = self._calc_objects_disp() if CALC_DISP_DATA else None

        self.n_colours = ow.n_colours
        self.manual_map = manual_map
        self.manual_map_disp = self.get_state_array_disp(self.manual_map) if CALC_DISP_DATA else None
        self.manual_w = manual_w
        self.ideal_lin_w = manual_w

        self.D_init: P0 = np.zeros(self.n_states)
        # state to task index
        self.initial_states: List[int] = []
        if self.n_tasks == self.n_states:
            for s in range(self.n_states):
                self.initial_states.append(s)
                self.D_init[s] = 1 / self.n_tasks
        elif self.n_tasks > self.n_states:
            raise ValueError('too many tasks')
        else:
            for i in range(self.n_tasks):
                while True:
                    p = (rn.randint(self.grid_size), rn.randint(self.grid_size))
                    s = self.point_to_int(p)
                    if s not in self.initial_states and p not in self.objects:
                        break
                if OW_SEP:
                    left = bool(i % 2)
                    top = bool(i // 2 % 2)
                    p = 0 if left else self.grid_size - 1, 0 if top else self.grid_size - 1
                    s = self.point_to_int(p)
                self.initial_states.append(s)
                self.D_init[s] = (1 / self.n_tasks)
        assert np.allclose(self.D_init.sum(), 1)

        self.feature_matrix = ow.feature_matrix
        self.n_features = self.feature_matrix.shape[1]

        self.true_reward = ow.calc_all_reward()
        self.true_reward_disp = self.get_state_array_disp(self.true_reward) if CALC_DISP_DATA else None

        # self.print_map()

    def point_to_int(self, p: Point) -> int:
        return p[0] + p[1] * self.grid_size

    def print_map(self, state: int = -1):
        red = "\u001b[31m"
        green = "\u001b[32m"
        blue = "\u001b[34m"
        magenta = "\u001b[35m"
        bg_black = "\u001b[40m"
        bg_grey = "\u001b[100m"
        bg_lred = "\u001b[41m"
        bg_lgreen = "\u001b[42m"
        bg_lblue = "\u001b[44m"
        reset = "\u001b[0m"

        for y in range(self.grid_size):
            map_line = ''
            init_line = ''
            for x in range(self.grid_size):
                p = (x, y)
                s = self.point_to_int(p)
                r = self.true_reward[s]
                if False and np.sometrue(self.feature_matrix[s, 2 * self.n_colours:]):
                    cell = bg_lred if self.feature_matrix[s, 2 * self.n_colours] else bg_lgreen
                else:
                    cell = bg_lgreen if r > R_HIGH else '' if r == R_HIGH else bg_grey if r >= R_MED else bg_black if r >= R_LOW else bg_lred
                if p in self.objects:
                    cell += green if self.objects[p].outer_colour else red
                    cell += 's' if s == state else '•'
                else:
                    cell += magenta
                    task_str = str(self.initial_states.index(s)) if s in self.initial_states else ' '
                    cell += 's' if s == state else task_str
                cell += reset
                map_line += cell
                init_line += 'x' if s in self.initial_states else '.'
            print(map_line)

    def get_policy_disp(self, policy: Policy, do_print: bool = False) -> NDArray['']:
        cells = '→↓←↑·'
        res = np.zeros((self.grid_size, self.grid_size), dtype=str)
        for y in range(self.grid_size):
            map_line = ''
            for x in range(self.grid_size):
                p = (x, y)
                s = self.point_to_int(p)
                a = np.argmax(policy[s])
                cell = cells[a]
                res[y, x] = cell
                map_line += cell
            if do_print:
                print(map_line)
        return res

    def get_state_array_disp(self, arr: NDArray['S']) -> NDArray['']:
        return arr.reshape((self.grid_size, self.grid_size))

    def policy_transition_matrix(self, policy: Policy) -> csr_matrix:
        T_pi = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                T_pi[s] += policy[s, a] * self.T[a][s, :]
        if EXTRA_ASSERT: assert np.allclose(T_pi.sum(axis=1), 1)
        return sparse.csr_matrix(np.transpose(T_pi))

    def reward_for_rho(self, rho: Rho) -> float:
        return np.dot(self.true_reward, rho)

    def state_to_task_instance(self, state: int) -> int:
        return self.initial_states.index(state)

    def state_to_task(self, state: int) -> int:
        return self.initial_states.index(state)

    def D_init_for_init_tasks(self, n_init_tasks: int) -> P0:
        D_init = np.zeros(self.n_states)
        for i in range(n_init_tasks):
            s = self.initial_states[i]
            D_init[s] += 1 / n_init_tasks
        return D_init

    def D_init_for_task(self, task: int) -> P0:
        """ unused """
        D_init = np.zeros(self.n_states)
        D_init[self.initial_states[task]] = 1
        return D_init

    @classmethod
    def make_manual_matrix(cls, n_manual: int, fs: NDArray['S']) -> NDArray['S,']:
        res = np.zeros((len(fs), n_manual))
        for s in range(len(fs)):
            res[s, fs[s]] = 1
        return res

    def _calc_objects_disp(self) -> NDArray['S,S']:
        res = np.full((self.grid_size, self.grid_size), np.nan)
        for (x, y) in self.objects.keys():
            res[y, x] = self.objects[x, y].inner_colour
        return res

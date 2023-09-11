"""
All possible constraints:

.5:
  0 > grass
1:
  0 > stone
2:
  0 > carf
  grass > carf
  stone > carf
2.5:
  0 > grass_carf
3:
  0 > stone_carf
5:
  0 > car
  grass > car
  stone > car
  carf > car
  grass_carf > car
  stone_carf > car
  0 > pedf
  grass > pedf
5.5:
  0 > grass_pedf
7:
  0 > carf_car
  stone > carf_car
  grass > carf_car
10:
  0 > ped
  grass > ped
  pedf > ped
  grass_pedf > ped
15:
  0 > pedf_ped
  grass > pedf_ped

"""
from typing import Tuple, Set

import dp_solver_utils
from const import *
from dimensions import *
from env import Env
from teacher import Teacher


class Scot2:
    def __init__(self, env: Env, teacher: Teacher):
        self.env: Env = env
        self.teacher: Teacher = teacher
        self.seq: Tuple[Trajectory] = self.get_seq()

    def get_seq(self) -> Tuple[Trajectory]:
        cons, all_road_cons = self.get_all_cons()
        reduced = self.reduce(cons)
        print('orig', len(cons), 'reduced', len(reduced))

        uncovered = reduced.copy()
        seq = []
        while len(uncovered):
            sss = sorted(all_road_cons, key = lambda x: -len(uncovered.intersection(x[1])))
            best_s = sss[0][0]
            uncovered.difference_update(sss[0][1])
            # rho not really used
            episode, rho = self.teacher.expert.sample_trajectory_from_state(self.teacher.len_episode, best_s)
            seq.append((rho, episode))
        # maybe shuffle
        # s_seq = sorted(seq, key=self.calc_traj_neg_r)
        return tuple(seq)

    def get_all_cons(self) -> Tuple[Set[str], List[Tuple[int,Set[str]]]]:
        road_count = self.env.n_tasks * self.env.n_lanes_per_task
        road_s_count = self.env.road_length * 2
        # collect constraints
        cons: Set[str] = set()
        all_road_cons: List[Tuple[int,Set[str]]] = []
        for road_idx in range(road_count):
            road_cons = set()
            all_road_cons.append((road_idx, road_cons))
            init_state = road_idx * road_s_count
            for s in range(init_state, init_state + road_s_count - 2, 2):
                f_l = self.env.feature_matrix[s + 2].astype(int)
                f_r = self.env.feature_matrix[s + 3].astype(int)
                as_l = self.teacher.expert.policy[s]
                as_r = self.teacher.expert.policy[s + 1]
                if np.array_equal(f_l, f_r):
                    assert np.allclose(as_l.round(2), .33) and np.allclose(as_l.round(2), .33)
                    continue
                a_l = np.argmax(as_l)
                a_r = np.argmax(as_r)
                assert as_l[a_l] == 1 and a_l != A_LE and as_r[a_r] == 1 and a_r != A_RI
                if a_l == A_ST:  # prefer left lane
                    assert a_r == A_LE
                    con = f_l - f_r
                else:
                    assert a_r == A_ST
                    con = f_r - f_l
                con_str = ' '.join([str(f) for f in con])
                cons.add(con_str)
                road_cons.add(con_str)
        return cons, all_road_cons

    def calc_traj_neg_r(self, traj: Trajectory):
        ep: Episode = traj[1]
        res = 0
        for s, a in ep:
            res += self.env.true_reward[s]
        return -res

    def get_next(self, iteration: int) -> Tuple[Rho, int, Episode]:
        opt_rho, opt_ep = self.seq[(iteration % len(self.seq))]
        opt_state: int = opt_ep[0][0]
        return dp_solver_utils.calc_episode_rho(self.env, opt_ep), opt_state, opt_ep

    def reduce(self, cons: Set[str]) -> Set[str]:
        grass                = '0 -1 0 0 0 0 0 0'
        stone                = '-1 0 0 0 0 0 0 0'
        carf                 = '0 0 0 0 0 -1 0 0'
        grass__carf          = '0 1 0 0 0 -1 0 0'
        stone__carf          = '1 0 0 0 0 -1 0 0'
        grass_carf           = '0 -1 0 0 0 -1 0 0'
        stone_carf           = '-1 0 0 0 0 -1 0 0'
        car                  = '0 0 -1 0 0 0 0 0'
        grass__car           = '0 1 -1 0 0 0 0 0'
        stone__car           = '1 0 -1 0 0 0 0 0'
        carf__car            = '0 0 -1 0 0 1 0 0'
        grass_carf__car      = '0 1 -1 0 0 1 0 0'
        stone_carf__car      = '1 0 -1 0 0 1 0 0'
        pedf                 = '0 0 0 0 0 0 -1 0'
        grass__pedf          = '0 1 0 0 0 0 -1 0'
        grass_pedf           = '0 -1 0 0 0 0 -1 0'
        carf_car             = '0 0 -1 0 0 -1 0 0'
        stone__carf_car      = '1 0 -1 0 0 -1 0 0'
        grass__carf_car      = '0 1 -1 0 0 -1 0 0'
        ped                  = '0 0 0 -1 0 0 0 0'
        grass__ped           = '0 1 0 -1 0 0 0 0'
        pedf__ped            = '0 0 0 -1 0 0 1 0'
        grass_pedf__ped      = '0 1 0 -1 0 0 1 0'
        pedf_ped             = '0 0 0 -1 0 0 -1 0'
        grass__pedf_ped      = '0 1 0 -1 0 0 -1 0'

        c = cons.copy()
        if grass in c and grass__carf in c or stone in c and stone__carf in c:
            c.discard(carf)
        if grass in c and grass__car in c or stone in c and stone__car in c or carf in c and carf__car in c:
            c.discard(car)
        if grass in c and grass__pedf in c:
            c.discard(pedf)
        if stone in c and stone__carf_car in c or grass in c and grass__carf_car in c:
            c.discard(carf_car)
        if grass in c and grass__ped in c or pedf in c and pedf__ped in c:
            c.discard(ped)
        if grass in c and grass__pedf_ped in c:
            c.discard(pedf_ped)

        if grass in c and grass_carf__car in c or stone in c and stone_carf__car in c:
            c.discard(carf__car)
        if grass in c and grass_pedf__ped in c:
            c.discard(pedf__ped)

        if grass_carf in c and grass_carf__car in c or stone_carf in c and stone_carf__car in c:
            c.discard(car)
        if grass_carf in c and grass_pedf__ped in c:
            c.discard(ped)

        if grass in c and grass__carf_car in c or stone in c and stone__carf_car in c:
            c.discard(carf_car)
        if grass in c and grass__pedf_ped in c:
            c.discard(pedf_ped)

        if carf in c and grass_carf__car in c or carf in c and stone_carf__car in c:
            c.discard(grass__car)
        if carf in c and grass_pedf__ped in c:
            c.discard(grass__ped)

        return c

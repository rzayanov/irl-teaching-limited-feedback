
import math
from typing import Dict

import numpy.random as rn

from const import *
from dimensions import *
from quadratic_learner import Learner


class OWObject(object):
    def __init__(self, inner_colour: int, outer_colour: int):
        self.inner_colour: int = inner_colour
        self.outer_colour: int = outer_colour

    def __str__(self):
        return f"<OWObject (In: {self.inner_colour}) (Out: {self.outer_colour})>"


class Objectworld():
    actions: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    n_actions: int
    n_states: int
    grid_size: int
    wind: float
    n_objects: int
    n_colours: int
    objects: Dict[Tuple[int, int], OWObject]
    transition_probability: NDArray["S,A,S"]

    def __init__(
            self, grid_size: int, n_objects: int, n_colours: int,
            manual_matrix: NDArray['S,'], manual_w: NDArray[''],
            wind: float
    ):

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
        self.n_actions: int = len(self.actions)
        self.n_states: int = grid_size ** 2
        self.grid_size: int = grid_size
        self.wind: float = wind
        self.n_objects: int = n_objects
        self.n_colours: int = n_colours

        # Generate objects.
        self.objects: Dict[Tuple[int, int], OWObject] = {}
        used_colours = np.zeros(2 * self.n_colours)
        for _ in range(self.n_objects):
            inner_colour = rn.randint(self.n_colours)
            outer_colour = rn.randint(self.n_colours)
            used_colours[inner_colour * 2] = 1
            used_colours[outer_colour * 2 + 1] = 1
            obj: OWObject = OWObject(inner_colour, outer_colour)
            while True:
                x: int = rn.randint(self.grid_size)
                y: int = rn.randint(self.grid_size)
                if (x, y) not in self.objects:
                    break
            self.objects[x, y] = obj
        if np.sum(used_colours) != used_colours.shape[0]:
            raise Exception('not all colours are used')

        self.manual_matrix = manual_matrix
        self.manual_w = manual_w

        # Preconstruct the transition probability array.
        tp = np.zeros((self.n_states, self.n_actions, self.n_states))
        for first_row_state in range(0, self.n_states, self.grid_size):
            self._tp_row(tp, first_row_state)
        self.transition_probability: NDArray["S,A,S"] = tp

        for a in range(self.n_actions):
            tp_a = self.transition_probability[:, a, :]
            proba_sum: np.ndarray = tp_a.sum(axis=1)
            assert np.allclose(proba_sum, 1)

        self.feature_matrix: FMtx = self._calc_feature_matrix()

    def _calc_feature_matrix(self) -> FMtx:
        if ENV_LINEAR:
            return self.manual_matrix
        return np.array([self._calc_nonlin_fv(i) for i in range(self.n_states)])

    def _calc_nonlin_fv(self, state: int) -> NDArray['F']:
        # this can be vectorized

        sx, sy = self.int_to_point(state)

        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for (x, y) in self.objects.keys():
            dist = math.hypot((x - sx), (y - sy))
            obj = self.objects[x, y]
            if obj.inner_colour in nearest_inner:
                if dist < nearest_inner[obj.inner_colour]:
                    nearest_inner[obj.inner_colour] = dist
            else:
                nearest_inner[obj.inner_colour] = dist
            if obj.outer_colour in nearest_outer:
                if dist < nearest_outer[obj.outer_colour]:
                    nearest_outer[obj.outer_colour] = dist
            else:
                nearest_outer[obj.outer_colour] = dist

        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        res = np.zeros(2 * self.n_colours)
        i = 0
        for c in range(self.n_colours):
            res[i] = nearest_inner[c]
            i += 1
            res[i] = nearest_outer[c]
            i += 1

        return res

    def int_to_point(self, i: int) -> Point:
        return i % self.grid_size, i // self.grid_size

    def calc_all_reward(self) -> Reward:
        if not self.n_objects:
            return np.dot(self.manual_matrix, self.manual_w)

        ideal_w = np.array([-5, 0, -5, 0, 1.5, 0, -1, 0])
        res = Learner.calc_reward(ideal_w, self.feature_matrix)

        return res

    def _tp_row(self, tp: NDArray['S,A,S'], first_row_state: int):
        wind_proba = self.wind / 4
        wind_proba_2 = wind_proba * 2
        move_proba = 1 - self.wind + wind_proba
        move_proba_2 = move_proba + wind_proba
        top_row = first_row_state == 0
        bottom_row = first_row_state == self.n_states - self.grid_size
        state_left = first_row_state - 1
        state_right = first_row_state + 1
        state_up = first_row_state - self.grid_size
        state_down = first_row_state + self.grid_size
        for state in range(first_row_state, first_row_state + self.grid_size):
            left_col = state == first_row_state
            right_col = state == first_row_state + self.grid_size - 1
            if top_row:
                if left_col:
                    tp[state, 0, state_right] = move_proba
                    tp[state, 0, state_down] = wind_proba
                    tp[state, 0, state] = wind_proba_2
                    tp[state, 1, state_right] = wind_proba
                    tp[state, 1, state_down] = move_proba
                    tp[state, 1, state] = wind_proba_2
                    tp[state, 2, state_right] = wind_proba
                    tp[state, 2, state_down] = wind_proba
                    tp[state, 2, state] = move_proba_2
                    tp[state, 3, state_right] = wind_proba
                    tp[state, 3, state_down] = wind_proba
                    tp[state, 3, state] = move_proba_2
                    tp[state, 4, state_right] = wind_proba
                    tp[state, 4, state_down] = wind_proba
                    tp[state, 4, state] = move_proba_2
                elif not right_col:
                    tp[state, 0, state_right] = move_proba
                    tp[state, 0, state_down] = wind_proba
                    tp[state, 0, state_left] = wind_proba
                    tp[state, 0, state] = wind_proba
                    tp[state, 1, state_right] = wind_proba
                    tp[state, 1, state_down] = move_proba
                    tp[state, 1, state_left] = wind_proba
                    tp[state, 1, state] = wind_proba
                    tp[state, 2, state_right] = wind_proba
                    tp[state, 2, state_down] = wind_proba
                    tp[state, 2, state_left] = move_proba
                    tp[state, 2, state] = wind_proba
                    tp[state, 3, state_right] = wind_proba
                    tp[state, 3, state_down] = wind_proba
                    tp[state, 3, state_left] = wind_proba
                    tp[state, 3, state] = move_proba
                    tp[state, 4, state_right] = wind_proba
                    tp[state, 4, state_down] = wind_proba
                    tp[state, 4, state_left] = wind_proba
                    tp[state, 4, state] = move_proba
                else:
                    tp[state, 0, state_down] = wind_proba
                    tp[state, 0, state_left] = wind_proba
                    tp[state, 0, state] = move_proba_2
                    tp[state, 1, state_down] = move_proba
                    tp[state, 1, state_left] = wind_proba
                    tp[state, 1, state] = wind_proba_2
                    tp[state, 2, state_down] = wind_proba
                    tp[state, 2, state_left] = move_proba
                    tp[state, 2, state] = wind_proba_2
                    tp[state, 3, state_down] = wind_proba
                    tp[state, 3, state_left] = wind_proba
                    tp[state, 3, state] = move_proba_2
                    tp[state, 4, state_down] = wind_proba
                    tp[state, 4, state_left] = wind_proba
                    tp[state, 4, state] = move_proba_2
            elif not bottom_row:
                if left_col:
                    tp[state, 0, state_right] = move_proba
                    tp[state, 0, state_down] = wind_proba
                    tp[state, 0, state_up] = wind_proba
                    tp[state, 0, state] = wind_proba
                    tp[state, 1, state_right] = wind_proba
                    tp[state, 1, state_down] = move_proba
                    tp[state, 1, state_up] = wind_proba
                    tp[state, 1, state] = wind_proba
                    tp[state, 2, state_right] = wind_proba
                    tp[state, 2, state_down] = wind_proba
                    tp[state, 2, state_up] = wind_proba
                    tp[state, 2, state] = move_proba
                    tp[state, 3, state_right] = wind_proba
                    tp[state, 3, state_down] = wind_proba
                    tp[state, 3, state_up] = move_proba
                    tp[state, 3, state] = wind_proba
                    tp[state, 4, state_right] = wind_proba
                    tp[state, 4, state_down] = wind_proba
                    tp[state, 4, state_up] = wind_proba
                    tp[state, 4, state] = move_proba
                elif not right_col:
                    tp[state, 0, state_right] = move_proba
                    tp[state, 0, state_down] = wind_proba
                    tp[state, 0, state_left] = wind_proba
                    tp[state, 0, state_up] = wind_proba
                    tp[state, 1, state_right] = wind_proba
                    tp[state, 1, state_down] = move_proba
                    tp[state, 1, state_left] = wind_proba
                    tp[state, 1, state_up] = wind_proba
                    tp[state, 2, state_right] = wind_proba
                    tp[state, 2, state_down] = wind_proba
                    tp[state, 2, state_left] = move_proba
                    tp[state, 2, state_up] = wind_proba
                    tp[state, 3, state_right] = wind_proba
                    tp[state, 3, state_down] = wind_proba
                    tp[state, 3, state_left] = wind_proba
                    tp[state, 3, state_up] = move_proba
                    tp[state, 4, state_right] = wind_proba
                    tp[state, 4, state_down] = wind_proba
                    tp[state, 4, state_left] = wind_proba
                    tp[state, 4, state_up] = wind_proba
                    tp[state, 4, state] = 1 - self.wind
                else:
                    tp[state, 0, state_down] = wind_proba
                    tp[state, 0, state_left] = wind_proba
                    tp[state, 0, state_up] = wind_proba
                    tp[state, 0, state] = move_proba
                    tp[state, 1, state_down] = move_proba
                    tp[state, 1, state_left] = wind_proba
                    tp[state, 1, state_up] = wind_proba
                    tp[state, 1, state] = wind_proba
                    tp[state, 2, state_down] = wind_proba
                    tp[state, 2, state_left] = move_proba
                    tp[state, 2, state_up] = wind_proba
                    tp[state, 2, state] = wind_proba
                    tp[state, 3, state_down] = wind_proba
                    tp[state, 3, state_left] = wind_proba
                    tp[state, 3, state_up] = move_proba
                    tp[state, 3, state] = wind_proba
                    tp[state, 4, state_down] = wind_proba
                    tp[state, 4, state_left] = wind_proba
                    tp[state, 4, state_up] = wind_proba
                    tp[state, 4, state] = move_proba
            else:
                if left_col:
                    tp[state, 0, state_right] = move_proba
                    tp[state, 0, state_up] = wind_proba
                    tp[state, 0, state] = wind_proba_2
                    tp[state, 1, state_right] = wind_proba
                    tp[state, 1, state_up] = wind_proba
                    tp[state, 1, state] = move_proba_2
                    tp[state, 2, state_right] = wind_proba
                    tp[state, 2, state_up] = wind_proba
                    tp[state, 2, state] = move_proba_2
                    tp[state, 3, state_right] = wind_proba
                    tp[state, 3, state_up] = move_proba
                    tp[state, 3, state] = wind_proba_2
                    tp[state, 4, state_right] = wind_proba
                    tp[state, 4, state_up] = wind_proba
                    tp[state, 4, state] = move_proba_2
                elif not right_col:
                    tp[state, 0, state_right] = move_proba
                    tp[state, 0, state_left] = wind_proba
                    tp[state, 0, state_up] = wind_proba
                    tp[state, 0, state] = wind_proba
                    tp[state, 1, state_right] = wind_proba
                    tp[state, 1, state_left] = wind_proba
                    tp[state, 1, state_up] = wind_proba
                    tp[state, 1, state] = move_proba
                    tp[state, 2, state_right] = wind_proba
                    tp[state, 2, state_left] = move_proba
                    tp[state, 2, state_up] = wind_proba
                    tp[state, 2, state] = wind_proba
                    tp[state, 3, state_right] = wind_proba
                    tp[state, 3, state_left] = wind_proba
                    tp[state, 3, state_up] = move_proba
                    tp[state, 3, state] = wind_proba
                    tp[state, 4, state_right] = wind_proba
                    tp[state, 4, state_left] = wind_proba
                    tp[state, 4, state_up] = wind_proba
                    tp[state, 4, state] = move_proba
                else:
                    tp[state, 0, state_left] = wind_proba
                    tp[state, 0, state_up] = wind_proba
                    tp[state, 0, state] = move_proba_2
                    tp[state, 1, state_left] = wind_proba
                    tp[state, 1, state_up] = wind_proba
                    tp[state, 1, state] = move_proba_2
                    tp[state, 2, state_left] = move_proba
                    tp[state, 2, state_up] = wind_proba
                    tp[state, 2, state] = wind_proba_2
                    tp[state, 3, state_left] = wind_proba
                    tp[state, 3, state_up] = move_proba
                    tp[state, 3, state] = wind_proba_2
                    tp[state, 4, state_left] = wind_proba
                    tp[state, 4, state_up] = wind_proba
                    tp[state, 4, state] = move_proba_2
            state_left += 1
            state_right += 1
            state_up += 1
            state_down += 1

from typing import List, Union, Tuple

from numpy.typing import NDArray

T0 = 0  # nothing
T1 = 1  # cars
T2 = 2  # stones on right
T3 = 3  # stones randomly + cars
T4 = 4  # grass on right
T5 = 5  # grass randomly + cars
T6 = 6  # grass on right + peds
T7 = 7  # hov on right + police


F0_STONE = 0  # -1
F1_GRASS = 1  # -.5
F2_CAR = 2  # -5
F3_PED = 3  # -10
F4_HOV = 4  # 1 or -5
F5_CAR_IN_F = 5  # -2
F6_PED_IN_F = 6  # -5
F7_POLICE = 7  # 0
F7A_POLICE = -1

FV0_NONE = 0
FV1_STONE = 1
FV2_GRASS = 2
FV3_CAR = 3
FV4_PED = 4
FV5_HOV = 5
FV6_CAR_IN_F = 6
FV7_PED_IN_F = 7

A_ST = 0
A_LE = 1
A_RI = 2


S = 801  # n_states
I = 40  # n_initial_states
A = 3  # n_actions
F = 8  # n_features
W = 16  # n_weights


SA = Tuple[int, int]
Episode = List[SA]
Weights = NDArray['W']
Reward = NDArray['S']
Vs = NDArray['S']
Qs = NDArray['S,A']
P0 = NDArray['S']
FMtx = NDArray['S,F']
Rho = NDArray['S']
Policy = NDArray['S,A']
Trajectory = Tuple[Rho, Episode]



# === object world

A_R = 0
A_D = 1
A_L = 2
A_U = 3
A_S = 4

Point = Tuple[int, int]

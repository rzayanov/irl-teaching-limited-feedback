import configparser
from decimal import Decimal

import numpy as np


CONFIG_NAME = "38"  # 38
# the number of teaching iterations
TEACHING_N_ITERS = 40  # 40 or 500
# stop the teaching process if this loss was achieved
STOP_LOSS = -10  # -10 or .5


def get_eps_round() -> int:
    sign, digits, exponent = Decimal(EPS).as_tuple()
    return 1 - len(digits) - exponent


def getint(key) -> int:
    val = cfg.getint(key)
    if val is None:
        raise configparser.NoOptionError(key, SECTION)
    return val


def getfloat(key) -> float:
    val = cfg.getfloat(key)
    if val is None:
        raise configparser.NoOptionError(key, SECTION)
    return val
    
    
def getboolean(key) -> bool:
    val = cfg.getboolean(key)
    if val is None:
        raise configparser.NoOptionError(key, SECTION)
    return val


def getstr(key) -> str:
    val = cfg.get(key)
    if val is None:
        raise configparser.NoOptionError(key, SECTION)
    return val


SECTION = 'MAIN'

DEBUG_ALL_EXAM_STATES = False
# prepare faster but less precise
FAST_PREPS = False
# calculate data that is used for debugging
CALC_DISP_DATA = False
SAVE_MCMC_DEBUG = False
# enable assertions that are harder to compute
EXTRA_ASSERT = False


USE_W_CACHE = False
USE_EVD_CACHE = False
SAVE_PICKLES = False
LOAD_PICKLES = False


cfg = configparser.ConfigParser()
cfg.read(f"{CONFIG_NAME}.ini")
cfg = cfg[SECTION]

SEED = getint('SEED')
EPS = getfloat('EPS')
EPS_ROUND = get_eps_round()


# use the same environment layout for all seeds
USE_SAME_ENV = getboolean('USE_SAME_ENV')  # False


# === TEACHING ALGORITHMS
# Random teacher - selects a random initial state for a demonstration
TEACH_AGN = getboolean('TEACH_AGN')
# Omniscient teacher - knows the learner model
TEACH_OMN = getboolean('TEACH_OMN')

# SCOT teacher (original implementation)
TEACH_SCOT = getboolean('TEACH_SCOT')
# SCOT teacher (alternative implementation)
TEACH_SCOT2 = getboolean('TEACH_SCOT2')
# "TUnlimF" teacher: has unlimited feedback
TEACH_CUR = getboolean('TEACH_CUR')
# not used
TEACH_CUR_TL = getboolean('TEACH_CUR_TL')
# Black-box teacher (from Kamalaruban et al.)
TEACH_BBOX = getboolean('TEACH_BBOX')

# not used, selects exam state to be the same as the previous demonstration
TEACH_REP = getboolean('TEACH_REP')
# "TLimF" teacher - uses modified VaR algorithm to select exam states
LEARN_VAR = getboolean('LEARN_VAR')
# whether "TLimF" should send demonstrations
TEACH_VAR = getboolean('TEACH_VAR')
# "Unmod" teacher - uses unmodified VaR algorithm to select exam states
TEACH_NOE = getboolean('TEACH_NOE')
# "NoAL" teacher - selects exam states randomly
LEARN_RANDOM = getboolean('LEARN_RANDOM')
# whether "NoAL" should send demonstrations
TEACH_RANDOM = getboolean('TEACH_RANDOM')


# === ENVIRONMENT
GAMMA = getfloat('GAMMA')  # .99
# number of different driving tasks
N_TASKS = getint('N_TASKS')
# number of tasks that the learner already knows
N_INIT_TASKS = getint('N_INIT_TASKS')
# the true reward is linear
ENV_LINEAR = getboolean('ENV_LINEAR')
# the learner's reward model is linear
LEARNER_QUAD_MUL = getfloat('LEARNER_QUAD_MUL')
# use object world or car driving
USE_OW = getboolean('USE_OW')


# === RHO CALCULATION
# calculate rho by sampling
RHO_STATE_SAMPLING = getboolean('RHO_STATE_SAMPLING')
SAMPLING_N_TRAJ = getint('SAMPLING_N_TRAJ')  # 10
# the length of each trajectory
# (out of place -- it's used in curriculum teaching and in sampling rho from state)
TRAJ_LEN = getint('TRAJ_LEN')  # 10
# calculate rho by using Bellman equations
RHO_STATE_BELLMAN = getboolean('RHO_STATE_BELLMAN')


# === LEARNER
# learner uses cross entropy
CROSS_ENT = getboolean('CROSS_ENT')  # False
# learner starts with random weights
LEARNER_RANDOM_W = getboolean('LEARNER_RANDOM_W')
# the limit of the random initial weights
LEARNER_RANDOM_W_MAX = getfloat('LEARNER_RANDOM_W_MAX')  # 0, or RWL1N / w_count * 2
# in MCMC, the initial radius of the sphere to sample from
# (out of place -- this relates mostly to MCMC)
RANDOM_W_L1_NORM = getfloat('RANDOM_W_L1_NORM')
# coefficients when computing the soft policy
ALPHA = getfloat('ALPHA')
ALPHA_I = 1 / ALPHA
# learning coefficient
ETA_MUL = getfloat('ETA_MUL')


# === OBJECT WORLD ENVIRONMENT
OW_GRID_SZ = getint('OW_GRID_SZ')
OW_WIND = getfloat('OW_WIND')  # .3
OW_GRID_OBJS = getint('OW_GRID_OBJS')
OW_MANUAL_FEATURES = getint('OW_MANUAL_FEATURES')
# one cell can have at most one non-zero feature
OW_MANUAL_ONE_HOT = getboolean('OW_MANUAL_ONE_HOT')
R_HIGH = getint('R_HIGH')
R_MED = getint('R_MED')
R_LOW = getint('R_LOW')


# === TEACHER
# compute the optimal learner weights and assume that the real learner has the opposite weights
PESSIMISTIC_INITIAL = getboolean('PESSIMISTIC_INITIAL')  # True
# number of teacher's MCE iterations
TEACHER_MCE_ITERS = getint('TEACHER_MCE_ITERS')  # 0
# how many previous exams to consider in MCE
TEACHER_MCE_EASING = getint('TEACHER_MCE_EASING')  # 0 or MCMC_EASING
TEACHER_MCE_ETA_MUL = getfloat('TEACHER_MCE_ETA_MUL')  # 1
TEACHER_MCE_ETA_SCALE = getboolean('TEACHER_MCE_ETA_SCALE')  # False
# if >0, each demonstration entry contains a state and the teacher's policy distribution from that state
LONG_DEMOS = getboolean('LONG_DEMOS')  # True


# === MCMC
# how to choose the radius of the sphere to sample from:
# - learner: use the norm of the true learner weights (the most knowledgable)
# - inferred: use the norm of the estimated learner weight
# - const: use a constant
BIRL_NORM_MODE = getstr('BIRL_NORM_MODE')  # 'inferred'
# number of samples to use
MCMC_CHAIN_LENGTH = getint('MCMC_CHAIN_LENGTH')  # 1000
# number of samples to skip between two used samples
MCMC_CHAIN_SKIP = getint('MCMC_CHAIN_SKIP')  # 10
# how many times to initialize the chain
MCMC_N_INITS = getint('MCMC_N_INITS')  # 10
# how many initial samples to skip
MCMC_BURN_IN = getint('MCMC_BURN_IN')  # 0
# start the chain from the previous maximum aposteriori sample
MCMC_REUSE_MAP = getboolean('MCMC_REUSE_MAP')  # False
# the step of the grid walk
R_STEP_SIZE = getfloat('R_STEP_SIZE')  # 0.05
# number of steps between each sample
R_STEP_COUNT = getint('R_STEP_COUNT')  # 5
# if >0, each exam entry contains a state and the learner's policy distribution from that state (used for debugging)
LONG_EXAM_SZ = getint('LONG_EXAM_SZ')  # 3
# how many exams to add per teaching iteration
EXAMS_PER_ITER = getint('EXAMS_PER_ITER')  # 1
# one exam is just one action or a trajectory
EXAM_AS_TRAJ = getboolean('EXAM_AS_TRAJ')  # False
# how many times a teacher can request any given initial state
EXAM_REUSE_COUNT = getint('EXAM_REUSE_COUNT')  # 1
# teacher maintains the full model of the learner
MCMC_USE_DEMOS = getboolean('MCMC_USE_DEMOS')  # False
# how many previous exams to consider
MCMC_EASING = getint('MCMC_EASING')  # 0
# multiplier for the previous exams
MCMC_EASING_MUL = getfloat('MCMC_EASING_MUL')  # 1
# assume that the learner uses MCE
MCMC_SOFT_Q = getboolean('MCMC_SOFT_Q')  # True
# consider the whole chain when choosing the demonstration
TEACH_WITH_CHAIN = getboolean('TEACH_WITH_CHAIN')  # False
# use soft value when computing EVD
SOFT_EVD = getboolean('SOFT_EVD')  # False


# === EXPERIMENTAL FEATURES
# coefficient for computing the MCMC probabilities
MCMC_BETA = 1
# the teacher knows the initial state of the learner
TEACHER_PREPARED = False
# if the infinite importance was achieved for several initial states, choose the random state among them
INF_IMPORTANCE_RANDOM = True
# in the object world environment, make all features separated by assigning them to different corners
OW_SEP = True


if TEACH_OMN or TEACH_SCOT or TEACH_CUR_TL:
    assert LONG_DEMOS

assert not TEACH_CUR_TL and not TEACH_BBOX, "need to add learners"

assert not TEACH_SCOT or not TEACH_SCOT2

if LEARN_VAR or LEARN_RANDOM:
    assert LEARNER_RANDOM_W

if CROSS_ENT:
    assert N_INIT_TASKS in (0, 4), 'need to implement pre-training'

if USE_OW:
    assert RHO_STATE_SAMPLING or RHO_STATE_BELLMAN
    assert ENV_LINEAR == bool(OW_MANUAL_FEATURES)
    assert ENV_LINEAR == (not OW_GRID_OBJS)
    if LEARNER_QUAD_MUL:
        assert ETA_MUL < 1
    assert ENV_LINEAR, 'need to specify env_ow.ideal_nonlin_w'
else:
    if ENV_LINEAR:
        assert N_TASKS in (7, 8)
    else:
        assert N_TASKS == 8

if TEACHER_MCE_ITERS:
    assert EXAM_AS_TRAJ  # MCE needs trajectories
    assert TEACHER_MCE_EASING
else:
    assert BIRL_NORM_MODE != 'inferred'

assert BIRL_NORM_MODE in ('inferred', 'const', 'learner')

assert MCMC_N_INITS <= MCMC_CHAIN_LENGTH

# assert MCMC_SOFT_Q, "if False, BBox is used for teaching, which always chooses T6"

if EXAMS_PER_ITER > 1 and LONG_EXAM_SZ:
    assert MCMC_EASING_MUL == 1, "not implemented"

if MCMC_USE_DEMOS:
    assert not MCMC_EASING and (MCMC_EASING_MUL == 1)
    assert MCMC_SOFT_Q and not USE_EVD_CACHE

if LEARN_VAR or not TEACHER_MCE_ITERS and (LEARN_RANDOM or TEACH_REP):
    assert (TEACH_RANDOM or TEACH_VAR) == bool(MCMC_EASING or (MCMC_EASING_MUL < 1) or MCMC_USE_DEMOS)

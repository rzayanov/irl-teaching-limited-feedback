import os
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Pool, Lock, cpu_count

from const import *
from runner import do_run_experiment
from timer import start_timer, print_timer

LOCK = Lock()


def main():
    parser = ArgumentParser()
    parser.add_argument('--experiment_count', nargs='?', type=int, const=1, default=1, required=True)
    parser.add_argument('--experiment_offset', nargs='?', type=int, const=0, default=0)
    experiment_count = parser.parse_args().experiment_count
    experiment_offset = parser.parse_args().experiment_offset

    start_timer("main")
    print('Datetime start:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('host:', os.uname()[1])
    print(f"{CONFIG_NAME=}")
    print(f"{SEED=}")

    os.makedirs('results/pickles', exist_ok=True)

    if experiment_count == 1:
        print(f"running one experiment")
        do_run_experiment(experiment_offset)
    else:
        assert not SAVE_PICKLES and not LOAD_PICKLES
        # assert experiment_count <= cpu_count()
        print(f"starting pool with {experiment_count=}")
        pool = Pool(processes=experiment_count)
        pool.map(do_run_experiment, range(experiment_offset, experiment_count + experiment_offset))
    print("All finished!")
    print('Datetime finish:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print_timer('main')


if __name__ == "__main__":
    main()

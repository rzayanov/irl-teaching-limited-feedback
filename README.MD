
### About the project

This project contains code for the following paper:

_Rustam Zayanov, Francisco S. Melo, and Manuel Lopes\
"Interactively Teaching an Inverse Reinforcement Learner with Limited Feedback"\
ArXiv e-print, [arXiv:2309.09095](https://arxiv.org/abs/2309.09095v1), (2023)_

Our source code is based on the code of the following publication:

_Gaurav Yengera, Rati Devidze, Parameswaran Kamalaruban, and Adish Singla\
"Curriculum design for teaching via demonstrations: Theory and applications"\
Advances in Neural Information Processing Systems, 34, 10496–10509, (2021)._

The original source code can be found at https://github.com/adishs/neurips2021_curriculum-teaching-demonstrations_code.

### Random seed

The seed of the first experiment is fixed and specified in `38.ini`.
The seed of the N-th experiment is calculated as `seed_1 + 100*(N - 1)`



### Environment setup

We have tested the code on the following machines:
- CPU: 4 cores, 2GHz, x86_64 architecture
- No GPU
- RAM: 2 GB
- OS: Ubuntu 20.04

We used the following packages:

```
python = "3.9.5"
numpy = "1.22.2"
scipy = "1.7.3"
matplotlib = "3.5.1"
```



### How to run the experiments

#### Generate results of 40 iterations

To execute 16 experiments on one machine, run:
`python -u main.py --experiment_count=16`

To execute the experiments on 4 machines in parallel, run:
`python -u main.py --experiment_count=4 --experiment_offset=X`
where `X` should be in `{0, 4, 8, 12}`, one per each machine. 
Merge the "results" folders in the end.

#### Generate plots

To generate plots, install Latex and run:
`python generate_plots.py`

#### Generate the results until all learners reach the 0.5 loss threshold

Edit `const.py` and set the following:
```
TEACHING_N_ITERS = 500
STOP_LOSS = .5
```
Then run 16 experiments as described previously. Then run `generate_plots.py` which will print average times in the output.

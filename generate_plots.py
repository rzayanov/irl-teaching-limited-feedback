import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from numpy.typing import NDArray
from scipy.stats import ttest_ind

from learner_idx import *

mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 25})
mpl.rc('legend', **{'fontsize': 17})
mpl.rc('text', usetex=True)

import os


def plot_reward_all(plot_path):
    reward_path = plot_path + "reward/"
    files = os.listdir(reward_path)
    reward_curve = None
    f_idx = 0
    for arr in files:
        new_curve = np.load(reward_path + arr)[:, np.newaxis, :]
        plot_reward(new_curve, plot_path + f"expected_reward_graph_{f_idx}.pdf")
        if not f_idx:
            reward_curve = new_curve
        else:
            reward_curve = np.append(reward_curve, new_curve, axis=1)
        f_idx += 1
    plot_reward(reward_curve, plot_path + "expected_reward_graph.pdf")


def plot_reward(reward_curves, path):
    mpl.rc('xtick', **{'labelsize': 18})
    mpl.rc('ytick', **{'labelsize': 18})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    spacing = 1
    plt.errorbar(np.arange(len(reward_curves[3][0]))[::spacing], np.mean(reward_curves[3], axis=0)[::spacing],
                 yerr=(np.std(reward_curves[3], axis=0) / np.sqrt(len(reward_curves)))[::spacing], color='forestgreen',
                 linewidth=3, label=r'$\textsc{Cur}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[4][0]))[::spacing], np.mean(reward_curves[4], axis=0)[::spacing],
                 yerr=(np.std(reward_curves[4], axis=0) / np.sqrt(len(reward_curves)))[::spacing], color='dodgerblue',
                 ls='-.', linewidth=2.5, label=r'$\textsc{Cur-T}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[5][0]))[::spacing], np.mean(reward_curves[5], axis=0)[::spacing],
                 yerr=(np.std(reward_curves[5], axis=0) / np.sqrt(len(reward_curves)))[::spacing], color='mediumpurple',
                 ls='--', marker='^', markersize=6, linewidth=2.5, label=r'$\textsc{Cur-L}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[2][0]))[::spacing], np.mean(reward_curves[2], axis=0)[::spacing],
                 yerr=(np.std(reward_curves[2], axis=0) / np.sqrt(len(reward_curves)))[::spacing], color='dimgray',
                 ls='-.', marker='v', markersize=6, linewidth=2.5, label=r'$\textsc{Scot}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[0][0]))[::spacing], np.mean(reward_curves[0], axis=0)[::spacing],
                 yerr=(np.std(reward_curves[0], axis=0) / np.sqrt(len(reward_curves)))[::spacing], color='orangered',
                 ls=':', linewidth=3.5, label=r'$\textsc{Agn}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[1][0]))[::spacing], np.mean(reward_curves[1], axis=0)[::spacing],
                 yerr=(np.std(reward_curves[1], axis=0) / np.sqrt(len(reward_curves)))[::spacing], color='gold',
                 ls='--', marker='s', markersize=6, linewidth=3, label=r'$\textsc{Omn}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[6][0]))[::spacing], np.mean(reward_curves[6], axis=0)[::spacing],
                 yerr=(np.std(reward_curves[6], axis=0) / np.sqrt(len(reward_curves)))[::spacing], color='firebrick',
                 ls='--', linewidth=2, label=r'$\textsc{BBox}$', elinewidth=1.0)
    plt.xlabel('Time t')
    plt.ylabel('Expected reward', labelpad=(5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.56), ncol=2, fancybox=False, shadow=False)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return


def plot_cur_all(results_path, folder, learner_idx, plot_each, iters, format):
    npy_path = results_path + folder + "/"
    full_arr = None
    for f in os.listdir(npy_path):
        f_idx = int(f.split('_')[1].split('.')[0])
        exp_arr = np.load(npy_path + f)
        if learner_idx is not None:
            learner_arr = exp_arr[learner_idx, :iters]
            file_path = results_path + f"{folder}_{f_idx}.{format}"
            plot_cur(learner_arr, 'Greys', file_path, 1)
        exp_arr_prep = exp_arr[:, np.newaxis, :, :iters]
        if plot_each:
            plot_cur_combined(exp_arr_prep, results_path, folder, f'_{f_idx}', format)
        if full_arr is None:
            full_arr = exp_arr_prep
        else:
            full_arr = np.append(full_arr, exp_arr_prep, axis=1)
    plot_cur_combined(full_arr, results_path, folder, '', format)

def plot_cur_combined(arr, results_path, folder, suf, format):
    ''' arr[learner,experiment,task,iteration] '''
    arr[arr < 0] = 0
    counts_arr = arr.sum(axis=1)
    # counts_arr[learner, task, iter]
    max_arr = counts_arr.argmax(axis=1)
    # max_arr[learner, iter]

    for learner_idx in range(len(counts_arr)):
        file_path = results_path + f"{folder}_{CODE_NAMES[learner_idx]}{suf}.{format}"
        plot_cur(counts_arr[learner_idx], PDF_CMAPS[learner_idx], file_path, arr.shape[1])

def plot_cur(arr, cmap, file_path, exp_count):
    ''' arr[task,iteration] '''
    mpl.rc('xtick', **{'labelsize': 20})
    mpl.rc('ytick', **{'labelsize': 20})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    ytick_list = []
    for i in range(8):
        ytick_list.append('T' + str(i))
    seg_count = 5  # min(5, exp_count)
    cmap = plt.get_cmap(cmap)

    new_cmap = LinearSegmentedColormap.from_list("new cmap", cmap(np.linspace(0, 1, seg_count)), N=seg_count)
    # norm = BoundaryNorm([*range(seg_count)], ncolors=seg_count, clip=True)
    # plt.pcolor(arr, cmap=new_cmap, norm=norm)

    plt.pcolor(arr, cmap=new_cmap, vmin=0, vmax=seg_count)
    plt.xlabel('Iteration')
    plt.ylabel('Task picked')
    plt.yticks(list(np.arange(8) + 0.5), ytick_list)

    if seg_count > 1:
        cbar = plt.colorbar()
        cbar_labels = [str(i) for i in range(seg_count)]
        cbar_labels[-1] += "+"
        cbar.set_ticks([i+.5 for i in range(seg_count)], labels=cbar_labels)

    plt.savefig(file_path, dpi=300, pil_kwargs={'quality': 60}, bbox_inches='tight')
    plt.close()
    return


def calc_time(results_path, folder):
    npy_path = results_path + folder + "/"
    if not os.path.isdir(npy_path):
        return
    threshs = [2, 1, .5, .2]
    print("Average time for learners to reach thresholds")
    exp_fnames = os.listdir(npy_path)
    times = None
    for exp_idx, exp_fname in enumerate(exp_fnames):
        # [learner, iter]
        exp_arr = np.load(npy_path + exp_fname)
        # [learner, exp, thresh]
        if times is None:
            times = np.zeros((len(exp_arr), len(exp_fnames), len(threshs)))
        for learner_idx, learner_losses in enumerate(exp_arr):
            for thresh_idx, thresh in enumerate(threshs):
                good_iters = np.nonzero(learner_losses < thresh)[0]
                best_iter = good_iters[0] if len(good_iters) else np.nan
                times[learner_idx, exp_idx, thresh_idx] = best_iter
    time_means = np.mean(times, axis=1).round()
    learners = get_learners(folder)
    print('rows:', ', '.join(PDF_NAMES[l] for l in learners))
    print('columns (thresholds):', ', '.join(str(t) for t in threshs))
    time_means = time_means[learners]
    print(time_means)
    print()


def plot_loss_all(results_path, folder, plot_each, iters, format):
    npy_path = results_path + folder + "/"
    if not os.path.isdir(npy_path):
        return
    full_arr = None
    f_names = os.listdir(npy_path)
    for f in f_names:
        f_idx = int(f.split('_')[1].split('.')[0])
        exp_arr = np.load(npy_path + f)[:, np.newaxis, :iters]
        # exp_arr /= exp_arr[0, 0, 0]
        if plot_each:
            plot_new(exp_arr, results_path + f"{folder}_graph_{f_idx}.{format}", folder)
        if full_arr is None:
            full_arr = exp_arr
        else:
            full_arr = np.append(full_arr, exp_arr, axis=1)
    if True:
        calc_significance(full_arr, folder)
    plot_new(full_arr, results_path + f"{folder}_graph.{format}", folder)


def plot_new(arr, file_path, folder):
    ''' arr[learner,experiment,iteration] '''
    mpl.rc('xtick', **{'labelsize': 18})
    mpl.rc('ytick', **{'labelsize': 18})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)

    learners = get_learners(folder)
    for learner_idx in learners:
        # draw_loss(arr, learner_idx, 'area')
        # draw_loss(arr, learner_idx, '+-')
        pass
    line_id = 0
    for learner_idx in learners:
        draw_loss(arr, learner_idx, 'errorbar', line_id, len(learners))
        # draw_loss(arr, learner_idx, 'line')
        line_id += 1

    plt.xticks(np.arange(0, arr.shape[2] + 1, step=5))
    plt.ylim(bottom=min(0, plt.ylim()[0]), top=max(1, plt.ylim()[1]))
    plt.xlabel('Iteration')
    plt.ylabel("Learner's loss" if folder == 'losses' else "Teacher's loss", labelpad=(5))
    # 1.26
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.56), ncol=2, fancybox=False, shadow=False)
    plt.savefig(file_path, dpi=300, pil_kwargs={'quality': 60}, bbox_inches='tight')
    plt.close()


def get_learners(folder):
    return [AGN_IDX, RND_IDX, NOE_IDX, VAR_IDX, CUR_IDX] if folder == 'losses' else [RND_IDX, NOE_IDX, VAR_IDX]


def calc_significance(arr, folder):
    # arr[learner,experiment,iteration]
    # hypothesis: loss is equal
    # reject if p < thresh
    if folder == 'teacher_losses':
        return
    print("Welch's t-test for", folder)
    thresh = .05
    learners = get_learners(folder)
    learner_count = len(learners)
    iter_count = arr.shape[2]
    res = np.full([learner_count, learner_count], np.nan, int)
    for idx1 in range(learner_count):
        for idx2 in range(learner_count):
            diff_count = 0
            for it in range(iter_count):
                sample1 = arr[learners[idx1], :, it]
                sample2 = arr[learners[idx2], :, it]
                t, p = ttest_ind(sample1, sample2, equal_var=False)
                if p < thresh:
                    diff_count += 1
            diff_perc = int(100 * diff_count / iter_count)
            res[idx1, idx2] = diff_perc
    print('columns and rows:', ', '.join(PDF_NAMES[l] for l in learners))
    print('(each cell indicates % of iterations in which the given pair of learners has a significant difference)')
    print(res)
    print()


def draw_loss(arr, learner_idx, draw_mode, line_id=0, n_learners=0):
    ''' arr[learner,experiment,iteration] '''
    spacing = 1
    learner_arr = arr[learner_idx]
    name = PDF_NAMES[learner_idx]
    color = PDF_COLORS[learner_idx]
    iterations = np.arange(len(learner_arr[0]))[::spacing]
    mean_vals = np.mean(learner_arr, axis=0)[::spacing]
    errs = (np.std(learner_arr, axis=0) / np.sqrt(arr.shape[1]))[::spacing]
    if draw_mode == 'area':
        plt.fill_between(x=iterations, y1=mean_vals - errs, y2=mean_vals + errs,
                         color=color, alpha=.2, edgecolor=None
                         )
    if draw_mode == '+-':
        for mul in [1, -1]:
            plt.plot(iterations, mean_vals + mul * errs,
                     color=color,
                     linewidth=2, alpha=.2
                     )
    if draw_mode == 'errorbar':
        err_every = line_id % n_learners, n_learners
        plt.errorbar(iterations, mean_vals, yerr=errs,
                     color=color, label=rf'$\textsc{{{name}}}$',
                     linewidth=3.5, elinewidth=1.2, errorevery=(err_every))
    if draw_mode == 'line':
        plt.plot(iterations, mean_vals,
                 color=color, label=rf'$\textsc{{{name}}}$',
                 linewidth=3.5
                 )
    plt.xticks(iterations)


def plot_simple_all(dirr, name, func, ymin, ymax):
    if not os.path.isdir(dirr):
        return
    files = os.listdir(dirr)
    f_idx = 0
    for arr in files:
        curve = np.load(dirr + arr)
        plt.figure()
        func(curve)
        plt.ylim(bottom=ymin, top=ymax)
        plt.savefig(f"{name}_{f_idx}.jpg", dpi=300, pil_kwargs={'quality': 60}, bbox_inches='tight')
        plt.close()
        f_idx += 1


def plot_simple(curve: NDArray, label=None):
    plt.plot(curve, '.', label=label)


def plot_sorted(curve: NDArray, label=None):
    curve.sort()
    plt.plot(curve, '.', label=label)


def plot_exp(curve: NDArray, label=None):
    plt.plot(np.exp(curve), '.', label=label)


def plot_exp_sorted(curve: NDArray, label=None):
    curve.sort()
    plt.plot(np.exp(curve), '.', label=label)


def plot_moving_sum(curve: NDArray, label=None):
    n = 100
    cum_sum = np.cumsum(curve, dtype=float)
    moving_sum = cum_sum[n:] - cum_sum[:-n]
    plt.plot(moving_sum, label=label)


def compare_mcmc_resampling(name, func, ymin, ymax, legend_loc):
    dirr = '../experiment-results/mcmc-compare/uniform/mcmc_resample_posteriors/'
    dirr2 = '../experiment-results/mcmc-compare/mcmc/mcmc_posteriors/'
    files = os.listdir(dirr)
    files2 = os.listdir(dirr2)
    f_idx = 0
    for arr in files:
        curve = np.load(dirr + arr)
        curve2 = np.load(dirr2 + files2[f_idx])
        plt.figure()
        func(curve, "resample")
        func(curve2, "mcmc")
        plt.ylim(bottom=ymin, top=ymax)
        plt.legend(loc=legend_loc, ncol=2, fancybox=False, shadow=False)
        plt.savefig(f"{name}_{f_idx}.jpg", dpi=300, pil_kwargs={'quality': 60}, bbox_inches='tight')
        plt.close()
        f_idx += 1


def main():
    print('Generating...')

    format = 'jpg'
    # format = 'pdf'
    iters = 40 # 20 or 50
    results_path = f"results/"
    # results_path = f"../experiment-results/38/"

    if True:
        calc_time(results_path, 'losses')

    if True:
        plot_each = False
        plot_loss_all(results_path, 'losses', plot_each, iters, format)
        # plot_loss_all(results_path, 'teacher_losses', plot_each, iters, format)
        plot_loss_all(results_path, 'teacher_soft_losses', plot_each, iters, format)

    if False:
        plot_each = False
        # plot_cur_all(results_path, 'curriculum', 7)
        plot_cur_all(results_path, 'demo_states', None, plot_each, iters, format)
        plot_cur_all(results_path, 'exam_states', None, plot_each, iters, format)

    # plot_loss_all(results_path, 'w_diffs')
    # plot_loss_all(results_path, 'p_diffs')

    if False:
        plot_simple_all("results/mcmc_switches/", "results/mcmc_switch", plot_moving_sum, 0, 100)
        plot_simple_all("results/mcmc_p_diffs/", "results/mcmc_p_diffs", plot_simple, 0, None)
        plot_simple_all("results/mcmc_w_diffs/", "results/mcmc_w_diffs", plot_simple, 0, None)
        plot_simple_all("results/mcmc_posteriors/", "results/mcmc_posteriors", plot_simple, None, 0)
        plot_simple_all("results/mcmc_posteriors/", "results/mcmc_posteriors_sorted", plot_sorted, None, 0)
        plot_simple_all("results/mcmc_resample_posteriors/", "results/mcmc_resample_posteriors_sorted", plot_sorted, None, 0)
        plot_simple_all("results/mcmc_posteriors/", "results/mcmc_exp", plot_exp, 0, 1)
        plot_simple_all("results/mcmc_posteriors/", "results/mcmc_exp_sorted", plot_exp_sorted, 0, 1)
        plot_simple_all("results/mcmc_resample_posteriors/", "results/mcmc_resample_exp_sorted", plot_exp_sorted, 0, 1)
        plot_simple_all("results/mcmc_switch_probas/", "results/mcmc_switch_probas", plot_simple, 0, 1)

    # compare_mcmc_resampling("results/mcmc_posteriors_sorted", plot_sorted, None, 0, "lower right")
    # compare_mcmc_resampling("results/mcmc_exp_sorted", plot_exp_sorted, 0, 1, "upper left")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pandas as pd
import matplotlib as mpl
from pylab import cm
import numpy as np

dataset = sys.argv[1]

if (dataset == 'nk'):
    data_path = "/playpen-ssd/athreya/set_summarization/data/nk/metrics_results"
    figure_save_name = "NK Cell"
    file_save_name = "NK"
elif (dataset == 'pree'):
    data_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia/metrics_results"
    figure_save_name = "Preeclampsia"
    file_save_name = "PreE"
else:
    data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn/metrics_results"
    figure_save_name = "HVTN"
    file_save_name = "HVTN"


mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
num_samples_per_set_range = [200, 500, 1000, 2500]


def rfe_plot():
    methods = ['Geo-Sketch', 'Kernel Herding', 'Hopper', 'IID']
    colors_dict = {'Geo-Sketch': 'tab:blue', 'Kernel Herding': 'tab:orange', 'Hopper': 'tab:green', 'IID': 'purple'}

    rfe = pd.read_csv(os.path.join(data_path, 'rfe_evaluation.csv'), index_col=0)
    rfe_melt = rfe.melt(id_vars=['subsamples', 'Sample Set'])
    rfe_melt['log_value'] = np.log(rfe_melt['value'])

    _, axes = plt.subplots(1, 1, figsize=(12, 5))
    # g = sns.lineplot(x = "subsamples", y = "log_value", hue = 'variable', data = rfe_melt, style = 'variable',
    #                 dashes=[""]*len(methods), markers=["o"]*len(methods), palette = colors_dict,
    #                 ci = 'sd', err_style = 'bars', ax = axes, err_kws={'capsize': 6})
    g = sns.lineplot(x="subsamples", y="log_value", hue='variable', data=rfe_melt, style='variable',
                     dashes=[""] * len(methods), markers=["o"] * len(methods), palette=colors_dict,
                     ci='sd', err_style='bars', ax=axes, err_kws={'capsize': 6})

    g.grid()
    g.set_xlabel("Number of Sampled Cells per set", size=16)
    g.set_ylabel("Log (Mean L1 Distance of RFE Values) \n Between Original and Sketched Sample-Sets", size=16)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 12})
    # g.set_ylim([-4.5, 2.5])
    g.xaxis.set_tick_params(labelsize=18)
    g.yaxis.set_tick_params(labelsize=18)
    # plt.show()
    plt.savefig(os.path.join(data_path, '{}_rfe_eval_new.png'.format(file_save_name)), dpi=600, transparent=False, bbox_inches='tight')


def singular_values_plot():
    methods = ['Geo-Sketch', 'Kernel Herding', 'Hopper', 'IID']
    colors_dict = {'Geo-Sketch': 'tab:blue', 'Kernel Herding': 'tab:orange', 'Hopper': 'tab:green', 'IID': 'purple'}

    sv = pd.read_csv(os.path.join(data_path, 'sv_evaluation.csv'), index_col=0)
    sv_melt = sv.melt(id_vars=['subsamples', 'Sample Set'])
    sv_melt['log_value'] = np.log(sv_melt['value'])

    _, axes = plt.subplots(1, 1, figsize=(6, 5))
    g = sns.lineplot(x="subsamples", y="log_value", hue='variable', data=sv_melt, style='variable',
                     dashes=[""] * len(methods), markers=["o"] * len(methods), palette=colors_dict,
                     ci='sd', err_style='bars', ax=axes, err_kws={'capsize': 6})

    g.grid()
    g.set_xlabel("Number of Sketched Cells", size=16)
    g.set_ylabel("Log (Mean L1 Distance of Singular Values) \n Between Original and Sketched Sample-Sets", size=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 14})
    g.set_ylim([-4, 4])
    g.xaxis.set_tick_params(labelsize=18)
    g.yaxis.set_tick_params(labelsize=18)
    # plt.show()
    plt.savefig(os.path.join(data_path, '{}_singular_values_new.png'.format(file_save_name)), dpi=600,
                transparent=False, bbox_inches='tight')


def cluster_freq_plot():
    methods = ['Geo-Sketch', 'Kernel Herding', 'Hopper', 'IID']
    colors_dict = {'Geo-Sketch': 'tab:blue', 'Kernel Herding': 'tab:orange', 'Hopper': 'tab:green', 'IID': 'purple'}

    cfreq = pd.read_csv(os.path.join(data_path, 'cluster_freq_evaluation.csv'), index_col=0)
    min_y, max_y = cfreq[['Kernel Herding', 'IID', 'Geo-Sketch', 'Hopper']].min().min(), cfreq[
        ['Kernel Herding', 'IID', 'Geo-Sketch', 'Hopper']].max().max() + 0.005
    clusters = np.unique(cfreq['clusters'])

    cfreq_melt = cfreq.melt(id_vars=['subsamples', 'clusters', 'Sample Set'])
    fig, axes = plt.subplots(1, len(clusters), figsize=(21, 5.5), gridspec_kw={'wspace': 0.175, 'bottom': 0.15})
    for i, ax in zip(range(0, len(clusters)), axes.flat):
        cfreq_subset = cfreq_melt[cfreq_melt['clusters'] == clusters[i]]
        g = sns.lineplot(x="subsamples", y="value", hue='variable', data=cfreq_subset, style='variable',
                         palette=colors_dict, dashes=[""] * len(methods), markers=["o"] * len(methods), ax=ax,
                         ci='sd', err_style='bars', err_kws={'capsize': 6})
        g.grid()
        ax.legend(loc='upper right', prop={'size': 14})
        g.xaxis.set_tick_params(labelsize=18)
        g.yaxis.set_tick_params(labelsize=18)
        g.set_ylim(0, 0.13)
        g.set_xlabel('')
        g.set_ylabel('')
        g.set_title(str(clusters[i]) + ' clusters', fontsize=18)

    title_ax = fig.add_subplot(111, frame_on=False)
    title_ax.set_xticks([])
    title_ax.set_yticks([])
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.text(0.5, 0.04, "Number of Sketched Cells", ha="center", va="center", fontsize=18)
    fig.text(0.07, 0.5, "Mean L1 Distance of Cluster Frequencies \n Between Original and Sketched Sample-Sets",
             ha="center", va="center", rotation=90, fontsize=16)

    plt.savefig(os.path.join(data_path, '{}_cluster_freq_new.png'.format(file_save_name)), dpi=600, transparent=False,
                bbox_inches='tight')

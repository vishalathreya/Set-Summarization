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
    rfe = np.load(os.path.join(data_path, "rfe_evaluation.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # min_y, max_y = rfe.min().min(), rfe.max().max()
    # plt.setp(axes, ylim=(min_y, max_y))
    axes[0].plot(num_samples_per_set_range, rfe[:, 2], label='Geo-Sketch', marker='x')
    axes[0].plot(num_samples_per_set_range, rfe[:, 0], label='Kernel Herding', marker='x')
    axes[0].plot(num_samples_per_set_range, rfe[:, 3], label='Hopper', marker='x')
    axes[0].plot(num_samples_per_set_range, rfe[:, 1], label='IID', c='purple', marker='x')
    axes[0].grid()
    axes[0].set_xlabel("Number of sampled cells per set")
    # axes[0].set_ylabel("Mean L1 distance RFE Value")
    plt.suptitle("{} - Mean L1 distance b/w true and sketched RFE values".format(figure_save_name))
    axes[0].legend(loc='upper right')

    axes[1].plot(num_samples_per_set_range, rfe[:, 1], label='IID', c='purple', marker='x')
    axes[1].plot(num_samples_per_set_range, rfe[:, 0], label='Kernel Herding', c='orange', marker='x')
    axes[1].grid()
    axes[1].set_xlabel("Number of sampled cells per set")
    axes[1].legend()
    # plt.show()
    plt.savefig('{}_rfe_eval.png'.format(file_save_name), dpi=600, transparent=False, bbox_inches='tight')

def singular_values_plot():
    sv = pd.read_csv(os.path.join(data_path, "sv_evaluation.csv"))
    # sv = np.load(os.path.join(data_path, "sv_evaluation.npy"))
    samples, kh_sv, iid_sv, geo_sv, hop_sv = sv['Samples per set'].values, sv['KH'].values, sv['IID'].values, sv['Geo'].values, sv['Hopper'].values
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # min_y, max_y = sv[['KH', 'IID', "Geo", "Hopper"]].min().min(), sv[['KH', 'IID', "Geo", "Hopper"]].max().max()
    # plt.setp(axes, ylim=(min_y, max_y))
    axes[0].plot(samples, geo_sv, label='Geo-Sketch', marker='x')
    axes[0].plot(samples, kh_sv, label='Kernel Herding', marker='x')
    axes[0].plot(samples, hop_sv, label='Hopper', marker='x')
    axes[0].plot(samples, iid_sv, label='IID', c='purple', marker='x')
    axes[0].legend()
    axes[0].grid()
    axes[0].title.set_text("All subsampling methods")
    axes[0].set_xlabel("Number of sampled cells per set")
    axes[0].set_ylabel("Mean L1 distance b/w true and sketched Singular Values")

    axes[1].plot(samples, kh_sv[:10], label='Kernel Herding', c='orange', marker='x')
    axes[1].plot(samples, iid_sv[:10], label='IID', c='purple', marker='x')
    axes[1].grid()
    axes[1].legend()
    axes[1].set_xlabel("Number of sampled cells per set")
    # axes[1].set_ylabel("Sum of mean L1 distance b/w true and sketched Singular Values")
    axes[1].title.set_text("IID and Kernel Herding close-up")
    # plt.show()

    fig.suptitle("{} Dataset Singular Value Comparison".format(figure_save_name))
    plt.savefig('{}_singular_values_all.png'.format(file_save_name), dpi=600, transparent=False, bbox_inches='tight')


def cluster_freq_plot():
    df = pd.read_csv(os.path.join(data_path, "cluster_freq_evaluation.csv"))
    min_y, max_y = df[['kh', 'iid', 'geo', 'hopper']].min().min(), df[['kh', 'iid', 'geo', 'hopper']].max().max() + 0.01
    fig, axes = plt.subplots(1, 3, figsize=(25, 5))
    plt.setp(axes, ylim=(min_y, max_y))

    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['geo'].values,
                 label='Geo-Sketch', marker='x')
    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['kh'].values, label='KH', marker='x')
    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['hopper'].values,
                 label='Hopper', marker='x')
    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['iid'].values,
                 label='IID', c='purple', marker='x')
    axes[0].title.set_text("10 clusters")
    axes[0].legend(loc='upper right')
    axes[0].grid()
    axes[0].set_xlabel("Number of sampled cells per set")
    axes[0].set_ylabel("Mean L1 distance b/w true and sketched cluster frequencies")

    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['geo'].values,
                 label='Geo-Sketch', marker='x')
    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['kh'].values, label='KH', marker='x')
    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['hopper'].values,
                 label='Hopper', marker='x')
    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['iid'].values,
                 label='IID', c='purple', marker='x')
    axes[1].title.set_text("30 clusters")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlabel("Number of sampled cells per set")

    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['geo'].values,
                 label='Geo-Sketch', marker='x')
    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['kh'].values, label='KH', marker='x')

    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['hopper'].values,
                 label='Hopper', marker='x')
    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['iid'].values,
                 label='IID', c='purple', marker='x')
    axes[2].legend()
    axes[2].grid()
    axes[2].set_xlabel("Number of sampled cells per set")
    axes[2].title.set_text("50 clusters")

    fig.suptitle("{} Dataset Subsample Cluster Frequency Comparison".format(figure_save_name))
    plt.savefig('{}_cluster_freq.png'.format(file_save_name), dpi=600, transparent=False, bbox_inches='tight')


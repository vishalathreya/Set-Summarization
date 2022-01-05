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
elif (dataset == 'preE'):
    data_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia/metrics_results"
else:
    data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn/metrics_results"


mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
num_samples_per_set_range = [200, 500, 1000, 2500]

def rfe_plot():
    rfe = np.load(os.path.join(data_path, "rfe_evaluation.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # min_y, max_y = rfe.min().min(), rfe.max().max()
    # plt.setp(axes, ylim=(min_y, max_y))
    axes[0].plot(num_samples_per_set_range, rfe[:, 2], label='Geo-Sketch')
    axes[0].plot(num_samples_per_set_range, rfe[:, 0], label='Kernel Herding')
    axes[0].plot(num_samples_per_set_range, rfe[:, 3], label='Hopper')
    axes[0].plot(num_samples_per_set_range, rfe[:, 1], label='IID', c='purple')
    axes[0].grid()
    axes[0].set_xlabel("Number of sample cells per set")
    # axes[0].set_ylabel("Mean L1 distance RFE Value")
    plt.suptitle("Mean L1 distance b/w true and sketched RFE values")
    axes[0].legend(loc='upper right')

    axes[1].plot(num_samples_per_set_range, rfe[:, 1], label='IID', c='purple')
    axes[1].plot(num_samples_per_set_range, rfe[:, 0], label='Kernel Herding', c='orange')
    axes[1].grid()
    axes[1].set_xlabel("Number of sample cells per set")
    axes[1].legend()
    # plt.show()
    plt.savefig('HVTN_rfe_eval.png', dpi=600, transparent=False, bbox_inches='tight')

def singular_values_plot():
    sv = np.load(os.path.join(data_path, "sv_evaluation.npy"))
    kh_sv, iid_sv, geo_sv, hop_sv = sv[0, :], sv[1, :], sv[2, :], sv[3, :]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    min_y, max_y = sv.min().min(), sv.max().max()
    plt.setp(axes, ylim=(min_y, max_y))
    axes[0].plot(geo_sv, label='Geo-Sketch')
    axes[0].plot(kh_sv, label='Kernel Herding')
    axes[0].plot(hop_sv, label='Hopper')
    axes[0].plot(iid_sv, label='IID')
    axes[0].legend()
    axes[0].grid()
    axes[0].title.set_text("All Singular Values")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Mean L1 distance b/w true and sketched Singular Values")

    axes[1].plot(geo_sv[:10], label='Geo-Sketch')
    axes[1].plot(kh_sv[:10], label='Kernel Herding')
    axes[1].plot(hop_sv[:10], label='Hopper')
    axes[1].plot(iid_sv[:10], label='IID')
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlabel("Singular Value Index")
    axes[1].set_ylabel("Mean L1 distance b/w true and sketched Singular Values")
    axes[1].title.set_text("First 10 SV")
    # plt.show()
    fig.suptitle("HVTN Dataset Singular Value Comparison")
    plt.savefig('HVTN_singular_values_all.png', dpi=600, transparent=False, bbox_inches='tight')

    # Plotting separate figure for close-up comparison b/w KH and IID
    plt.plot(kh_sv[:10], label='Kernel Herding', c='orange')
    plt.plot(iid_sv[:10], label='IID', c='purple')
    plt.legend()
    plt.grid()
    plt.xlabel("Singular Value Index")
    plt.ylabel("Mean L1 distance b/w true and sketched Singular Values")
    plt.title("HVTN Dataset Singular Value Comparison")
    # plt.show()
    plt.savefig('HVTN_singular_values_kh_iid_orig.png', dpi=600, transparent=False, bbox_inches='tight')


def cluster_freq_plot():
    df = pd.read_csv(os.path.join(data_path, "cluster_freq_evaluation.csv"))
    min_y, max_y = df[['kh', 'iid', 'geo', 'hopper']].min().min(), df[['kh', 'iid', 'geo', 'hopper']].max().max() + 0.01
    fig, axes = plt.subplots(1, 3, figsize=(25, 5))
    plt.setp(axes, ylim=(min_y, max_y))

    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['geo'].values,
                 label='Geo-Sketch')
    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['kh'].values, label='KH')
    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['hopper'].values,
                 label='Hopper')
    axes[0].plot(df.loc[df['clusters'] == 10]['subsamples'].values, df.loc[df['clusters'] == 10]['iid'].values,
                 label='IID')
    axes[0].title.set_text("10 clusters")
    axes[0].legend(loc='upper right')
    axes[0].grid()
    axes[0].set_xticks((200, 500, 1000, 2500))
    axes[0].set_xlabel("Number of sample cells per set")
    axes[0].set_ylabel("Mean L1 distance b/w true and sketched cluster frequencies")

    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['geo'].values,
                 label='Geo-Sketch')
    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['kh'].values, label='KH')
    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['hopper'].values,
                 label='Hopper')
    axes[1].plot(df.loc[df['clusters'] == 30]['subsamples'].values, df.loc[df['clusters'] == 30]['iid'].values,
                 label='IID')
    axes[1].title.set_text("30 clusters")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xticks((200, 500, 1000, 2500))
    axes[1].set_xlabel("Number of sample cells per set")

    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['geo'].values,
                 label='Geo-Sketch')
    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['kh'].values, label='KH')

    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['hopper'].values,
                 label='Hopper')
    axes[2].plot(df.loc[df['clusters'] == 50]['subsamples'].values, df.loc[df['clusters'] == 50]['iid'].values,
                 label='IID')
    axes[2].legend()
    axes[2].grid()
    axes[2].set_xticks((200, 500, 1000, 2500))
    axes[2].set_xlabel("Number of sample cells per set")
    axes[2].title.set_text("50 clusters")

    fig.suptitle("HVTN Dataset Subsample Cluster Frequency Comparison")
    plt.savefig('HVTN_cluster_freq.png', dpi=600, transparent=False, bbox_inches='tight')


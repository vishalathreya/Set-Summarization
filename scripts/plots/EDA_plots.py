# LOO Cluster Index TSNE

import anndata
import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


sketch1, sketch2 = 1, 2
num_samples_per_set = 500
num_clusters = 50
output_data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn/loo_data"

iid_data = anndata.read_h5ad(os.path.join(output_data_path, "iid_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch1)))
geo_data = anndata.read_h5ad(os.path.join(output_data_path, "geo_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch1)))
hop_data = anndata.read_h5ad(os.path.join(output_data_path, "hop_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch1)))
kh_1x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "1.0", sketch1)))
kh_2x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "2.0", sketch1)))
kh_0_5x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.5", sketch1)))
kh_0_2x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.2", sketch1)))


def get_subsample_train_test_data(subsample_data, train_sets, test_sets):
    subsample_train, subsample_test = subsample_data[subsample_data.obs.FCS_File.isin(train_sets)], subsample_data[subsample_data.obs.FCS_File.isin(test_sets)]
    subsample_train_X, subsample_train_Y = subsample_train.X, subsample_train.obs.label.values
    subsample_test_X, subsample_test_Y = subsample_test.X, subsample_test.obs.label.values

    return subsample_train, subsample_test, subsample_train_X, subsample_train_Y, subsample_test_X, subsample_test_Y

def get_cluster_centers(subsample_data, train_sets, test_sets, num_clusters=15):
    """
    Getting KMeans cluster centers. Trying to run it multiple times to see if they vary with each run by comparing them on TSNE plots
    """
    subsample_train, subsample_test, subsample_train_X, subsample_train_Y, subsample_test_X, subsample_test_Y = get_subsample_train_test_data(subsample_data, train_sets, test_sets)
    km = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10)
    subsample_train_preds = km.fit_predict(subsample_train_X)
    return km.cluster_centers_



fcs_files = iid_data.obs.FCS_File.values.unique()
test_set = fcs_files[10]
train_sets = list(fcs_files[:])
train_sets.remove(test_set)


# TSNE
iid_centers = get_cluster_centers(iid_data, train_sets, [test_set], num_clusters=num_clusters)
kh_1x_centers = get_cluster_centers(kh_1x_data, train_sets, [test_set], num_clusters=num_clusters)
kh_2x_centers = get_cluster_centers(kh_2x_data, train_sets, [test_set], num_clusters=num_clusters)
kh_0_5x_centers = get_cluster_centers(kh_0_5x_data, train_sets, [test_set], num_clusters=num_clusters)
kh_0_2x_centers = get_cluster_centers(kh_0_2x_data, train_sets, [test_set], num_clusters=num_clusters)
geo_centers = get_cluster_centers(geo_data, train_sets, [test_set], num_clusters=num_clusters)
hop_centers = get_cluster_centers(hop_data, train_sets, [test_set], num_clusters=num_clusters)

data_to_tsne = iid_data.X[np.random.choice(iid_data.X.shape[0], 5000)]

data_to_tsne_stacked = np.vstack((data_to_tsne, iid_centers, kh_1x_centers, geo_centers, hop_centers, kh_2x_centers, kh_0_5x_centers, kh_0_2x_centers))
tsne = TSNE(n_components=2, perplexity=40, verbose=1)
tsne = tsne.fit_transform(data_to_tsne_stacked)
x_tsne = tsne[:5000, :]
iid_tsne = tsne[5000: 5000+num_clusters, :]
kh_1x_tsne = tsne[5000+num_clusters: 5000+2*num_clusters, :]
geo_tsne = tsne[5000+2*num_clusters: 5000+3*num_clusters, :]
hop_tsne = tsne[5000+3*num_clusters:5000+4*num_clusters, :]
kh_2x_tsne = tsne[5000+4*num_clusters:5000+5*num_clusters, :]
kh_0_5x_tsne = tsne[5000+5*num_clusters:5000+6*num_clusters, :]
kh_0_2x_tsne = tsne[5000+6*num_clusters:5000+7*num_clusters, :]


test_index=10
np.save("loo_cluster_centers_tsne_50clusters_testset_index_{}.npy".format(test_index), tsne)
tsne = np.load("loo_cluster_centers_tsne_50clusters_testset_index_{}.npy".format(test_index))

sns.set(rc={'figure.figsize':(10,10)})
sns.set_style("white")

custom = [Line2D([], [], marker='+', color='0.7', linestyle='None'),
          Line2D([], [], marker='.', color='#008080', linestyle='None'),
          Line2D([], [], marker='.', color='#580F41', linestyle='None'),
          Line2D([], [], marker='.', color='orange', linestyle='None'),
          Line2D([], [], marker='.', color='#DDA0DD', linestyle='None')
          ]


sns.scatterplot(x=x_tsne[:,0], y=x_tsne[:,1], linewidth=0.9, color='0.7', marker = '+', s=150)
sns.scatterplot(x=iid_tsne[:,0], y=iid_tsne[:,1], color='#008080', marker='X', s=300)
sns.scatterplot(x=kh_1x_tsne[:,0], y=kh_1x_tsne[:,1], color='#580F41', marker='X', s=200)
sns.scatterplot(x=hop_tsne[:,0], y=hop_tsne[:,1], color='orange', marker='X', s=100)
sns.scatterplot(x=geo_tsne[:,0], y=geo_tsne[:,1], color='#DDA0DD', marker='X', s=50)

plt.legend(custom, ["Original", "IID", "KH", "Hopper", "Geo"], loc='best')
plt.title("TSNE LOO cluster centers num_clusters=50, test index={}".format(test_index))
plt.show()



# PCA
data_to_tsne2 = np.vstack((iid_centers, kh_centers, geo_centers, hop_centers))

pca = PCA(n_components=2)
pca = pca.fit_transform(data_to_tsne2)
iid_pca = pca[:num_clusters, :]
kh_pca = pca[num_clusters: num_clusters+num_clusters, :]
geo_pca = pca[num_clusters+num_clusters: num_clusters+2*num_clusters, :]
hop_pca = pca[num_clusters+2*num_clusters:, :]


sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], linewidth=0.9, color='0.7', marker = '+', s=150)
sns.scatterplot(x=iid_pca[:,0], y=iid_pca[:,1], color="orange", marker='X', s=75)
sns.scatterplot(x=hop_pca[:,0], y=hop_pca[:,1], color='#008080', marker='X', s=75)
sns.scatterplot(x=geo_pca[:,0], y=geo_pca[:,1], color='#DDA0DD', marker='X', s=75)
sns.scatterplot(x=kh_pca[:,0], y=kh_pca[:,1], color='#580F41', marker='X', s=75)
plt.legend(custom, ["IID", "Hopper", "Geo", "KH"], loc='best')

plt.show()

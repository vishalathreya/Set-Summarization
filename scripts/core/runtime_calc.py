import numpy as np
import os
import sys
import anndata
import pickle as pkl
from multiprocessing import Pool, current_process, Lock
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd
from hopper import treehopper, PCATreePartition, hopper
import time
from geosketch import gs


def get_gamma_range(X):
    from scipy.spatial import distance_matrix
    inds = np.random.choice(X.shape[0], size=1000)
    distances = distance_matrix(X[inds], X[inds])
    gamma_0 = np.median(distances)
    gammas = [gamma_0 / i for i in (4, 3, 2, 1, 0.5, 0.33, 0.25, 0.1)]

    return gammas


def random_feats(X, gamma=6, frequency_seed=None):
    scale = 1 / gamma
    if(frequency_seed is not None):
        np.random.seed(frequency_seed)
        W = np.random.normal(scale=scale, size=(X.shape[1], 1000))
    else:
        W = np.random.normal(scale=scale, size=(X.shape[1], 1000))
    XW = np.dot(X, W)
    sin_XW = np.sin(XW)
    cos_XW = np.cos(XW)
    Xnew = np.concatenate((cos_XW, sin_XW), axis=1)
    del sin_XW
    del cos_XW
    del XW
    del W
    return Xnew


def kernel_herding(X, phi, num_samples):
    w_t = np.mean(phi, axis=0)
    w_0 = w_t
    subsample = []
    indices = []
    for i in range(1, num_samples + 1):
        new_ind = np.argmax(np.dot(phi, w_t))
        x_t = X[new_ind]
        w_t = w_t + w_0 - phi[new_ind]
        indices.append(new_ind)
        subsample.append(x_t)

    return indices, subsample


def kernel_herding_main(X, phi, num_subsamples):
    kh_indices, kh_samples = kernel_herding(X, phi, num_subsamples)
    kh_rf = phi[kh_indices]
    return kh_indices, kh_samples, kh_rf


def geosketch_main(X, num_subsamples, phi=None):
    geo_indices = gs(X, num_subsamples, replace=False)
    geo_samples = X[geo_indices]

    geo_rf = None
    if(phi is not None):
        geo_rf = phi[geo_indices]

    return geo_indices, geo_samples, geo_rf


def hopper_main(X, num_subsamples, phi=None):
    if (num_subsamples >= 200):
        th = treehopper(X, partition=PCATreePartition, max_partition_size=1000)
        th.hop(num_subsamples)
        hop_indices = th.path[:num_subsamples]
    else:
        th = hopper(X)
        hop_indices = th.hop(num_subsamples)
    hop_samples = X[hop_indices]

    hop_rf = None
    if(phi is not None):
        hop_rf = phi[hop_indices]

    return hop_indices, hop_samples, hop_rf


data_path = "/home/athreya/private/set_summarization/data/"
data = anndata.read_h5ad(os.path.join(data_path, "hvtn_preprocessed.h5ad"))
# data = anndata.read_h5ad(os.path.join(data_path, "preeclampsia_preprocessed.h5ad"))
# data = anndata.read_h5ad(os.path.join(data_path, "nk_cell_preprocessed.h5ad"))

fcs_files = data.obs['FCS_File'].values.unique()
fcs_filename = fcs_files[10]
fcs_data = data[data.obs["FCS_File"] == fcs_filename]
fcs_X = fcs_data.X
label = fcs_data.obs.label.unique()[0]
label_vec = np.repeat(label, 500).reshape(-1, 1)

np.random.seed(0)
gammas = get_gamma_range(fcs_X)
gamma = gammas[3] * 1
phi = random_feats(fcs_X, gamma)

# runtimes_pre = {"iid":[], "kh":[], "hop":[], "geo":[]}
# runtimes_nk = {"iid":[], "kh":[], "hop":[], "geo":[]}
runtimes_hvtn = {"iid":[], "kh":[], "hop":[], "geo":[]}

runtimes = runtimes_hvtn

# PreE -: (210758, 33)
# NK -: (23528, 43)
# HVTN -: (179234, 11)

for i in range(5):
	start = time.time()
	iid_indices = np.random.choice(fcs_X.shape[0], 500, replace=False)
	iid_sample_index = fcs_data.obs.iloc[iid_indices].index
	iid_sample_data = fcs_data[fcs_data.obs.index.isin(iid_sample_index)]
	rt = time.time() - start
	runtimes['iid'].append(rt)

for i in range(5):
	start = time.time()
	geo_indices, geo_samples, geo_rf = geosketch_main(fcs_X, 500, phi)
	geo_sample_data = fcs_data[fcs_data.obs.iloc[geo_indices].index]
	rt = time.time() - start
	runtimes['geo'].append(rt)

for i in range(5):
	start = time.time()
	hop_indices, hop_samples, hop_rf = hopper_main(fcs_X, 500, phi)
	hop_sample_data = fcs_data[fcs_data.obs.iloc[hop_indices].index]
	rt = time.time() - start
	runtimes['hop'].append(rt)

for i in range(5):
	start = time.time()
	gammas = get_gamma_range(fcs_X)
	gamma = gammas[3] * 1
	phi = random_feats(fcs_X, gamma)
	kh_indices, kh_samples, kh_rf = kernel_herding_main(fcs_X, phi, 500)
	kh_sample_data = fcs_data[fcs_data.obs.iloc[kh_indices].index]
	rt = time.time() - start
	runtimes['kh'].append(rt)

print("kh -: {}, {}, {}".format(runtimes['kh'], np.mean(runtimes['kh']), np.std(runtimes['kh'])))
print("iid -: {}, {}, {}".format(runtimes['iid'], np.mean(runtimes['iid']), np.std(runtimes['iid'])))
print("geo -: {}, {}, {}".format(runtimes['geo'], np.mean(runtimes['geo']), np.std(runtimes['geo'])))
print("hop -: {}, {}, {}".format(runtimes['hop'], np.mean(runtimes['hop']), np.std(runtimes['hop'])))
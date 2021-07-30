import numpy as np
import os
import sys
import anndata
import pickle as pkl

from multiprocessing import Pool, current_process

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


from sklearn.manifold import TSNE

import pandas as pd

from model import *


start, end, num_processes, gamma, proc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]

def preprocess_input(input_data):
    hvtn_req_markers = ["FSC-A", "FSC-H", "CD4", "SSC-A", "ViViD", "TNFa", "IL4", "IFNg", "CD8", "CD3", "IL2"]

    # Take only required markers and GAG,ENV samples
    req_hvtn_data = input_data[:, input_data.var.pns_label.isin(hvtn_req_markers) | input_data.var.pnn_label.isin(hvtn_req_markers)]
    req_hvtn_data = req_hvtn_data[
        req_hvtn_data.obs.Sample_Treatment.str.contains("GAG") | req_hvtn_data.obs.Sample_Treatment.str.contains("ENV")]

    # Creating label column from Sample Treatment
    req_hvtn_data.obs['label'] = req_hvtn_data.obs.Sample_Treatment.apply(lambda x: 1 if "GAG" in x else 0)
    # Transform using arcsinh
    # req_hvtn_data.X = np.arcsinh((1. / 5) * req_hvtn_data.X)
    req_hvtn_data.X = StandardScaler().fit_transform(req_hvtn_data.X)

    return req_hvtn_data


def get_iid_subsamples(data, num_samples_per_set):
    # Sample num_samples_per_set points from each fcs file
    fcs_samples = data.obs.groupby('FCS_File').apply(lambda x: x.sample(min(num_samples_per_set, x.shape[0]), replace=False))
    fcs_sample_inds = [i[1] for i in fcs_samples.index]
    iid_sample_data = data[data.obs.index.isin(fcs_sample_inds)]

    return iid_sample_data


# Main
# input_data = anndata.read_h5ad(data_path + "hvtn.h5ad")
# standard_data = preprocess_input(input_data)
# standard_data.write(os.path.join(data_path, "hvtn_standard_scaler_preprocessed.h5ad"))

# IID subsampling
#data_path = "/home/athreya/private/set_summarization/data/preeclampsia"
data_path = "/home/athreya/private/set_summarization/data/"
num_samples_per_set = 500
# data = anndata.read_h5ad(os.path.join(data_path, "preeclampsia_preprocessed.h5ad"))
data = anndata.read_h5ad(os.path.join(data_path, "hvtn_preprocessed.h5ad"))
iid_sample_data = get_iid_subsamples(data, num_samples_per_set)
iid_sample_data.write(os.path.join(data_path, "iid_subsamples_{}k_per_set.h5ad".format(num_samples_per_set/1000)))


# KH subsampling
fcs_file = data.obs.FCS_File.values.unique()[start:end]
print(fcs_file)
print()
print()


def get_gamma_range(X):
    from scipy.spatial import distance_matrix
    inds = np.random.choice(X.shape[0], size=1000)
    distances = distance_matrix(X[inds], X[inds])
    gamma_0 = np.median(distances)
    gammas = [gamma_0 / i for i in (4, 3, 2, 1, 0.5, 0.33, 0.25, 0.1)]

    return gammas

def nn_gamma_range(X):
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=4, n_jobs=-1)
    inds = np.random.choice(X.shape[0], size=1000)
    neigh.fit(X[inds])
    dist, _ = neigh.kneighbors(X[inds])
    gamma_0_3nn = dist[:, 3].mean()
    gammas_3nn = [gamma_0_3nn / i for i in (4, 3, 2, 1, 0.5, 0.33, 0.25, 0.1)]

    return gammas_3nn

def parallel_kh_subsampling(fcs_filename):
    global num_samples_per_set
    fcs_data = data[data.obs.FCS_File == fcs_filename]
    fcs_X = fcs_data.X
    gammas = get_gamma_range(fcs_X)
    gamma = gammas[3]  # gamma_0 value
    print("Starting {} -> gamma_0 = {}, on process {}".format(fcs_filename, gamma, current_process().pid))
    phi = random_feats(fcs_X, gamma)
    kh_indices, kh_samples, kh_rf = kernel_herding_main(fcs_X, phi, num_samples_per_set)
    kh_sample_data = fcs_data[fcs_data.obs.iloc[kh_indices].index]

    print("Finished {} on process {}. Data size = {}".format(fcs_filename, current_process().pid, kh_sample_data.X.shape))
    kh_sample_data.write(os.path.join(data_path, "kh_samples", "kh_subsamples_{}k_per_set_{}_gamma0.h5ad".format(num_samples_per_set/1000, fcs_filename.split(".")[0])))


def parallel_geo_hopper_subsampling(fcs_filename):
    global num_samples_per_set
    print("Starting {} on process {}".format(fcs_filename, current_process().pid))
    fcs_data = data[data.obs.FCS_File == fcs_filename]
    fcs_X = fcs_data.X
    geo_indices, geo_samples, geo_rf = geosketch_main(fcs_X, num_samples_per_set)
    hop_indices, hop_samples, hop_rf = hopper_main(fcs_X, num_samples_per_set)

    geo_sample_data = fcs_data[fcs_data.obs.iloc[geo_indices].index]
    hop_sample_data = fcs_data[fcs_data.obs.iloc[hop_indices].index]

    print("Finished Geosketch on {} on process {}. Data size = {}".format(fcs_filename, current_process().pid, geo_sample_data.X.shape))
    print("Finished Hopper on {} on process {}. Data size = {}".format(fcs_filename, current_process().pid, hop_sample_data.X.shape))

    geo_sample_data.write(os.path.join(data_path, "geo_samples", "geo_subsamples_{}k_per_set_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0])))
    hop_sample_data.write(os.path.join(data_path, "hop_samples", "hop_subsamples_{}k_per_set_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0])))


#
# pool = Pool(processes=num_processes)
# if(proc == 'geo'):
#     pool.map(parallel_geo_hopper_subsampling, fcs_file)
# else:
#     pool.map(parallel_kh_subsampling, fcs_file)
# pool.close()


# KFold -> clustering -> cluster_freq vector -> classifier
data_path = "/home/athreya/private/set_summarization/data/"
# data_path = "/home/athreya/private/set_summarization/data/preeclampsia"
num_samples_per_set = 500
iid_sample_data = anndata.read_h5ad(os.path.join(data_path, "iid_subsamples_{}k_per_set_2.h5ad".format(num_samples_per_set / 1000)))
kh_sample_data = anndata.read_h5ad(os.path.join(data_path, "kh_subsamples_{}k_per_set_gamma0_2.h5ad".format(num_samples_per_set / 1000)))
geo_sample_data = anndata.read_h5ad(os.path.join(data_path, "geo_subsamples_{}k_per_set_2.h5ad".format(num_samples_per_set / 1000)))
hop_sample_data = anndata.read_h5ad(os.path.join(data_path, "hop_subsamples_{}k_per_set_2.h5ad".format(num_samples_per_set / 1000)))

iid_sample_data2 = anndata.read_h5ad(os.path.join(data_path, "iid_subsamples_{}k_per_set.h5ad".format(num_samples_per_set / 1000)))
kh_sample_data2 = anndata.read_h5ad(os.path.join(data_path, "kh_subsamples_{}k_per_set_gamma0.h5ad".format(num_samples_per_set / 1000)))
geo_sample_data2 = anndata.read_h5ad(os.path.join(data_path, "geo_subsamples_{}k_per_set.h5ad".format(num_samples_per_set / 1000)))
hop_sample_data2 = anndata.read_h5ad(os.path.join(data_path, "hop_subsamples_{}k_per_set.h5ad".format(num_samples_per_set / 1000)))


kf5 = KFold(n_splits=5, shuffle=True)
fcs_files = iid_sample_data.obs.FCS_File.values.unique()

method = 2
for method in (1,2):
    for num_clusters in (15, 30, 50):
        results = []
        print("Method = {}, # Clusters = {}".format(method, num_clusters))
        for i, (train_inds, test_inds) in enumerate(kf5.split(fcs_files)):
            # Splitting out train and test sample set fcs files
            train_sets, test_sets = fcs_files[train_inds], fcs_files[test_inds]
            ## IID
            km = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10)
            iid_train_vec, iid_train_labels, iid_test_vec, iid_test_labels = get_classification_input(iid_sample_data, iid_sample_data2, train_sets, test_sets, km, num_clusters, method, is_iid=1)
            best_params, acc, cf_matrix = train_classifier(iid_train_vec, iid_train_labels, iid_test_vec, iid_test_labels, model_type='svm')
            results.append([i+1, "iid", acc, cf_matrix])

            # KH
            kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_sample_data, kh_sample_data2, train_sets, test_sets, km, num_clusters, method, is_iid=0)
            best_params, acc, cf_matrix = train_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
            results.append([i + 1, "kh", acc, cf_matrix])

            # Geo
            geo_train_vec, geo_train_labels, geo_test_vec, geo_test_labels = get_classification_input(geo_sample_data, geo_sample_data2, train_sets, test_sets, km, num_clusters, method, is_iid=0)
            best_params, acc, cf_matrix = train_classifier(geo_train_vec, geo_train_labels, geo_test_vec, geo_test_labels, model_type='svm')
            results.append([i + 1, "geo", acc, cf_matrix])

            # Hopper
            hop_train_vec, hop_train_labels, hop_test_vec, hop_test_labels = get_classification_input(hop_sample_data, hop_sample_data2, train_sets, test_sets, km, num_clusters, method, is_iid=0)
            best_params, acc, cf_matrix = train_classifier(hop_train_vec, hop_train_labels, hop_test_vec, hop_test_labels, model_type='svm')
            results.append([i + 1, "hop", acc, cf_matrix])

        df = pd.DataFrame(results, columns=['Fold #', "method", "Accuracy", "CF_matrix"])
        df2 = df.set_index(['Fold #', 'method'])
        print(df2.groupby("method").mean())
        # print(df2)


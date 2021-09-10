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

from model import *
from train import *
import utils

start, end, num_processes, proc, scale_factor, iteration = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5]), int(sys.argv[6])

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


def parallel_one_point_representation(fcs_filename):
    global gammas
    fcs_data = data[data.obs.FCS_File == fcs_filename]
    fcs_X = fcs_data.X
    label = fcs_data.obs.label.unique()[0]
    label_vec = np.repeat(label, num_samples_per_set).reshape(-1, 1)

    for gamma in gammas:
        phi = random_feats(fcs_X, gamma, frequency_seed=0)
        phi_mean = np.mean(phi, axis=0).reshape(1, -1)
        phi_max = np.max(phi, axis=0).reshape(1, -1)
        # Append label at the end of vectors
        phi_mean = np.hstack((phi_mean, np.asarray([label]).reshape(1,-1)))
        phi_max = np.hstack((phi_max, np.asarray([label]).reshape(1, -1)))
        np.save(os.path.join(output_data_path, "one_point_rep", "{}_meanvec_gamma{}.npy".format(fcs_filename, gamma)), phi_mean)
        np.save(os.path.join(output_data_path, "one_point_rep", "{}_maxvec_gamma{}.npy".format(fcs_filename, gamma)), phi_max)

    print("Finished {}".format(fcs_filename))



def parallel_subsampling(fcs_filename):
    global num_samples_per_set
    fcs_data = data[data.obs.FCS_File == fcs_filename]
    fcs_X = fcs_data.X
    label = fcs_data.obs.label.unique()[0]
    label_vec = np.repeat(label, num_samples_per_set).reshape(-1, 1)

    gammas = get_gamma_range(fcs_X)
    gamma = gammas[3] * scale_factor  # gammas[3] = gamma_0 value
    print("Starting {} -> gamma0 = {}, scale_factor = {}, gamma = {}, on process {}".format(fcs_filename, gammas[3], scale_factor, gamma, current_process().pid))

    phi = random_feats(fcs_X, gamma)
    print("Calculated Random Features on {}".format(fcs_filename))

    # IID subsamples
    iid_indices = np.random.choice(fcs_X.shape[0], num_samples_per_set, replace=False)
    iid_sample_index = fcs_data.obs.iloc[iid_indices].index
    iid_sample_data = fcs_data[fcs_data.obs.index.isin(iid_sample_index)]
    iid_rf = phi[iid_indices]
    iid_rf = np.hstack((iid_rf, label_vec))
    iid_sample_data.write(os.path.join(output_data_path, "iid_samples", "iid_subsamples_{}k_per_set_{}_gamma{}x.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], scale_factor)))
    np.save(os.path.join(output_data_path, "iid_samples", "{}_gamma{}x_iidrf.npy".format(fcs_filename.split(".")[0], scale_factor)), iid_rf)

    # Geo
    geo_indices, geo_samples, geo_rf = geosketch_main(fcs_X, num_samples_per_set, phi)
    geo_sample_data = fcs_data[fcs_data.obs.iloc[geo_indices].index]
    geo_rf = np.hstack((geo_rf, label_vec))
    print("Finished Geosketch on {}.".format(fcs_filename))
    geo_sample_data.write(os.path.join(output_data_path, "geo_samples", "geo_subsamples_{}k_per_set_{}_gamma{}x.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], scale_factor)))
    np.save(os.path.join(output_data_path, "geo_samples", "{}_gamma{}x_georf.npy".format(fcs_filename.split(".")[0], scale_factor)), geo_rf)

    # Hopper
    hop_indices, hop_samples, hop_rf = hopper_main(fcs_X, num_samples_per_set, phi)
    hop_sample_data = fcs_data[fcs_data.obs.iloc[hop_indices].index]
    hop_rf = np.hstack((hop_rf, label_vec))
    print("Finished Hopper on {}".format(fcs_filename))
    hop_sample_data.write(os.path.join(output_data_path, "hop_samples", "hop_subsamples_{}k_per_set_{}_gamma{}x.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], scale_factor)))
    np.save(os.path.join(output_data_path, "hop_samples", "{}_gamma{}x_hoprf.npy".format(fcs_filename.split(".")[0], scale_factor)), hop_rf)

    # KH
    kh_indices, kh_samples, kh_rf = kernel_herding_main(fcs_X, phi, num_samples_per_set)
    kh_sample_data = fcs_data[fcs_data.obs.iloc[kh_indices].index]
    kh_rf = np.hstack((kh_rf, label_vec))
    print("Finished KH on {}.".format(fcs_filename))
    kh_sample_data.write(os.path.join(output_data_path, "kh_samples", "kh_subsamples_{}k_per_set_{}_gamma{}x.h5ad".format(num_samples_per_set/1000, fcs_filename.split(".")[0], scale_factor)))
    np.save(os.path.join(output_data_path, "kh_samples", "{}_gamma{}x_khrf.npy".format(fcs_filename.split(".")[0], scale_factor)), kh_rf)

    # Writing sample set name to finished_sets.txt in case program hangs in the middle and we need to re-run for remaining sets
    lock.acquire()
    print("Acquired lock for {}".format(fcs_filename))
    print("Finished {}. Writing to finished_sets.txt".format(fcs_filename))
    with open(os.path.join(output_data_path, "finished_sets.txt"), "a") as f:
        f.write(fcs_filename)
        f.write("\n")
    lock.release()
    print("Released lock for {}".format(fcs_filename))



data_path = "/home/athreya/private/set_summarization/data/"
output_data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn"
# data_path = "/home/athreya/private/set_summarization/data/preeclampsia"
num_samples_per_set = 500

if(proc == 'subsample'):
    # Main
    # input_data = anndata.read_h5ad(data_path + "hvtn.h5ad")
    # standard_data = preprocess_input(input_data)

    # data = anndata.read_h5ad(os.path.join(data_path, "preeclampsia_preprocessed.h5ad"))
    data = anndata.read_h5ad(os.path.join(data_path, "hvtn_preprocessed.h5ad"))
    print("Finished reading preprocessed data. Starting {} pools".format(num_processes))
    # KH subsampling
    fcs_file = data.obs.FCS_File.values.unique()[start:end]

    lock = Lock()

    # Overwrite finished sample sets info from previous run with Current Run params
    with open(os.path.join(output_data_path, "finished_sets.txt"), 'w') as f:
        f.write("Finished Sets for Scale Factor = {}, Iteration = {} -:".format(scale_factor, iteration))

    pool = Pool(processes=num_processes)
    pool.map(parallel_subsampling, fcs_file)
    pool.close()


if(proc == 'merge'):
    import shutil
    for method in ["iid", "kh", "hop", "geo"]:
        folder_path = os.path.join(output_data_path, "{}_samples".format(method))
        file_regex = "gamma{}x".format(scale_factor)

        # Merge original subsample anndata
        output_file_name = "{}_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(method, num_samples_per_set / 1000, scale_factor, iteration)
        utils.merge_anndata(folder_path, file_regex, output_file_name)
        shutil.move(os.path.join(folder_path, output_file_name), os.path.join(output_data_path, output_file_name))

        # Merge KHRF subsample npy files
        output_file_name = "{}_khrf_{}k_per_set_gamma{}x_{}.npy".format(method, num_samples_per_set / 1000, scale_factor, iteration)
        utils.merge_npy(folder_path, file_regex, output_file_name)
        shutil.move(os.path.join(folder_path, output_file_name), os.path.join(output_data_path, output_file_name))
        print("Merged {}".format(method))

    with open(os.path.join(output_data_path, "finished_sets.txt"), 'w') as f:
        f.write("Finished Merge for Scale Factor = {}, Iteration = {} -:".format(scale_factor, iteration))


if(proc == 'classify'):
    # KFold -> clustering -> cluster_freq vector -> classifier
    num_sketches = 3
    num_clusters = 50
    output_data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn/loo_data"
    # results_file = "classification_results_{}.csv".format(scale_factor)
    # cross_validation(output_data_path, data_path, num_sketches, num_samples_per_set, results_file, scale_factor)

    # KH Leave One Out
    results_file = "loo_classification_results_kh_{}sketches_{}clusters.csv".format(num_sketches, num_clusters)
    leave_one_out_kh_validation(output_data_path, data_path, num_sketches, num_samples_per_set, num_processes, results_file, num_clusters)

    # Other Methods Leave One Out
    # results_file = "loo_classification_results_others_{}sketches_{}clusters.csv".format(num_sketches, num_clusters)
    # leave_one_out_others_validation(output_data_path, data_path, num_sketches, num_samples_per_set, num_processes, results_file, num_clusters)


#
# gamma_0 = 2.5
# gammas = [gamma_0 / i for i in (0.125, 0.25, 0.5, 1, 2, 4, 8)]
# if (proc == 'meanvector_prep'):
#     data = anndata.read_h5ad(os.path.join(data_path, "hvtn_preprocessed.h5ad"))
#     print("Finished reading preprocessed data. Starting {} pools".format(num_processes))
#     fcs_file = data.obs.FCS_File.values.unique()[start:end]
#
#     pool = Pool(processes=num_processes)
#     pool.map(parallel_one_point_representation, fcs_file)
#     pool.close()
#
#
# # Run merge for mean and max vectors before running classify below (so that vectors from all sample sets are merged into 1 final vector)
# if(proc == 'mean_classify'):
#     for gamma in sorted(gammas):
#         mean_x, mean_y = np.load(os.path.join(output_data_path, "one_point_rep", "final_meanvec_gamma{}.npy".format(gamma)))[:, :2000], np.load(os.path.join(output_data_path, "one_point_rep", "final_meanvec_gamma{}.npy".format(gamma)))[:, 2000]
#         mean_khrf_x, mean_khrf_y = np.load(os.path.join(output_data_path, "one_point_rep", "final_meanvec_gamma0.5x_khrf.npy"))[:, :2000], np.load(os.path.join(output_data_path, "one_point_rep", "final_meanvec_gamma0.5x_khrf.npy"))[:, 2000]
#         # max_x, max_y = np.load(os.path.join(output_data_path, "one_point_rep", "final_maxvec_gamma{}.npy".format(gamma)))[:, :2000], np.load(os.path.join(output_data_path, "one_point_rep",  "final_maxvec_gamma{}.npy".format(gamma)))[:, 2000]
#         print("Running for gamma = {}".format(gamma))
#         kf5 = KFold(n_splits=5, shuffle=True)
#         results = []
#         for i, (train_inds, test_inds) in enumerate(kf5.split(mean_x)):
#             # Splitting out train and test sample set fcs files
#             # Testing on Global mean vectors
#             print("Training and Testing on Global")
#             train_vec, train_labels, test_vec, test_labels = mean_x[train_inds], mean_y[train_inds], mean_x[test_inds], mean_y[test_inds]
#             model, acc, cf_matrix = train_classifier(train_vec, train_labels, test_vec, test_labels, model_type='svm')
#             results.append([i+1, "trainglobal_testglobal", gamma, acc])
#
#             # Testing on KH mean vectors
#             print("Training on Global and Testing on KH")
#             preds = model.predict(mean_khrf_x)
#             acc = metrics.accuracy_score(mean_khrf_y, preds)
#             results.append([i + 1, "trainglobal_testKH", gamma, acc])
#
#             # Train on KH, test on KH
#             print("Training and Testing on KH")
#             train_vec, train_labels, test_vec, test_labels = mean_khrf_x[train_inds], mean_khrf_y[train_inds], mean_khrf_x[test_inds], mean_khrf_y[test_inds]
#             model, acc, cf_matrix = train_classifier(train_vec, train_labels, test_vec, test_labels, model_type='svm')
#             results.append([i + 1, "trainKH_testKH", gamma, acc])
#
#
#         df = pd.DataFrame(results, columns=['Fold #', "Rep", "Gamma", "Acc"])
#         # df2 = df.set_index(['Fold #', 'subsampling'])
#         # df_final = df2.groupby("subsampling").mean().reset_index()
#
#
#         classification_results_file = os.path.join(data_path, "meanvector_classification_KH_results.csv")
#         if(os.path.isfile(classification_results_file)):
#             df.to_csv(classification_results_file, mode='a', header=None, index=False)
#         else:
#             df.to_csv(classification_results_file, index=False)
#

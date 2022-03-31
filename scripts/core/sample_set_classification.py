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

import argparse
from model import *
from train import *
import utils

# Compulsory -: proc, num_samples_per_set, input_path, output_path, sample_key
# Optional -: start, end (if subsampling), num_processes, scale_factor, iteration
# start, end, num_processes, proc, scale_factor, iteration, num_samples_per_set = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), \
#                                                                                 sys.argv[4], float(sys.argv[5]), int(sys.argv[6]), \
#                                                                                 int(sys.argv[7])

def setup_data_folders(output_path):
    print("Setting up folders in {}".format(output_path))
    os.makedirs(os.path.join(output_path, "orig_samples"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "iid_samples"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "kh_samples"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "hop_samples"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "geo_samples"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "merged_samples_data"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "metrics_results"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "classification_results"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)

def preprocess_input(input_data):
    hvtn_req_markers = ["FSC-A", "FSC-H", "CD4", "SSC-A", "ViViD", "TNFa", "IL4", "IFNg", "CD8", "CD3", "IL2"]

    # Take only required markers and GAG,ENV samples
    req_hvtn_data = input_data[:, input_data.var.pns_label.isin(hvtn_req_markers) | input_data.var.pnn_label.isin(hvtn_req_markers)]
    req_hvtn_data = req_hvtn_data[
        req_hvtn_data.obs.Sample_Treatment.str.contains("GAG") | req_hvtn_data.obs.Sample_Treatment.str.contains("ENV")]

    # Creating label column from Sample Treatment
    req_hvtn_data.obs['label'] = req_hvtn_data.obs.Sample_Treatment.apply(lambda x: 1 if "GAG" in x else 0)
    # Transform using arcsinh
    req_hvtn_data.X = np.arcsinh((1. / 5) * req_hvtn_data.X)
    # req_hvtn_data.X = StandardScaler().fit_transform(req_hvtn_data.X)

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

def print_args(args_dict):
    print("Argument values passed in current run -:")
    for k, v in sorted(args_dict.items()):
        print(k, "=", v)


def parallel_subsampling(fcs_filename):
    # global args.num_samples_per_set, args.iteration
    print("Inside parallel subsampling num_samples_per_set = {}, iteration = {}".format(args.num_samples_per_set, args.iteration))
    fcs_data = data[data.obs[args.sample_key] == fcs_filename]
    fcs_X = fcs_data.X
    label = fcs_data.obs.label.unique()[0]
    label_vec = np.repeat(label, args.num_samples_per_set).reshape(-1, 1)

    gammas = get_gamma_range(fcs_X)
    gamma = gammas[3] * args.scale_factor  # gammas[3] = gamma_0 value
    print("Starting {} -> gamma0 = {}, scale_factor = {}, gamma = {}, on process {}".format(fcs_filename, gammas[3], args.scale_factor, gamma, current_process().pid))

    phi = random_feats(fcs_X, gamma)
    # Save random features of original data
    # h5ad original samples
    fcs_data.write(os.path.join(args.output_path, "orig_samples", "orig_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(args.num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration)))
    # Random fourier features of samples
    np.save(os.path.join(args.output_path, "orig_samples", "{}_{}k_per_set_gamma{}x_phi_{}.npy".format(fcs_filename.split(".")[0], args.num_samples_per_set / 1000, args.scale_factor, args.iteration)), phi)
    print("Calculated Random Features on {}".format(fcs_filename))

    # IID subsamples
    iid_indices = np.random.choice(fcs_X.shape[0], args.num_samples_per_set, replace=False)
    iid_sample_index = fcs_data.obs.iloc[iid_indices].index
    iid_sample_data = fcs_data[fcs_data.obs.index.isin(iid_sample_index)]
    iid_rf = phi[iid_indices]
    iid_rf = np.hstack((iid_rf, label_vec))
    iid_sample_data.write(os.path.join(args.output_path, "iid_samples", "iid_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(args.num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration)))
    np.save(os.path.join(args.output_path, "iid_samples", "{}_{}k_per_set_gamma{}x_iidrf_{}.npy".format(fcs_filename.split(".")[0], args.num_samples_per_set / 1000, args.scale_factor, args.iteration)), iid_rf)

    # Geo
    geo_indices, geo_samples, geo_rf = geosketch_main(fcs_X, args.num_samples_per_set, phi)
    geo_sample_data = fcs_data[fcs_data.obs.iloc[geo_indices].index]
    geo_rf = np.hstack((geo_rf, label_vec))
    print("Finished Geosketch on {}.".format(fcs_filename))
    geo_sample_data.write(os.path.join(args.output_path, "geo_samples", "geo_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(args.num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration)))
    np.save(os.path.join(args.output_path, "geo_samples", "{}_{}k_per_set_gamma{}x_georf_{}.npy".format(fcs_filename.split(".")[0], args.num_samples_per_set / 1000, args.scale_factor, args.iteration)), geo_rf)

    # Hopper
    hop_indices, hop_samples, hop_rf = hopper_main(fcs_X, args.num_samples_per_set, phi)
    hop_sample_data = fcs_data[fcs_data.obs.iloc[hop_indices].index]
    hop_rf = np.hstack((hop_rf, label_vec))
    print("Finished Hopper on {}".format(fcs_filename))
    hop_sample_data.write(os.path.join(args.output_path, "hop_samples", "hop_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(args.num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration)))
    np.save(os.path.join(args.output_path, "hop_samples", "{}_{}k_per_set_gamma{}x_hoprf_{}.npy".format(fcs_filename.split(".")[0], args.num_samples_per_set / 1000, args.scale_factor, args.iteration)), hop_rf)

    # KH
    kh_indices, kh_samples, kh_rf = kernel_herding_main(fcs_X, phi, args.num_samples_per_set)
    kh_sample_data = fcs_data[fcs_data.obs.iloc[kh_indices].index]
    kh_rf = np.hstack((kh_rf, label_vec))
    print("Finished KH on {}.".format(fcs_filename))
    kh_sample_data.write(os.path.join(args.output_path, "kh_samples", "kh_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(args.num_samples_per_set/1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration)))
    np.save(os.path.join(args.output_path, "kh_samples", "{}_{}k_per_set_gamma{}x_khrf_{}.npy".format(fcs_filename.split(".")[0], args.num_samples_per_set / 1000, args.scale_factor, args.iteration)), kh_rf)

    # Writing sample set name to finished_sets.txt in case program hangs in the middle and we need to re-run for remaining sets
    lock.acquire()
    print("Acquired lock for {}".format(fcs_filename))
    print("Finished {}. Writing to finished_sets.txt".format(fcs_filename))
    with open(os.path.join(args.output_path, "finished_sets.txt"), "a") as f:
        f.write(fcs_filename)
        f.write("\n")
    lock.release()
    print("Released lock for {}".format(fcs_filename))


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    # Required values
    parser.add_argument('proc', choices=['subsample', 'merge', 'classify'], help='process name')
    parser.add_argument('--input_path', metavar='i', help='path pointing to input anndata file', required=True)
    parser.add_argument('--output_path', metavar='o', help='folder path for output data', required=True)
    parser.add_argument('--sample_key', help='anndata key containing name of sample set', required=True)
    # Optional values with defaults
    parser.add_argument('--start', type=int,
                        help='start index of list of sample sets used in subsample function', default=0)
    parser.add_argument('--end', type=int,
                        help='end index of list of sample sets used in subsample function', default=100)
    parser.add_argument('--num_processes', metavar='P', type=int,
                        help='number of processes used in parallel subsampling', default=10)
    parser.add_argument('--scale_factor', type=float,
                        help='bandwidth scaling for Fourier features used in Kernel Herding', default=1.0)
    parser.add_argument('--iteration', metavar='iter', type=int,
                        help='Run iteration', default=1)
    parser.add_argument('--num_samples_per_set', metavar='n', type=int,
                        help='number of sketched samples per sample set', default=500)
    args = parser.parse_args()

    args_dict = vars(args)
    print_args(args_dict)

    if(args.proc == 'subsample'):
        # Preprocess HVTN data
        # input_data = anndata.read_h5ad(data_path + "hvtn.h5ad")
        # standard_data = preprocess_input(input_data)
        # standard_data.write(os.path.join(data_path, "hvtn_preprocessed.h5ad"))

        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn"
        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia"
        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/nk"

        # data_path = "/home/athreya/private/set_summarization/data/"
        # data = anndata.read_h5ad(os.path.join(data_path, "hvtn_preprocessed.h5ad"))
        # data = anndata.read_h5ad(os.path.join(data_path, "preeclampsia_preprocessed.h5ad"))
        # data = anndata.read_h5ad(os.path.join(data_path, "nk_cell_preprocessed.h5ad"))

        # Setup output folders
        setup_data_folders(args.output_path)

        data = anndata.read_h5ad(args.input_path)
        print("Finished reading preprocessed data. Starting {} pools to subsample from sets start={} to end={} in the list of sample sets".format(args.num_processes, args.start, args.end))

        # KH subsampling
        fcs_files = data.obs[args.sample_key].values.unique()[args.start:args.end]
        fcs_files = data.obs[args.sample_key].values.unique()[args.start:args.end]
        # fcs_file = data.obs[args.sample_key].values.unique()[[33, 24, 32, 25, 30, 7]]         # Run for specific sample sets in case proc failed in between

        lock = Lock()

        # Overwrite finished sample sets info from previous run with Current Run params
        with open(os.path.join(args.output_path, "finished_sets.txt"), 'w') as f:
            f.write("Finished Sets for Scale Factor = {}, Iteration = {} -:\n".format(args.scale_factor, args.iteration))

        pool = Pool(processes=args.num_processes)
        pool.map(parallel_subsampling, fcs_files)
        pool.close()


    if(args.proc == 'merge'):
        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn"
        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia"
        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/nk"

        # input_path -: parent folder which contains kh_samples, iid_samples, hop_samples, etc folders
        # output_path -: {parent_folder}/merged_samples_data/
        import shutil
        for method in ["iid", "kh", "hop", "geo"]:
            folder_path = os.path.join(args.input_path, "{}_samples".format(method))
            file_regex = "gamma{}x_{}".format(args.scale_factor, args.iteration)

            # Merge subsample anndata
            output_file_name = "{}_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(method, args.num_samples_per_set / 1000, args.scale_factor, args.iteration)
            print("Merging into {}".format(output_file_name))
            utils.merge_anndata(folder_path, file_regex, output_file_name)
            shutil.move(os.path.join(folder_path, output_file_name), os.path.join(args.output_path, output_file_name))

            # Merge subsample npy files
            # output_file_name = "{}_khrf_{}k_per_set_gamma{}x_{}.npy".format(method, num_samples_per_set / 1000, scale_factor, iteration)
            # utils.merge_npy(folder_path, file_regex, output_file_name)
            # shutil.move(os.path.join(folder_path, output_file_name), os.path.join(output_data_path, output_file_name))
            # print("Merged {}".format(method))

        print("Finished Merge for Scale Factor = {}, Iteration = {} -:".format(args.scale_factor, args.iteration))


    if(args.proc == 'classify'):
        # Number of unique sketches available for each method (Pairs of sketches are taken at a time,
        # 1 is used for training and the other is used for test)
        num_sketches = 3

        # data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn"
        # data_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia"
        # data_path = "/playpen-ssd/athreya/set_summarization/data/nk"
        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia/loo_data"
        # output_data_path = "/playpen-ssd/athreya/set_summarization/data/nk/loo_data"


        # 5-Fold Cross Validation
        results_file = "5fold_cv_classification_results_{}subsamples_{}.csv".format(args.num_samples_per_set / 1000, args.scale_factor)
        cross_validation(args.input_path, args.output_path, num_sketches, args.num_samples_per_set, results_file, args.scale_factor)

        # KH Leave One Out
        cluster_counts = [15, 30, 50]

        # input_path -: {parent_folder}/merged_samples_data
        # output_path -: {parent_folder} or wherever the results csv should be placed
        for num_clusters in cluster_counts:
            # output_data_path = "/playpen-ssd/athreya/set_summarization/data/hvtn/loo_data"
            # output_data_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia/loo_data"
            # output_data_path = "/playpen-ssd/athreya/set_summarization/data/nk/loo_data"

            results_file = "loo_classification_results_kh_{}subsamples_{}sketches_{}clusters.csv".format(args.num_samples_per_set / 1000, num_sketches, num_clusters)
            # leave_one_out_kh_validation(args.input_path, args.output_path, num_sketches, args.num_samples_per_set, args.num_processes, results_file, num_clusters)

            # Other Methods Leave One Out
            results_file = "loo_classification_results_others_{}subsamples_{}sketches_{}clusters.csv".format(args.num_samples_per_set / 1000, num_sketches, num_clusters)
            leave_one_out_others_validation(args.input_path, args.output_path, num_sketches, args.num_samples_per_set, args.num_processes, results_file, num_clusters)



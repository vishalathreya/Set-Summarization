# Ported code from Colab notebook for computing subsampling metrics (cluster freq, singular values, RFE)

import os
import argparse
import anndata
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scripts.core.sample_set_classification import print_args


def rfe_metric_calculation(vec, beta):
  return np.mean(np.dot(vec, beta), axis=0)

def rfe_eval(fcs_filename):
    avg_l1 = []
    for num_samples_per_set in num_samples_per_set_range:
        print("RFE Eval for num_subsamples = {}".format(num_samples_per_set))
        phi = np.load(os.path.join(args.output_path, "orig_samples", "{}_{}k_per_set_gamma{}x_phi_{}.npy".format(fcs_filename.split(".")[0], num_samples_per_set / 1000, args.scale_factor, args.iteration)))
        kh_rf = np.load(os.path.join(args.output_path, "kh_samples", "{}_{}k_per_set_gamma{}x_khrf_{}.npy".format(fcs_filename.split(".")[0], num_samples_per_set / 1000, args.scale_factor, args.iteration)))
        iid_rf = np.load(os.path.join(args.output_path, "iid_samples", "{}_{}k_per_set_gamma{}x_iidrf_{}.npy".format(fcs_filename.split(".")[0], num_samples_per_set / 1000, args.scale_factor, args.iteration)))
        geo_rf = np.load(os.path.join(args.output_path, "geo_samples", "{}_{}k_per_set_gamma{}x_georf_{}.npy".format(fcs_filename.split(".")[0], num_samples_per_set / 1000, args.scale_factor, args.iteration)))
        hop_rf = np.load(os.path.join(args.output_path, "hop_samples", "{}_{}k_per_set_gamma{}x_hoprf_{}.npy".format(fcs_filename.split(".")[0], num_samples_per_set / 1000, args.scale_factor, args.iteration)))

        # Take first 2k feature length, ignoring last column which is the label
        iid_rf = iid_rf[:, :2000]
        geo_rf = geo_rf[:, :2000]
        hop_rf = hop_rf[:, :2000]
        kh_rf = kh_rf[:, :2000]

        np.random.seed(25)
        eval = []
        for i in range(5):
            beta = np.random.normal(size=2000).reshape(-1, 1)

            orig_eval = rfe_metric_calculation(phi, beta)[0]
            kh_eval = rfe_metric_calculation(kh_rf, beta)[0]
            iid_eval = rfe_metric_calculation(iid_rf, beta)[0]
            geo_eval = rfe_metric_calculation(geo_rf, beta)[0]
            hop_eval = rfe_metric_calculation(hop_rf, beta)[0]
            eval.append([orig_eval, kh_eval, iid_eval, geo_eval, hop_eval])

        subset_methods = np.asarray(eval)[:, 1:]        # Columns = methods, rows = trials
        original_data = np.asarray(eval)[:, 0].reshape(-1, 1)
        # Average L1 distance across 5 trials
        avg_l1.append(np.mean(np.abs(subset_methods - original_data), axis=0))      # Columns = methods, rows = num_samples_per_set

    avg_l1 = np.asarray(avg_l1)
    kh_l1, iid_l1, geo_l1, hop_l1 = avg_l1[:, 0], avg_l1[:, 1], avg_l1[:, 2], avg_l1[:, 3]

    return kh_l1, iid_l1, geo_l1, hop_l1

def rfe_helper():
    # Variables to accumulate L1 distance across sample sets
    kh_final_l1, iid_final_l1, geo_final_l1, hop_final_l1 = np.zeros((1, len(num_samples_per_set_range))), np.zeros((1, len(num_samples_per_set_range))), \
                                                            np.zeros((1, len(num_samples_per_set_range))), np.zeros((1, len(num_samples_per_set_range)))

    for fcs_filename in fcs_files:
        kh_l1, iid_l1, geo_l1, hop_l1 = rfe_eval(fcs_filename)
        kh_final_l1 += kh_l1
        iid_final_l1 += iid_l1
        geo_final_l1 += geo_l1
        hop_final_l1 += hop_l1
        print("Finished calculating RFE for {}".format(fcs_filename))

    # Average across sample sets
    kh_final_l1 /= len(fcs_files)
    iid_final_l1 /= len(fcs_files)
    geo_final_l1 /= len(fcs_files)
    hop_final_l1 /= len(fcs_files)

    l1_results = np.vstack((kh_final_l1, iid_final_l1, geo_final_l1, hop_final_l1)).T       # COLUMNS are methods
    np.save(os.path.join(args.output_path, "metrics_results", "rfe_evaluation.npy"), l1_results)



def singular_values_eval(fcs_filename, num_samples_per_set):
    print("Singular Value Eval for num_subsamples = {}".format(num_samples_per_set))
    orig_samples = anndata.read_h5ad(os.path.join(args.output_path, "orig_samples", "orig_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
    kh_samples = anndata.read_h5ad(os.path.join(args.output_path, "kh_samples", "kh_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
    iid_samples = anndata.read_h5ad(os.path.join(args.output_path, "iid_samples", "iid_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
    geo_samples = anndata.read_h5ad(os.path.join(args.output_path, "geo_samples", "geo_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
    hop_samples = anndata.read_h5ad(os.path.join(args.output_path, "hop_samples", "hop_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X

    orig_sv = np.sqrt(1.0 / orig_samples.shape[0]) * np.linalg.svd(orig_samples, compute_uv=False)
    kh_sv = np.sqrt(1.0 / np.asarray(kh_samples).shape[0]) * np.linalg.svd(kh_samples, compute_uv=False)
    iid_sv = np.sqrt(1.0 / iid_samples.shape[0]) * np.linalg.svd(iid_samples, compute_uv=False)
    geo_sv = np.sqrt(1.0 / geo_samples.shape[0]) * np.linalg.svd(geo_samples, compute_uv=False)
    hop_sv = np.sqrt(1.0 / hop_samples.shape[0]) * np.linalg.svd(hop_samples, compute_uv=False)

    # If number of features > number of samples (e.g sketch data shape = (500, 900)) while original data shape = (6000, 900)),
    # len(orig_sv) == 900 != len(kh_sv) == 500. So we take the first min(len(orig_sv), len(kh_sv)) singular values in reverse sorted order
    # which is what np.linalg.svd returns

    if(len(orig_sv) > len(kh_sv)):
        orig_sv = orig_sv[:len(kh_sv)]

    # Calculate L1 distance b/w true and sketched set
    kh_sv_l1 = np.linalg.norm(kh_sv - orig_sv, ord=1)
    iid_sv_l1 = np.linalg.norm(iid_sv - orig_sv, ord=1)
    geo_sv_l1 = np.linalg.norm(geo_sv - orig_sv, ord=1)
    hop_sv_l1 = np.linalg.norm(hop_sv - orig_sv, ord=1)

    return kh_sv_l1, iid_sv_l1, geo_sv_l1, hop_sv_l1

def singular_values_helper():
    sv_results = []
    # Variables to accumulate singular values across sample sets
    for num_samples_per_set in num_samples_per_set_range:
        print("Running singular values for {} samples per set".format(num_samples_per_set))

        kh_final_sv, iid_final_sv, geo_final_sv, hop_final_sv = 0, 0, 0, 0
        for fcs_filename in fcs_files:
            kh_sv_l1, iid_sv_l1, geo_sv_l1, hop_sv_l1 = singular_values_eval(fcs_filename, num_samples_per_set)
            kh_final_sv += kh_sv_l1
            iid_final_sv += iid_sv_l1
            geo_final_sv += geo_sv_l1
            hop_final_sv += hop_sv_l1
            print("Finished calculating Singular Values for {}".format(fcs_filename))

        # Average across sample sets
        kh_final_sv /= len(fcs_files)
        iid_final_sv /= len(fcs_files)
        geo_final_sv /= len(fcs_files)
        hop_final_sv /= len(fcs_files)

        # Save singular values for each num_samples_per_set
        sv_results.append([num_samples_per_set, kh_final_sv, iid_final_sv, geo_final_sv, hop_final_sv])       # Columns are methods
    final = pd.DataFrame(sv_results, columns=['Samples per set', 'KH', 'IID', 'Geo', 'Hopper'])
    final.to_csv(os.path.join(args.output_path, "metrics_results", "sv_evaluation.csv"))



def cluster_freq_eval(fcs_filename):
    cluster_freq_metric = []
    for num_samples_per_set in num_samples_per_set_range:
        orig_samples = anndata.read_h5ad(os.path.join(args.output_path, "orig_samples", "orig_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
        kh_samples = anndata.read_h5ad(os.path.join(args.output_path, "kh_samples", "kh_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
        iid_samples = anndata.read_h5ad(os.path.join(args.output_path, "iid_samples", "iid_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
        geo_samples = anndata.read_h5ad(os.path.join(args.output_path, "geo_samples", "geo_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X
        hop_samples = anndata.read_h5ad(os.path.join(args.output_path, "hop_samples", "hop_subsamples_{}k_per_set_{}_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0], args.scale_factor, args.iteration))).X

        cluster_range = [10, 30, 50]
        for num_clusters in cluster_range:
            km = KMeans(init="k-means++", n_clusters=num_clusters, n_init=4, random_state=0)
            orig_preds = km.fit_predict(orig_samples)
            kh_preds = km.predict(kh_samples)
            iid_preds = km.predict(iid_samples)
            geo_preds = km.predict(geo_samples)
            hop_preds = km.predict(hop_samples)

            cluster_rows = []
            # Cluster frequency heatmap for each subsample method
            for i in range(num_clusters):
                cluster_rows.append([num_clusters, (orig_preds == i).sum() / orig_preds.shape[0],
                                     (kh_preds == i).sum() / kh_preds.shape[0],
                                     (iid_preds == i).sum() / iid_preds.shape[0],
                                     (geo_preds == i).sum() / geo_preds.shape[0],
                                     (hop_preds == i).sum() / hop_preds.shape[0]])

            dfc = pd.DataFrame(cluster_rows, columns=["# clusters", "Original", "KH", "IID", "Geo", "Hopper"])

            kh_avg_l1 = np.mean(np.abs((dfc['KH'] - dfc['Original']).values))
            iid_avg_l1 = np.mean(np.abs((dfc['IID'] - dfc['Original']).values))
            geo_avg_l1 = np.mean(np.abs((dfc['Geo'] - dfc['Original']).values))
            hop_avg_l1 = np.mean(np.abs((dfc['Hopper'] - dfc['Original']).values))
            cluster_freq_metric.append([fcs_filename, num_samples_per_set, num_clusters, kh_avg_l1, iid_avg_l1, geo_avg_l1, hop_avg_l1])

    return pd.DataFrame(cluster_freq_metric, columns=['Sample Set', 'subsamples', 'clusters', 'kh', 'iid', 'geo', 'hopper'])

def cluster_freq_helper():
    cluster_results = pd.DataFrame(columns=['Sample Set', 'subsamples', 'clusters', 'kh', 'iid', 'geo', 'hopper'])
    for i, fcs_filename in enumerate(fcs_files):

        sample_cluster_results = cluster_freq_eval(fcs_filename)
        cluster_results = pd.concat([cluster_results, sample_cluster_results])
        if((i % 10 == 0) and i > 0):        # Write to disk after every 10 sample sets
            # save to disk after
            print("Finished {} sample sets. Saving current results to disk".format(i+1))
            copy = cluster_results.copy()
            cluster_freq_avg_l1 = copy.groupby(by=['subsamples', 'clusters']).mean()
            cluster_freq_avg_l1.reset_index(drop=False, inplace=True)
            cluster_freq_avg_l1.to_csv(os.path.join(args.output_path, "metrics_results", "cluster_freq_evaluation.csv"))
        print("Finished calculating Cluster Frequency for {}".format(fcs_filename))

    cluster_freq_avg_l1 = cluster_results.groupby(by=['subsamples', 'clusters']).mean()
    cluster_freq_avg_l1.reset_index(drop=False, inplace=True)

    cluster_freq_avg_l1.to_csv(os.path.join(args.output_path, "metrics_results", "cluster_freq_evaluation.csv"))


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    # Required values
    parser.add_argument('--input_path', metavar='i', help='path pointing to input anndata file', required=True)
    parser.add_argument('--output_path', metavar='o', help='folder path for output data', required=True)
    parser.add_argument('--sample_key', help='anndata key containing name of sample set', required=True)
    # Optional values with defaults
    parser.add_argument('--scale_factor', type=float,
                        help='bandwidth scaling for Fourier features used in Kernel Herding', default=1.0)
    parser.add_argument('--iteration', metavar='iter', type=int,
                        help='Run iteration', default=1)
    args = parser.parse_args()
    print_args(vars(args))

    # input_path -: path to the source .h5ad file in order to fetch the list of sample sets
        # Example
        # input_path = "/home/athreya/private/set_summarization/data/"
        # data = anndata.read_h5ad(os.path.join(input_path, "preeclampsia_preprocessed.h5ad"))
    # output_path -: path to read the sketch data to calculate metrics as well as to write results file
        # Example
        # output_path = "/playpen-ssd/athreya/set_summarization/data/preeclampsia/"

    data = anndata.read_h5ad(args.input_path)
    num_samples_per_set_range = [200, 500, 1000, 2500]      # Set to list of values for which sketch data exists
    fcs_files = data.obs[args.sample_key].values.unique()

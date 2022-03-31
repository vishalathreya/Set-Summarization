import time

import numpy as np
import os
import anndata

from multiprocessing import Pool, current_process, Lock

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import pandas as pd
from functools import partial

from model import *


def cross_validation(input_path, output_path, num_sketches, num_samples_per_set, results_file, scale_factor=None):
    for sketch1, sketch2 in [(i, j) for i in range(1, num_sketches+1) for j in range(1, num_sketches+1) if(i != j)]:
    # for sketch1, sketch2 in [(1, 2),]:
        print("Reading data for sketch1 = {}, sketch2 = {}".format(sketch1, sketch2))
        kh_sample_data = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch1)))
        iid_sample_data = anndata.read_h5ad(os.path.join(input_path, "iid_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch1)))
        geo_sample_data = anndata.read_h5ad(os.path.join(input_path, "geo_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch1)))
        hop_sample_data = anndata.read_h5ad(os.path.join(input_path, "hop_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch1)))

        kh_sample_data2 = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch2)))
        iid_sample_data2 = anndata.read_h5ad(os.path.join(input_path, "iid_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch2)))
        geo_sample_data2 = anndata.read_h5ad(os.path.join(input_path, "geo_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch2)))
        hop_sample_data2 = anndata.read_h5ad(os.path.join(input_path, "hop_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, sketch2)))

        for num_trials in range(20):
            print("Starting trial {}".format(num_trials +1))
            kf5 = KFold(n_splits=5, shuffle=True)
            fcs_files = iid_sample_data.obs.FCS_File.values.unique()

            final_results = pd.DataFrame()
            # for method in (1, 2):
            for method in (2,):
                for num_clusters in (15, 30, 50):
                    results = []
                    print("Method = {}, # Clusters = {}".format(method, num_clusters))
                    for i, (train_inds, test_inds) in enumerate(kf5.split(fcs_files)):
                        # Splitting out train and test sample set fcs files
                        train_sets, test_sets = fcs_files[train_inds], fcs_files[test_inds]
                        ## IID
                        km = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10)
                        iid_train_vec, iid_train_labels, iid_test_vec, iid_test_labels = get_classification_input(iid_sample_data, iid_sample_data2, train_sets, test_sets, num_clusters, km, method, is_iid=1)
                        model, acc, cf_matrix = train_classifier(iid_train_vec, iid_train_labels, iid_test_vec, iid_test_labels, model_type='svm')
                        results.append([i + 1, "iid", acc])

                        # KH
                        kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_sample_data, kh_sample_data2, train_sets, test_sets, num_clusters, km, method, is_iid=0)
                        model, acc, cf_matrix = train_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
                        results.append([i + 1, "kh", acc])

                        # Geo
                        geo_train_vec, geo_train_labels, geo_test_vec, geo_test_labels = get_classification_input(geo_sample_data, geo_sample_data2, train_sets, test_sets, num_clusters, km, method, is_iid=0)
                        model, acc, cf_matrix = train_classifier(geo_train_vec, geo_train_labels, geo_test_vec, geo_test_labels, model_type='svm')
                        results.append([i + 1, "geo", acc])

                        # Hopper
                        hop_train_vec, hop_train_labels, hop_test_vec, hop_test_labels = get_classification_input(hop_sample_data, hop_sample_data2, train_sets, test_sets, num_clusters, km, method, is_iid=0)
                        model, acc, cf_matrix = train_classifier(hop_train_vec, hop_train_labels, hop_test_vec, hop_test_labels, model_type='svm')
                        results.append([i + 1, "hop", acc])

                    df = pd.DataFrame(results, columns=['Fold #', "subsampling", "Acc"])
                    df2 = df.set_index(['Fold #', 'subsampling'])
                    df_final = df2.groupby("subsampling").mean().reset_index()
                    df_final['clusters'] = num_clusters
                    df_final['method'] = method
                    final_results = pd.concat((final_results, df_final))

            print("Finished for trial {}. Writing to file".format(num_trials+1))
            classification_results_file = os.path.join(output_path, results_file)
            if(os.path.isfile(classification_results_file)):
                final_results.to_csv(classification_results_file, mode='a', header=None, index=False)
            else:
                final_results.to_csv(classification_results_file, index=False)


def parallel_leaveoneout_kh_classification(test_set, num_clusters, kh_data):
    """
    Performs Leave-One-Out Classification and returns the ypred for test_set (For Kernel Herding subsamples)
    :param test_set:
    :param num_clusters:
    :param kh_data:
    :return:
    """
    print("{} process -: {}".format(current_process().pid, test_set))

    kh_1x_data, kh_1x_data2, kh_2x_data, kh_2x_data2, kh_0_5x_data, kh_0_5x_data2, kh_0_2x_data, kh_0_2x_data2 = kh_data
    method = 2
    fcs_files = list(kh_1x_data.obs.FCS_File.values.unique())

    # Splitting out train and test sample set fcs files
    train_sets = fcs_files[:]
    train_sets.remove(test_set)
    # KH
    iteration_results = []
    start = time.time()
    kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_1x_data, kh_1x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    t1 = time.time()
    model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
    iteration_results.append((best_estimator_score, ypred, 1))
    print("Time for get_classification_input = {}, Time for Classifier = {}".format(t1-start, time.time()-t1))
    kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_2x_data, kh_2x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
    iteration_results.append((best_estimator_score, ypred, 2))

    kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_0_5x_data, kh_0_5x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
    iteration_results.append((best_estimator_score, ypred, 3))

    kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_0_2x_data, kh_0_2x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
    iteration_results.append((best_estimator_score, ypred, 4))

    # Prediction using Model with Best gamma and SVM hyperparams
    best_model_ypred = max(iteration_results, key= lambda x: x[0])

    # correct_pred_count = (np.asarray(total_ypred) == np.asarray(total_ytest)).sum()
    print("Finished {} in {} secs. Pred = {}, True = {}".format(test_set, time.time()-start, best_model_ypred[1], kh_test_labels))

    return best_model_ypred[1] == kh_test_labels, best_model_ypred[2], best_model_ypred[1]   # Prediction == GT, Gamma indicator to calculate pick rate, ypred



def leave_one_out_kh_validation(input_path, output_path, num_sketches, num_samples_per_set, num_processes, results_file, num_clusters):
    print("Starting LOO validation for Kernel Herding sketches using {} processes and {} KMeans clusters".format(num_processes, num_clusters))
    pool = Pool(processes=num_processes)
    for sketch1, sketch2 in [(i, j) for i in range(1, num_sketches+1) for j in range(1, num_sketches+1) if(i != j)]:
    # for sketch1, sketch2 in [(1,2),]:
        print("Reading data for sketch1 = {}, sketch2 = {}".format(sketch1, sketch2))

        # Load sketches 1&2 for each gamma value
        kh_1x_data = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "1.0", sketch1)))
        kh_1x_data2 = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "1.0", sketch2)))
        kh_2x_data = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "2.0", sketch1)))
        kh_2x_data2 = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "2.0", sketch2)))
        kh_0_5x_data = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.5", sketch1)))
        kh_0_5x_data2 = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.5", sketch2)))
        kh_0_2x_data = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.2", sketch1)))
        kh_0_2x_data2 = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.2", sketch2)))

        # Re-run N times
        for num_trials in range(5):
            start = time.time()
            print("Starting trial {}".format(num_trials + 1))
            fcs_files = list(kh_1x_data.obs.FCS_File.values.unique())

            results = []

            pool_results = pool.map(partial(parallel_leaveoneout_kh_classification, num_clusters=num_clusters, kh_data=(kh_1x_data, kh_1x_data2, kh_2x_data, kh_2x_data2, kh_0_5x_data, kh_0_5x_data2, kh_0_2x_data, kh_0_2x_data2)), fcs_files, chunksize=len(fcs_files)//num_processes)
            print(pool_results)
            pool_results = [(i[0][0], i[1], i[2][0]) for i in pool_results]
            pool_results = np.asarray(pool_results)
            # Aggregate the results from each of the workers
            acc = np.sum(pool_results[:, 0]) / len(fcs_files)                                           # Classification accuracy
            gamma_pick_rate = np.bincount(pool_results[:, 1], minlength=5)[1:] / len(fcs_files)         # Gamma Pick Rate
            ypred_positive_count = np.sum(pool_results[:, 2])                                           # Positive ypred Count (ypred == 1)
            print(gamma_pick_rate)

            results.append([sketch1, sketch2, num_trials + 1, "kh_hyperparam", acc, gamma_pick_rate, ypred_positive_count])
            final_results = pd.DataFrame(results, columns=["Sketch1", "Sketch2", "Trial #", "Method", "Acc", "Gamma Pick Rate", "ypred +ve Count"])
            print("Finished for trial {}. Writing to file".format(num_trials + 1))
            classification_results_file = os.path.join(output_path, results_file)
            if(os.path.isfile(classification_results_file)):
                final_results.to_csv(classification_results_file, mode='a', header=None, index=False)
            else:
                final_results.to_csv(classification_results_file, index=False)

            print("Time taken for trial {} = {} secs".format(num_trials + 1, time.time()-start))

    pool.close()
    pool.join()
    pool.terminate()


def parallel_leaveoneout_others_classification(test_set, num_clusters, data):
    """
    Performs Leave-One-Out Classification and returns the ypred for test_set (For Geo, IID & Hopper subsamples)
    :param test_set:
    :param num_clusters:
    :param kh_data:
    :return:
    """
    print("{} process -: {}".format(current_process().pid, test_set))

    iid_data, iid_data2, geo_data, geo_data2, hop_data, hop_data2, kh_data, kh_data2 = data
    method = 2
    fcs_files = list(iid_data.obs.FCS_File.values.unique())

    # Splitting out train and test sample set fcs files
    train_sets = fcs_files[:]
    train_sets.remove(test_set)
    # print("Length of train_sets = {}".format(len(train_sets)))
    # KH
    iteration_results = []
    start = time.time()

    # IID
    iid_train_vec, iid_train_labels, iid_test_vec, iid_test_labels = get_classification_input(iid_data, iid_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    t1 = time.time()
    model, best_estimator_score, ypred = leave_one_out_classifier(iid_train_vec, iid_train_labels, iid_test_vec, iid_test_labels, model_type='svm')
    iteration_results.append(ypred == iid_test_labels)
    print("Time for get_classification_input = {}, Time for Classifier = {}".format(t1-start, time.time()-t1))

    # Geo
    geo_train_vec, geo_train_labels, geo_test_vec, geo_test_labels = get_classification_input(geo_data, geo_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    model, best_estimator_score, ypred = leave_one_out_classifier(geo_train_vec, geo_train_labels, geo_test_vec, geo_test_labels, model_type='svm')
    iteration_results.append(ypred == geo_test_labels)

    # Hopper
    hop_train_vec, hop_train_labels, hop_test_vec, hop_test_labels = get_classification_input(hop_data, hop_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    model, best_estimator_score, ypred = leave_one_out_classifier(hop_train_vec, hop_train_labels, hop_test_vec, hop_test_labels, model_type='svm')
    iteration_results.append(ypred == hop_test_labels)

    # KH with same gamma (1x) instead of picking between diff gammas
    kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_data, kh_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
    model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
    iteration_results.append(ypred == kh_test_labels)

    print("Finished {} in {} secs.".format(test_set, time.time()-start))

    return iteration_results



def leave_one_out_others_validation(input_path, output_path, num_sketches, num_samples_per_set, num_processes, results_file, num_clusters):
    print("Starting LOO validation for IID, Geo, Hopper and KH (only 1 gamma) sketches using {} processes and {} KMeans clusters".format(num_processes, num_clusters))
    pool = Pool(processes=num_processes)
    for sketch1, sketch2 in [(i, j) for i in range(1, num_sketches+1) for j in range(1, num_sketches+1) if(i != j)]:
    # for sketch1, sketch2 in [(1,2), (2,1)]:
        print("Reading data for sketch1 = {}, sketch2 = {}".format(sketch1, sketch2))

        # Load sketches 1&2 for each of the methods
        iid_data = anndata.read_h5ad(os.path.join(input_path, "iid_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch1)))
        geo_data = anndata.read_h5ad(os.path.join(input_path, "geo_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch1)))
        hop_data = anndata.read_h5ad(os.path.join(input_path, "hop_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch1)))
        kh_data = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch1)))

        iid_data2 = anndata.read_h5ad(os.path.join(input_path, "iid_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch2)))
        geo_data2 = anndata.read_h5ad(os.path.join(input_path, "geo_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch2)))
        hop_data2 = anndata.read_h5ad(os.path.join(input_path, "hop_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch2)))
        kh_data2 = anndata.read_h5ad(os.path.join(input_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, 1.0, sketch2)))


        # Re-run N times
        for num_trials in range(5):
            start = time.time()
            print("Starting trial {}".format(num_trials + 1))
            fcs_files = list(iid_data.obs.FCS_File.values.unique())

            results = []

            pool_results = pool.map(partial(parallel_leaveoneout_others_classification, num_clusters=num_clusters, data=(iid_data, iid_data2, geo_data, geo_data2, hop_data, hop_data2, kh_data, kh_data2)), fcs_files, chunksize=len(fcs_files)//num_processes)
            # Aggregate the results from each of the workers
            pool_results = np.asarray(pool_results)
            print("pool_results.shape = {}, len(fcs_files) = {}, pool_results = {}".format(pool_results.shape, len(fcs_files), pool_results))
            acc = np.sum(pool_results, axis=0) / len(fcs_files)

            results.append([sketch1, sketch2, num_trials + 1, "iid", acc[0][0]])
            results.append([sketch1, sketch2, num_trials + 1, "geo", acc[1][0]])
            results.append([sketch1, sketch2, num_trials + 1, "hop", acc[2][0]])
            results.append([sketch1, sketch2, num_trials + 1, "kh", acc[3][0]])
            final_results = pd.DataFrame(results, columns=["Sketch1", "Sketch2", "Trial #", "Method", "Acc"])
            print("Finished for trial {}. Writing to file".format(num_trials + 1))
            classification_results_file = os.path.join(output_path, results_file)
            if(os.path.isfile(classification_results_file)):
                final_results.to_csv(classification_results_file, mode='a', header=None, index=False)
            else:
                final_results.to_csv(classification_results_file, index=False)

            print("Time taken for trial {} = {} secs".format(num_trials, time.time()-start))

    pool.close()
    pool.join()
    pool.terminate()


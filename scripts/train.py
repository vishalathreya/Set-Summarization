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

from model import *


def cross_validation(output_data_path, data_path, num_sketches, num_samples_per_set, results_file, scale_factor=None):
    for iteration1, iteration2 in [(i ,j) for i in range(1 ,num_sketches+1) for j in range(1 ,num_sketches+1) if( i!=j)]:
        print("Reading data for iteration1 = {}, iteration2 = {}".format(iteration1, iteration2))
        kh_sample_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration1)))
        iid_sample_data = anndata.read_h5ad(os.path.join(output_data_path, "iid_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration1)))
        geo_sample_data = anndata.read_h5ad(os.path.join(output_data_path, "geo_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration1)))
        hop_sample_data = anndata.read_h5ad(os.path.join(output_data_path, "hop_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration1)))

        kh_sample_data2 = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration2)))
        iid_sample_data2 = anndata.read_h5ad(os.path.join(output_data_path, "iid_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration2)))
        geo_sample_data2 = anndata.read_h5ad(os.path.join(output_data_path, "geo_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration2)))
        hop_sample_data2 = anndata.read_h5ad(os.path.join(output_data_path, "hop_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, scale_factor, iteration2)))

        for num_trials in range(30):
            print("Starting trial {}".format(num_trials +1))
            kf5 = KFold(n_splits=5, shuffle=True)
            fcs_files = iid_sample_data.obs.FCS_File.values.unique()

            final_results = pd.DataFrame()
            for method in (1, 2):
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
                        results.append([ i +1, "iid", acc])

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
            classification_results_file = os.path.join(data_path, results_file)
            if(os.path.isfile(classification_results_file)):
                final_results.to_csv(classification_results_file, mode='a', header=None, index=False)
            else:
                final_results.to_csv(classification_results_file, index=False)



def leave_one_out_validation(output_data_path, data_path, num_sketches, num_samples_per_set, results_file, num_clusters):
    for iteration1, iteration2 in [(i ,j) for i in range(1 ,num_sketches+1) for j in range(1 ,num_sketches+1) if( i!=j)]:
        print("Reading data for iteration1 = {}, iteration2 = {}".format(iteration1, iteration2))

        kh_1x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "1.0", iteration1)))
        kh_1x_data2 = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "1.0", iteration2)))
        kh_2x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "2.0", iteration1)))
        kh_2x_data2 = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "2.0", iteration2)))
        kh_0_5x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.5", iteration1)))
        kh_0_5x_data2 = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.5", iteration2)))
        kh_0_2x_data = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.2", iteration1)))
        kh_0_2x_data2 = anndata.read_h5ad(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_gamma{}x_{}.h5ad".format(num_samples_per_set / 1000, "0.2", iteration2)))

        for num_trials in range(10):
            print("Starting trial {}".format(num_trials + 1))
            fcs_files = list(kh_1x_data.obs.FCS_File.values.unique())
            total_ypred = []
            total_ytest = []
            method = 2

            results = []
            print("Method = {}, # Clusters = {}".format(method, num_clusters))

            for ind, test_set in enumerate(fcs_files):
                # Splitting out train and test sample set fcs files
                train_sets = fcs_files[:]
                train_sets.remove(test_set)
                # KH
                iteration_results = []
                kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_1x_data, kh_1x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
                model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
                iteration_results.append((best_estimator_score, ypred))

                kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_2x_data, kh_2x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
                model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
                iteration_results.append((best_estimator_score, ypred))

                kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_0_5x_data, kh_0_5x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
                model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
                iteration_results.append((best_estimator_score, ypred))

                kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels = get_classification_input(kh_0_2x_data, kh_0_2x_data2, train_sets, [test_set], num_clusters, method=method, is_iid=0)
                model, best_estimator_score, ypred = leave_one_out_classifier(kh_train_vec, kh_train_labels, kh_test_vec, kh_test_labels, model_type='svm')
                iteration_results.append((best_estimator_score, ypred))

                # Prediction using Model with Best gamma and SVM hyperparams
                best_model_ypred = max(iteration_results, key= lambda x: x[0])
                total_ypred.append(best_model_ypred[1])
                total_ytest.append(kh_test_labels)
                print("Finished for test set = {}".format(ind))

            acc = (np.asarray(total_ypred) == np.asarray(total_ytest)).sum() / len(fcs_files)
            results.append([iteration1, iteration2, num_trials+1, "kh", acc])
            final_results = pd.DataFrame(results, columns=["Sketch1", "Sketch2", "Trial #", "Method", "Acc"])
            print("Finished for trial {}. Writing to file".format(num_trials+1))
            classification_results_file = os.path.join(data_path, results_file)
            if(os.path.isfile(classification_results_file)):
                final_results.to_csv(classification_results_file, mode='a', header=None, index=False)
            else:
                final_results.to_csv(classification_results_file, index=False)


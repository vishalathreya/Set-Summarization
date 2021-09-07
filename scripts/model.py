import numpy as np
from sklearn.cluster import KMeans

from hopper import treehopper, PCATreePartition, hopper
from geosketch import gs

import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier



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



def get_subsample_train_test_data(subsample_data, train_sets, test_sets):
    subsample_train, subsample_test = subsample_data[subsample_data.obs.FCS_File.isin(train_sets)], subsample_data[subsample_data.obs.FCS_File.isin(test_sets)]
    subsample_train_X, subsample_train_Y = subsample_train.X, subsample_train.obs.label.values
    subsample_test_X, subsample_test_Y = subsample_test.X, subsample_test.obs.label.values

    return subsample_train, subsample_test, subsample_train_X, subsample_train_Y, subsample_test_X, subsample_test_Y



def get_cluster_freq_vector(km, subsample_data, subsample_preds, num_clusters):
    """
    For each set (train/test) in each method (IID, KH, Geo, Hopper) this clusters points using a fitted KMeans
    and represents each sample set as a vector of cluster frequencies (sample set's points in cluster / total points in cluster)

    Usage -:  get_cluster_freq_vector(km, iid_sample_train, iid_sample_train_preds, num_clusters=30)
             get_cluster_freq_vector(km, kh_sample_test, kh_sample_test_preds, num_clusters=30)

    :param km: trained KMeans model
    :param subsample_data: the AnnData of the subset method data
    :param subsample_preds: predicted cluster IDs using km of each point in subsample_data
    :param num_clusters: number of clusters in km
    :return: (num_sample_set X num_clusters) cluster freq vec, (num_sample_set, ) sample set labels, sample_sets used
    """
    cluster_sizes = dict([(i, (subsample_preds == i).sum()) for i in range(num_clusters)])
    try:
        # For NK Cell data
        fcs_filename_key = "fcs_filename"
        sample_sets = subsample_data.obs.fcs_filename.unique()
    except AttributeError:
        # For HVTN data
        fcs_filename_key = "FCS_File"
        sample_sets = subsample_data.obs[fcs_filename_key].unique()
    vec = []
    sample_labels = []
    for sample in sample_sets:
        sample_x = subsample_data[subsample_data.obs[fcs_filename_key] == sample, :]
        sample_label = subsample_data[subsample_data.obs[fcs_filename_key] == sample].obs.label.unique()[0]
        sample_preds = km.predict(sample_x.X)       # Predicting again to get cluster ID of this sample set's points (same as filtering from subsample_preds)
        sample_freq = []
        for i in range(num_clusters):
            if(cluster_sizes[i] != 0):
                sample_freq.append((sample_preds == i).sum() / cluster_sizes[i])
            else:
                # print("Cluster size = 0 for cluster {}, size = {}, (sample_preds==i).sum() = {}".format(i, cluster_sizes[i], (sample_preds == i).sum()))
                sample_freq.append(0)

        # Append sample set label
        sample_labels.append(sample_label)
        vec.append(sample_freq)
    return np.asarray(vec), np.asarray(sample_labels), sample_sets


def get_classification_input(subsample_data, subsample_data2, train_sets, test_sets, num_clusters, km=None, method=1, is_iid=0):
    """
    Represents each sample set as a cluster frequency vector using KMeans
    :param subsample_data: Train Kmeans on this
    :param subsample_data2: Use Kmeans centers to predict on this
    :param train_sets:
    :param test_sets:
    :param num_clusters:
    :param km:
    :param method:
    :param is_iid:
    :return:
    """

    # subsample_train, subsample_test, subsample_train_X, subsample_train_Y, subsample_test_X, subsample_test_Y = get_subsample_train_test_data(subsample_data, train_sets, test_sets)
    _, _, subsample_train_cluster_X, _, _, _ = get_subsample_train_test_data(subsample_data, train_sets, test_sets)
    subsample_train, subsample_test, subsample_train_X, subsample_train_Y, subsample_test_X, subsample_test_Y = get_subsample_train_test_data(subsample_data2, train_sets, test_sets)

    if(method == 2):
        # Fit the KMeans on one sketch
        km = KMeans(init="k-means++", n_clusters=num_clusters, n_init=5)
        # subsample_train_preds = km.fit_predict(subsample_train_X)
        subsample_train_preds = km.fit_predict(subsample_train_cluster_X)
    else:
        # Fit cluster on IID train data
        if(is_iid):
            # subsample_train_preds = km.fit_predict(subsample_train_X)
            subsample_train_preds = km.fit_predict(subsample_train_cluster_X)
        else:
            subsample_train_preds = km.predict(subsample_train_X)

    # Use trained KMeans centers to predict on the other sketch
    subsample_train_preds = km.predict(subsample_train_X)
    subsample_test_preds = km.predict(subsample_test_X)

    # Compute the cluster frequency vector
    subsample_train_vec, subsample_train_labels, subsample_train_sample_sets = get_cluster_freq_vector(km, subsample_train, subsample_train_preds, num_clusters=km.cluster_centers_.shape[0])
    subsample_test_vec, subsample_test_labels, subsample_test_sample_sets = get_cluster_freq_vector(km, subsample_test, subsample_test_preds, num_clusters=km.cluster_centers_.shape[0])

    return subsample_train_vec, subsample_train_labels, subsample_test_vec, subsample_test_labels


def get_cluster_centers(subsample_data, train_sets, test_sets, num_clusters=15, method=2):
    """
    Getting KMeans cluster centers. Trying to run it multiple times to see if they vary with each run by comparing them on TSNE plots
    """
    subsample_train, subsample_test, subsample_train_X, subsample_train_Y, subsample_test_X, subsample_test_Y = get_subsample_train_test_data(subsample_data, train_sets, test_sets)
    if (method == 2):
        km = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10)
        subsample_train_preds = km.fit_predict(subsample_train_X)
        return km.cluster_centers_
    else:
        print("Method != 2. Cannot run. Exiting...")


def train_classifier(xtrain, ytrain, xtest, ytest, model_type='svm'):
    if(model_type == 'svm'):
        param_grid = {'C': [0.1, 1, 100, 1000], 'kernel': ['rbf', 'poly'],
                  'degree': [3, 4, 6], 'gamma': [1, 0.1, 0.001, 0.0001]}
        model = SVC()
    elif(model_type == 'svm_linear'):
        param_grid = {'C': [0.1, 1, 100, 1000], 'kernel': ['linear'], 'tol': [1e-5]}
        model = SVC()
    else:
        param_grid = {'n_estimators': [200, 500, 1000, 5000]}
        model = RandomForestClassifier()

    grid = GridSearchCV(model, param_grid, n_jobs=-1)
    grid.fit(xtrain, ytrain)
    preds = grid.predict(xtest)
    acc = metrics.accuracy_score(ytest, preds)
    cf_matrix = metrics.confusion_matrix(ytest, preds)

    return grid, acc, cf_matrix


def leave_one_out_classifier(xtrain, ytrain, xtest, ytest, model_type='svm'):
    if(model_type == 'svm'):
        param_grid = {'C': [0.1, 1, 100, 1000], 'kernel': ['rbf', 'poly'],
                  'degree': [3, 4, 6], 'gamma': [1, 0.1, 0.001, 0.0001]}
        model = SVC()
    elif(model_type == 'svm_linear'):
        param_grid = {'C': [0.1, 1, 100, 1000], 'kernel': ['linear'], 'tol': [1e-5]}
        model = SVC()
    else:
        param_grid = {'n_estimators': [200, 500, 1000, 5000]}
        model = RandomForestClassifier()

    # Using only 1 split for grid search CV
    train_inds, test_inds = train_test_split(range(xtrain.shape[0]), test_size=0.2)
    grid = GridSearchCV(model, param_grid, cv=[(train_inds, test_inds)])
    grid.fit(xtrain, ytrain)
    # The score using the best hyperparam combination. Used to compare with the scores coming from data using other gamma values
    best_estimator_score = grid.best_score_
    ypred = grid.predict(xtest)

    return grid, best_estimator_score, ypred
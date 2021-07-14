import numpy as np
import anndata
import pickle as pkl

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


input_data_path = "/home/athreya/private/set_summarization/data/"
input_data = anndata.read_h5ad("private/set_summarization/hvtn_downsampled.h5ad")
X = input_data.X

def random_feats(X, gamma=6):
  scale = 1/gamma
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
  for i in range(1, num_samples+1):
    new_ind = np.argmax(np.dot(phi, w_t))
    x_t = X[new_ind]
    w_t = w_t + w_0 - phi[new_ind]
    indices.append(new_ind)
    subsample.append(x_t)

    if(i%50 == 0):
      print("Done subsample {}".format(i))

  return indices, subsample


def kernel_main(X, phi, num_subsamples):
  kh_indices, kh_samples = kernel_herding(X, phi, num_subsamples)
  kh_rf = phi[kh_indices]
  return kh_indices, kh_samples, kh_rf


def sample_set_cluster_freq_vector(km, downsampled_data, orig_preds, num_clusters):
  cluster_sizes = dict([(i, (orig_preds==i).sum()) for i in range(num_clusters)])
  print(cluster_sizes)
  try:
    fcs_filename_key = "fcs_filename"
    sample_sets = downsampled_data.obs.fcs_filename.unique()
  except AttributeError:
    fcs_filename_key = "FCS_File"
    sample_sets = downsampled_data.obs[fcs_filename_key].unique()
  vec = []
  sample_labels = []
  for sample in sample_sets:
    sample_x = downsampled_data[downsampled_data.obs[fcs_filename_key] == sample, :]
    sample_label = downsampled_data[downsampled_data.obs[fcs_filename_key]==sample].obs.label.unique()[0]
    sample_preds = km.predict(sample_x.X)
    sample_freq = []
    for i in range(num_clusters):
      sample_freq.append((sample_preds == i).sum()/cluster_sizes[i])
    
    # Append sample set label
    sample_labels.append(sample_label)
    # print("Sample = {}, label = {}".format(sample, sample_label))
    print(sample_freq)
    vec.append(sample_freq)

  return np.asarray(vec), np.asarray(sample_labels), sample_sets


def subsample_method_data(downsampled_adata, method_inds):
  method_adata = downsampled_adata[downsampled_adata.obs.iloc[method_inds].index]
  method_X = method_adata.X
  method_preds = km.predict(method_X)
  method_labels = method_adata.obs['label'].values

  return method_adata, method_X, method_preds, method_labels


def train_classifier(vec, sample_labels):
  kf5 = KFold(n_splits=5, shuffle=True, random_state=100)
  for train_inds, test_inds in kf5.split(vec):
    xtrain, ytrain = vec[train_inds], sample_labels[train_inds]
    xtest, ytest = vec[test_inds], sample_labels[test_inds]
    param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    grid = GridSearchCV(SVC(),param_grid)
    # grid = RandomForestClassifier()
    grid.fit(xtrain,ytrain)
    preds = grid.predict(xtest)
    # print(grid.best_params_)
    print(grid.score(xtest,ytest))
  # print(metrics.accuracy_score(ytest, preds))
    print(metrics.confusion_matrix(ytest, preds))


gamma = 7.5
phi = random_feats(X, gamma)

num_subsamples=10000
kh_indices, kh_samples, kh_rf = kernel_main(X, phi, num_subsamples)


with open("private/set_summarization/kmeans_30_rs1120.pkl", "rb") as f:
  km = pkl.load(f)

num_clusters = 10
kh_adata, kh_X, kh_preds, kh_labels = subsample_method_data(input_data, kh_indices)
kh_vec, kh_sample_labels, kh_sample_sets = sample_set_cluster_freq_vector(km, kh_adata, kh_preds, num_clusters=num_clusters)
train_classifier(kh_vec, kh_sample_labels)




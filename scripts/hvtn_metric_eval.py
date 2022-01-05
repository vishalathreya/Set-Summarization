import numpy as np
import os
import sys
import anndata

from multiprocessing import Pool, current_process
import pandas as pd

from model import *
import matplotlib.pyplot as plt

start, end, num_processes, proc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

data_path = "/home/athreya/private/set_summarization/data/"
output_data_path = "/playpen-ssd/athreya/set_summarization/data/"
num_samples_per_set = 500
# data = anndata.read_h5ad(os.path.join(data_path, "preeclampsia_preprocessed.h5ad"))
data = anndata.read_h5ad(os.path.join(data_path, "hvtn_preprocessed.h5ad"))


# KH subsampling
fcs_file = data.obs.FCS_File.values.unique()[start:end]
print(fcs_file)
print()
print()


def parallel_kh_subsampling(fcs_filename):
    global num_samples_per_set
    print("Base data ID = {}".format(id(data)))
    print("Starting {} on process {}".format(fcs_filename, current_process().pid))
    fcs_data = data[data.obs.FCS_File == fcs_filename]
    fcs_X = fcs_data.X
    iid_indices = np.random.choice(fcs_X.shape[0], num_samples_per_set, replace=False)
    iid_sample_index = fcs_data.obs.iloc[iid_indices].index
    iid_sample_data = fcs_data[fcs_data.obs.index.isin(iid_sample_index)]
    iid_sample_data.write(os.path.join(output_data_path, "iid_subsamples_{}k_per_set_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0])))

    gammas = get_gamma_range(fcs_X)
    for gamma in gammas:
        phi = random_feats(fcs_X, gamma)
        kh_indices, kh_samples, kh_rf = kernel_herding_main(fcs_X, phi, num_samples_per_set)
        kh_sample_data = fcs_data[fcs_data.obs.iloc[kh_indices].index]

        iid_rf = phi[iid_indices]
        np.save(os.path.join(output_data_path, "{}_gamma{}_iidrf.npy".format(fcs_filename.split(".")[0], gamma)), iid_rf)
        np.save(os.path.join(output_data_path, "{}_gamma{}_phi.npy".format(fcs_filename.split(".")[0], gamma)), phi)
        np.save(os.path.join(output_data_path, "{}_gamma{}_khrf.npy".format(fcs_filename.split(".")[0], gamma)), kh_rf)
        print("Finished {} on process {}. Data size = {}".format(fcs_filename, current_process().pid, kh_sample_data.X.shape))
        kh_sample_data.write(os.path.join(output_data_path, "kh_subsamples_{}k_per_set_{}_gamma{}.h5ad".format(num_samples_per_set/1000, fcs_filename.split(".")[0], gamma)))



def parallel_geo_hopper_subsampling(fcs_filename):
    global num_samples_per_set
    print("Base data ID = {}".format(id(data)))
    print("Starting {} on process {}".format(fcs_filename, current_process().pid))
    fcs_data = data[data.obs.FCS_File == fcs_filename]
    fcs_X = fcs_data.X

    # Load the phi vector
    files = os.listdir(output_data_path)
    files = [i for i in files if i.startswith(fcs_filename.split(".")[0])]
    gammas = list(set([float(i.split("gamma")[1].split("_")[0]) for i in files]))
    geo_indices, geo_samples, geo_rf = geosketch_main(fcs_X, num_samples_per_set)
    hop_indices, hop_samples, hop_rf = hopper_main(fcs_X, num_samples_per_set)

    geo_sample_data = fcs_data[fcs_data.obs.iloc[geo_indices].index]
    hop_sample_data = fcs_data[fcs_data.obs.iloc[hop_indices].index]

    print("Finished Geosketch on {} on process {}. Data size = {}".format(fcs_filename, current_process().pid, geo_sample_data.X.shape))
    print("Finished Hopper on {} on process {}. Data size = {}".format(fcs_filename, current_process().pid, hop_sample_data.X.shape))

    for gamma in gammas:
        phi = np.load(os.path.join(output_data_path, "{}_gamma{}_phi.npy".format(fcs_filename.split(".")[0], gamma)))
        geo_rf = phi[geo_indices]
        hop_rf = phi[hop_indices]
        np.save(os.path.join(output_data_path, "{}_gamma{}_georf.npy".format(fcs_filename.split(".")[0], gamma)), geo_rf)
        np.save(os.path.join(output_data_path, "{}_gamma{}_hoprf.npy".format(fcs_filename.split(".")[0], gamma)), hop_rf)

    geo_sample_data.write(os.path.join(output_data_path, "geo_subsamples_{}k_per_set_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0])))
    hop_sample_data.write(os.path.join(output_data_path, "hop_subsamples_{}k_per_set_{}.h5ad".format(num_samples_per_set / 1000, fcs_filename.split(".")[0])))


def get_gamma_range(X):
    from scipy.spatial import distance_matrix
    inds = np.random.choice(X.shape[0], size=100)
    distances = distance_matrix(X[inds], X[inds])
    gamma_0 = np.median(distances)
    gammas = [gamma_0 / i for i in (4, 3, 2, 1, 0.5, 0.33, 0.25, 0.1)]

    return gammas

#
# pool = Pool(processes=num_processes)
# if(proc == 'geo'):
#     pool.map(parallel_geo_hopper_subsampling, fcs_file)
# else:
#     pool.map(parallel_kh_subsampling, fcs_file)
# pool.close()


# RFE Metric Evaluation
def eval_summary(vec, beta):
  return np.mean(np.dot(vec, beta), axis=0)


def rfe_evaluation(phi, iid_rf, kh_rf, geo_rf, hop_rf):
    eval = []
    for i in range(5):
      beta = np.random.normal(size=2000).reshape(-1,1)

      orig_eval = eval_summary(phi, beta)[0]
      kh_eval = eval_summary(kh_rf, beta)[0]
      iid_eval = eval_summary(iid_rf, beta)[0]
      geo_eval = eval_summary(geo_rf, beta)[0]
      hop_eval = eval_summary(hop_rf, beta)[0]
      eval.append([orig_eval, kh_eval, iid_eval, geo_eval, hop_eval])

    df = pd.DataFrame(eval, columns = ["Orig dataset", "KH", "IID", "Geosketch", "Hopper"])
    print(df.head())
    subset_methods = np.asarray(eval)[:, 1:]
    original_data = np.asarray(eval)[:, 0].reshape(-1, 1)

    avg_l1 = np.mean(np.abs(subset_methods - original_data), axis=0)

    print(avg_l1)
    return avg_l1



files = os.listdir(output_data_path)
samples = [i for i in set([i.split("_")[0] for i in files]) if len(i) > 3]

def rfe_driver(fcs_filename):
    print(fcs_filename)
    fcs_files = [i for i in files if i.startswith(fcs_filename)]
    gammas = list(set([i.split("gamma")[1].split("_")[0] for i in fcs_files]))
    l1 = []
    for gamma in gammas:
        phi = np.load(os.path.join(output_data_path, "{}_gamma{}_phi.npy".format(fcs_filename, gamma)))
        iid_rf = np.load(os.path.join(output_data_path, "{}_gamma{}_iidrf.npy".format(fcs_filename, gamma)))
        kh_rf = np.load(os.path.join(output_data_path, "{}_gamma{}_khrf.npy".format(fcs_filename, gamma)))
        geo_rf = np.load(os.path.join(output_data_path, "{}_gamma{}_georf.npy".format(fcs_filename, gamma)))
        hop_rf = np.load(os.path.join(output_data_path, "{}_gamma{}_hoprf.npy".format(fcs_filename, gamma)))

        l1.append(rfe_evaluation(phi, iid_rf, kh_rf, geo_rf, hop_rf))

    return gammas, np.asarray(l1)


fil = samples[2]
gammas, l1 = rfe_driver(fil)
gammas = [gamma[:5] for gamma in gammas]
plt.plot(gammas, l1[:, 0], label='KH')
plt.plot(gammas, l1[:, 1], label='IID')
plt.plot(gammas, l1[:, 2], label='Geo')
plt.plot(gammas, l1[:, 3], label='Hopper')
plt.title("sample = {}".format(fil))
plt.xlabel("Gamma values")
plt.ylabel("Avg L1 distance to Original RF metric")
plt.legend()
plt.show()
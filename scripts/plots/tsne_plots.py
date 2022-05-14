from hopper import treehopper, PCATreePartition, hopper
from geosketch import gs

import anndata
import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns




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




# a_001 = 6k total, 200 num_subsamples
# a_002 = 15k total, 500 num_subsamples


np.random.seed(0)
fcs_data = anndata.read_h5ad("a_002_NK.h5ad")
fcs_X = fcs_data.X
X = fcs_X
gammas = get_gamma_range(fcs_X)
gamma = gammas[3]
phi = random_feats(fcs_X, gamma)

num_subsamples = 500
iid_indices = np.random.choice(phi.shape[0], size=num_subsamples, replace=False)
iid_rf, iid_samples = phi[iid_indices], X[iid_indices]


kh_indices, kh_samples, kh_rf = kernel_herding_main(X, phi, num_subsamples)
geo_indices, geo_samples, geo_rf = geosketch_main(X, num_subsamples)
hop_indices, hop_samples, hop_rf = hopper_main(X, num_subsamples)

num_clusters = 15
km = KMeans(init="k-means++", n_clusters=num_clusters, n_init=4,
                random_state=0)
orig_preds = km.fit_predict(X)
kh_samples = np.asarray(kh_samples)
kh_preds = km.predict(kh_samples)
iid_preds = km.predict(iid_samples)
geo_preds = km.predict(geo_samples)
hop_preds = km.predict(hop_samples)


# 3rd nearest neighbor search
def nnbr(mat):
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=4)       # each point is its own nearest nbr
    knn.fit(mat)
    dist, inds = knn.kneighbors(mat)
    return dist[:,-1], inds

geo_dist, geo_nbr = nnbr(geo_samples)
kh_dist, kh_nbr = nnbr(kh_samples)
iid_dist, iid_nbr = nnbr(iid_samples)
hop_dist, hop_nbr = nnbr(hop_samples)
combined_samples = np.vstack([geo_samples, hop_samples, iid_samples, kh_samples])
comb_dist, comb_nbr = nnbr(combined_samples)
comb_geo = comb_dist[:len(geo_samples)]
comb_kh = comb_dist[-1:-501:-1]
comb_hop = comb_dist[500:1000]
comb_iid = comb_dist[1000:1500]
# comb_rest = comb_dist[1000:]

size = 6000
subs_indices = np.random.choice(X.shape[0], size=size, replace=False)
orig_subs_index_mapping = dict(zip(subs_indices, range(size)))
X_subsampled = X[subs_indices]
cluster_centers = km.cluster_centers_

#X_subsampled2 = np.vstack((X_subsampled, cluster_centers, kh_samples, geo_samples, hop_samples))
X_subsampled2 = np.vstack((X_subsampled, cluster_centers, kh_samples, iid_samples, geo_samples, hop_samples))
tsne = TSNE(n_components=2, perplexity=40, verbose=1)
tsne = tsne.fit_transform(X_subsampled2)
x_tsne = tsne[:size, :]
ccenter_tsne = tsne[size:size+num_clusters, :]
kh_tsne = tsne[size+num_clusters: size+num_clusters+num_subsamples, :]
iid_tsne = tsne[size+num_clusters+num_subsamples: size+num_clusters+2*num_subsamples, :]
geo_tsne = tsne[size+num_clusters+2*num_subsamples: size+num_clusters+3*num_subsamples, :]
hop_tsne = tsne[size+num_clusters+3*num_subsamples:, :]
rest_tsne = np.vstack((iid_tsne, kh_tsne))


# Plot
comb_geo[130] = 12
comb_geo[202] = 12
comb_kh[73] = 8
comb_hop[1] = 12
comb_hop[3] = 12


# sns.set(rc={'figure.figsize':(10,10)})
sns.set_style("white")
### Start 3rd NNBR tsne heatmap
np.random.seed(0)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
min_val = min(comb_geo.min(), comb_kh.min(), comb_hop.min(), comb_iid.min())
max_val = max(comb_geo.max(), comb_kh.max(), comb_hop.max(), comb_iid.max())
norm = plt.Normalize(min_val, max_val)
sm = plt.cm.ScalarMappable(cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), norm=norm)
palette = 'ch:s=-.2,r=.6'
sm.set_array([])
sns.scatterplot(x=kh_tsne[:,0], y=kh_tsne[:,1], hue=comb_kh, palette=palette, ax=axes[0], s=50, hue_norm=(min_val, max_val))
axes[0].get_legend().remove()
axes[0].title.set_text("Kernel Herding")

sns.scatterplot(x=iid_tsne[:,0], y=iid_tsne[:,1], hue=comb_iid, palette=palette, ax=axes[1], s=50, hue_norm=(min_val, max_val))
axes[1].get_legend().remove()
axes[1].figure.colorbar(sm)
axes[1].title.set_text("IID")

sns.scatterplot(x=geo_tsne[:,0], y=geo_tsne[:,1], hue=comb_geo, palette=palette, ax=axes[0], s=50, hue_norm=(min_val, max_val))
axes[0].get_legend().remove()
axes[0].title.set_text("Geo-Sketch")

sns.scatterplot(x=hop_tsne[:,0], y=hop_tsne[:,1], hue=comb_hop, palette=palette, ax=axes[1], s=50, hue_norm=(min_val, max_val))
axes[1].get_legend().remove()
# axes[1].figure.colorbar(sm)
axes[1].title.set_text("Hopper")


# plt.show()
# plt.savefig("kh_iid_3rd_nnbr.png", dpi=600, transparent=False, bbox_inches='tight')
plt.savefig("tsne color bar.png", dpi=600, transparent=False, bbox_inches='tight')

### End 3rd NNBR tsne heatmap





np.random.seed(0)
scatter_inds = np.random.choice(x_tsne.shape[0], size=size//2)
ax = sns.scatterplot(x=x_tsne[scatter_inds,0], y=x_tsne[scatter_inds,1], linewidth=0.9, color='0.7', marker = '+', s=150)
sns.scatterplot(x=ccenter_tsne[:,0], y=ccenter_tsne[:,1], marker='X', s=500, color='orange', ax=ax)




sns.scatterplot(x=iid_tsne[:,0], y=iid_tsne[:,1], color='0.8')
sns.scatterplot(x=kh_tsne[:,0], y=kh_tsne[:,1], color='0.8')

custom = [Line2D([], [], marker='+', color='0.7', linestyle='None'),
          Line2D([], [], marker='.', color='orange', linestyle='None'),
          Line2D([], [], marker='.', color='0.8', linestyle='None'),
          Line2D([], [], marker='.', color='0.8', linestyle='None'),
          Line2D([], [], marker='.', color='0.8', linestyle='None'),
          Line2D([], [], marker='.', color='0.8', linestyle='None')
          ]

# sns.scatterplot(x=hop_tsne[:,0], y=hop_tsne[:,1], color='#008080')
# sns.scatterplot(x=geo_tsne[:,0], y=geo_tsne[:,1], color='#DDA0DD')
# sns.scatterplot(x=iid_tsne[:,0], y=iid_tsne[:,1], color='red')
# sns.scatterplot(x=kh_tsne[:,0], y=kh_tsne[:,1], color='#580F41')

# custom = [Line2D([], [], marker='+', color='0.7', linestyle='None'),
#           Line2D([], [], marker='.', color='orange', linestyle='None'),
#           Line2D([], [], marker='.', color='#008080', linestyle='None'),
#           Line2D([], [], marker='.', color='#DDA0DD', linestyle='None'),
#           Line2D([], [], marker='.', color='red', linestyle='None'),
#           Line2D([], [], marker='.', color='#580F41', linestyle='None')
#           ]

plt.xlabel("tSNE-dimension-1")
plt.ylabel("tSNE-dimension-2")
#plt.legend(custom, ["Original", "Cluster Centers",  "Hopper", "Geo", "KH"], loc='best')
plt.legend(custom, ["Original", "Cluster Centers",  "Hopper", "Geo", "IID", "KH"], loc='best')
plt.title("PreE Cluster tSNE")
plt.savefig("PBMC_1063411248_PreE_unstim PreE Cluster tSNE with_IID.png", dpi=600, transparent=False, bbox_inches='tight')




np.save("x_tsne.npy", x_tsne)
np.save("ccenter_tsne.npy", ccenter_tsne)
np.save("kh_tsne.npy", kh_tsne)
np.save("iid_tsne.npy", iid_tsne)
np.save("geo_tsne.npy", geo_tsne)
np.save("hop_tsne.npy", hop_tsne)



x_tsne = np.load("x_tsne.npy")
ccenter_tsne = np.load("ccenter_tsne.npy")
kh_tsne = np.load("kh_tsne.npy")
iid_tsne = np.load("iid_tsne.npy")
geo_tsne = np.load("geo_tsne.npy")
hop_tsne = np.load("hop_tsne.npy")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import seaborn as sns

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2


dataset = 'nk'

source_path = '/home/athreya/private/set_summarization/data/'
if(dataset == 'hvtn'):
    data_path = os.path.join(source_path, "hvtn", "classification_results")
    figure_save_name = "HVTN"
    file_save_name = "HVTN"
elif(dataset == 'nk'):
    data_path = os.path.join(source_path, "nk", "classification_results")
    figure_save_name = "NK Cell"
    file_save_name = "NK"
else:
    # Preeclampsia
    data_path = os.path.join(source_path, "preeclampsia", "classification_results")
    figure_save_name = "Preeclampsia"
    file_save_name = "PreE"


# Plotting LOO-CV
def LOO_CV_plot():
    subsamples = 500
    kh_15 = pd.read_csv(os.path.join(data_path, "loo_classification_results_kh_{}subsamples_3sketches_15clusters.csv".format(subsamples / 1000)))
    others_15 = pd.read_csv(os.path.join(data_path, "loo_classification_results_others_{}subsamples_3sketches_15clusters.csv".format(subsamples / 1000)))
    kh_30 = pd.read_csv(os.path.join(data_path, "loo_classification_results_kh_{}subsamples_3sketches_30clusters.csv".format(subsamples / 1000)))
    others_30 = pd.read_csv(os.path.join(data_path, "loo_classification_results_others_{}subsamples_3sketches_30clusters.csv".format(subsamples / 1000)))
    kh_50 = pd.read_csv(os.path.join(data_path, "loo_classification_results_kh_{}subsamples_3sketches_50clusters.csv".format(subsamples / 1000)))
    others_50 = pd.read_csv(os.path.join(data_path, "loo_classification_results_others_{}subsamples_3sketches_50clusters.csv".format(subsamples / 1000)))

    # Convert values from list of strings, if needed, to float. If it throws an error, values are already in float
    others_30['Acc'] = others_30['Acc'].apply(lambda x: x[1:len(x)-1])
    others_50['Acc'] = others_50['Acc'].apply(lambda x: x[1:len(x)-1])
    others_30['Acc'] = others_30['Acc'].astype(float)
    others_50['Acc'] = others_50['Acc'].astype(float)


    final_15 = pd.concat((others_15, kh_15[['Sketch1', 'Sketch2', 'Trial #', 'Method', 'Acc']]), axis=0)
    final_30 = pd.concat((others_30, kh_30[['Sketch1', 'Sketch2', 'Trial #', 'Method', 'Acc']]), axis=0)
    final_50 = pd.concat((others_50, kh_50[['Sketch1', 'Sketch2', 'Trial #', 'Method', 'Acc']]), axis=0)

    # Considering only 30 clusters for final paper plot
    final_30 = final_30.loc[final_30['Method'] != 'kh']			# Chucking static kh scale factor, considering only kh hyperparameter
    final_30.replace({"geo": "Geo-Sketch", "hop": "Hopper", "iid": "IID", "kh_hyperparam": "Kernel Herding"}, inplace=True)
    my_pal = {method: sns.color_palette("muted")[3] if method == "Kernel Herding" else sns.color_palette("muted")[-1] for method in final_30.Method.unique()}

    assert final_15.shape[0] == kh_15.shape[0] + others_15.shape[0]
    assert final_30.shape[0] == kh_30.shape[0] + others_30.shape[0]
    assert final_50.shape[0] == kh_50.shape[0] + others_50.shape[0]


    # f1 = final_15.boxplot(column="Acc", by='Method', showmeans=True, showcaps=True)
    # plt.legend()
    # plt.title("15 clusters, 6 sketch combos, 5 trials per combo, {} subsamples".format(subsamples))
    # f1.set_ylim(0, 1)
    # plt.show()


    # Using only 30 clusters for LOO-CV plot since 5-Fold CV looks better
    f2 = sns.boxplot(x=final_30["Method"], y=final_30["Acc"], palette=my_pal, order=['Geo-Sketch', 'Hopper', 'IID', 'Kernel Herding'])
    plt.xlabel("")
    plt.ylabel("Classification Accuracy")
    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=10)
    #f2 = final_30.boxplot(column="Acc", by='Method', showmeans=True, showcaps=True)
    plt.legend()
    plt.ylim([0,1])
    plt.title("{}".format(figure_save_name))
    # plt.show()
    plt.savefig(os.path.join(data_path, '{}_LOO_30clusters_new.png'.format(file_save_name)), dpi=600, transparent=False, bbox_inches='tight')


    # f3 = final_50.boxplot(column="Acc", by='Method', showmeans=True, showcaps=True)
    # plt.legend()
    # plt.title("50 clusters, 6 sketch combos, 5 trials per combo, {} subsamples".format(subsamples))
    # f3.set_ylim(0, 1)
    # plt.show()



# Plotting all clusters 5-Fold CV
def five_fold_CV_plot():
    cv = pd.read_csv(os.path.join(data_path, "5fold_cv_classification_results_1.0.csv"))
    #cv = pd.read_csv(os.path.join(data_path, "5fold_cv_classification_results_{}subsamples_1.0.csv".format(subsamples / 1000)))
    cv.replace({"geo": "Geo-Sketch", "hop": "Hopper", "iid": "IID", "kh": "Kernel Herding"}, inplace=True)

    my_pal = {method: sns.color_palette("muted")[3] if method == "Kernel Herding" else sns.color_palette("muted")[-1] for method in cv.subsampling.unique()}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))

    a2_15 = cv.loc[cv['clusters']==15]
    sns.boxplot(x=a2_15["subsampling"], y=a2_15["Acc"], palette=my_pal, ax=ax1)
    #a2_15.boxplot(column="Acc", by='subsampling', showmeans=True, showcaps=True, ax=ax1, grid=False)
    ax1.title.set_text('15 Clusters')
    ax1.set_ylabel("Classification Accuracy", fontsize=13)
    ax1.set_xlabel("")
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)

    a2_30 = cv.loc[cv['clusters']==30]
    sns.boxplot(x=a2_30["subsampling"], y=a2_30["Acc"], palette=my_pal, ax=ax2)
    #a2_30.boxplot(column="Acc", by='subsampling', showmeans=True, ax=ax2, grid=False)
    ax2.title.set_text('30 Clusters')
    ax2.set_xlabel("")
    ax2.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='x', labelsize=10)

    a2_50 = cv.loc[cv['clusters']==50]
    sns.boxplot( x=a2_50["subsampling"], y=a2_50["Acc"], palette=my_pal, ax=ax3)
    #a2_50.boxplot(column="Acc", by='subsampling', showmeans=True, ax=ax3, grid=False)
    ax3.title.set_text('50 Clusters')
    ax3.set_xlabel("")
    ax3.tick_params(axis='y', labelsize=10)
    ax3.tick_params(axis='x', labelsize=10)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)

    plt.suptitle("{} Dataset".format(figure_save_name))
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(data_path, '{}_5-fold_multiple_clusters.png'.format(file_save_name)), dpi=600, transparent=False, bbox_inches='tight')




    # Separate figure with only 30 clusters 5-Fold CV
    sns.boxplot(x=a2_15["subsampling"], y=a2_15["Acc"], palette=my_pal)
    plt.xlabel("")
    plt.ylabel("Classification Accuracy")
    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.ylim(0,1)
    plt.title("{} Dataset using 15 clusters".format(figure_save_name))
    #plt.show()
    plt.savefig(os.path.join(data_path, '{}_5-fold_15clusters.png'.format(file_save_name)), dpi=600, transparent=False, bbox_inches='tight')

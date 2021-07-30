from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import os
import anndata

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def download_gdrive_file(file_id, output_file_name):
    """
    Requires the client_secrets.json file to be present in the working directory. Can be obtained from Google Developer console (under Google Drive service API)
    :param file_id: Drive unique File ID (can be obtained from file's shareable link)
    :param output_file_name: Filename to be downloaded as
    :return: Downloads file as "output_file_name" into working directory
    """

    gauth = GoogleAuth()
    gauth.CommandLineAuth() # Use this if authenticating through SSH. Authenticate in local browser and copy-paste auth code into terminal
    # gauth.LocalWebserverAuth() # Use this if authenticating from local machine
    drive = GoogleDrive(gauth)
    file_obj = drive.CreateFile({'id': file_id})
    file_obj.GetContentFile(output_file_name)


def merge_anndata(folder_path, file_regex, output_filename):
    files = os.listdir(folder_path)
    to_merge = [i for i in files if file_regex in i]
    print(len(set(to_merge)), to_merge[0])
    anndata_lst = [anndata.read_h5ad(os.path.join(folder_path,i)) for i in to_merge]
    merged_data = anndata.concat(anndata_lst)

    merged_data.write(os.path.join(folder_path, output_filename))

    return merged_data

folder_path = "/home/athreya/private/set_summarization/data/hop_samples"
file_regex = "hop"
output_filename = "hop_subsamples_0.5k_per_set.h5ad"
merge_anndata(folder_path, file_regex, output_filename)


def get_preeclampsia_data(data_path):
    file_list = os.listdir(data_path)
    metadata_file = [i for i in file_list if 'file_list.csv' in i][0]
    refer = pd.read_csv(os.path.join(data_path, metadata_file))
    adata_lst = []
    for fil in file_list:
        if(fil != metadata_file):
            temp = pd.read_csv(os.path.join(data_path, fil))
            file_name = fil.split(".")[0]
            label_type = refer.loc[refer['FCS_file_name'] == file_name]['Group'].values[0]
            label = 1 if label_type == 'Control' else 0
            obs = pd.DataFrame([file_name]*temp.shape[0], columns=['FCS_File'])
            obs['label'] = label
            adata = anndata.AnnData(temp, obs=obs)
            adata.var['markers'] = temp.columns
            adata.var.index = range(1, temp.shape[1] + 1)
            adata_lst.append(adata)

    merged_data = anndata.concat(adata_lst)
    merged_data.obs.index = [str(i) for i in range(merged_data.shape[0])]
    # merged_data.X = np.arcsinh((1. / 5) * merged_data.X)
    merged_data.X = StandardScaler().fit_transform(merged_data.X)
    merged_data.write(os.path.join(data_path, "preeclampsia_preprocessed.h5ad"))
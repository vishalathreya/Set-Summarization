from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import os
import anndata

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
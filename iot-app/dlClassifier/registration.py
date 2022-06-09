import numpy as np
import os

from triplet_fp_UakNN_no_plot import UakNN_params, load_knn_models

import pickle
import zipfile
from tensorflow import keras
from sklearn.neighbors import NearestNeighbors
from itertools import combinations

unit_type = np.int16
seg_len = 1024

# data and model paths
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)

def register_add_npz(data_dir, mac, old_npz_path, new_npz_path):
    bin_files = os.listdir (data_dir)
    data_list = []

    for bin_file in bin_files:
        bin_mac = bin_file.split('_') [1]
        if bin_mac == mac:
            data = np.fromfile (os.path.join(data_dir, bin_file), dtype=unit_type)
            data_i, data_q = data[::2], data[1::2]
            data_i, data_q = data_i[:seg_len], data_q[:seg_len]
            data_i, data_q = np.expand_dims(data_i, axis=1), np.expand_dims (data_q, axis=1) 
            data = np.concatenate([data_i, data_q], axis=1)
            data = np.expand_dims(data, axis=0)
            data_list.append(data)
    data_mac = np.concatenate (data_list, axis=0)

    old_npz_file = np.load (old_npz_path, allow_pickle=True)
    old_data, old_labels = old_npz_file['data'], old_npz_file['labels'] 
    old_label_dict = dict(enumerate (old_npz_file['label_dict'].flatten(), 1))[1]
    
    old_data = old_data.flatten().reshape(3206,1024,2) # **** ekledim ülkü sor

    new_data = np.concatenate([old_data, data_mac], axis=0)
    int_labels = [old_label_dict[key] for key in old_label_dict.keys()]    
    # integer Labels must be ordered

    assert len(int_labels) == np.max (int_labels) + 1 #if not orderly throw error if assert fails

    mac_label = np.max(int_labels) + 1
    mac_labels = np.array([mac] * data_mac.shape[0])
    new_labels = np.concatenate([old_labels, mac_labels], axis=0)
    new_label_dict = old_label_dict
    new_label_dict[mac] = mac_label

    np.savez(new_npz_path, data=new_data, labels=new_labels, label_dict=new_label_dict)

    """
    working_dir = "/home/mp6/code_projects/iotAppEndtoEnd"
    data_dir = os.path.join(working_dir, "dlClassifier/data")
    train_npz_fname = "train_data_registration_deneme.npz" #"train_data_sep17.npz" # **
    """

    #train_data_path = os.path.join(data_dir, train_npz_fname)
    train_npz_file = np.load(new_npz_path, allow_pickle=True)

    
    train_data, train_labels = train_npz_file['data'], train_npz_file['labels']
    train_label_dict = dict(enumerate(train_npz_file['label_dict'].flatten(), 1))[1]

    train_data = (train_data-np.mean(train_data, axis=(1, 2), keepdims=True)) / np.std(train_data, axis=(1, 2), keepdims=True)

    train_int_labels = [int(train_label_dict[train_label]) for train_label in train_labels]
    train_int_labels = np.array(train_int_labels)
    y_train = np.reshape(train_int_labels, train_data.shape[0])
    neighbor_count = len(train_label_dict)

    model_dir = os.path.join(PROJECT_ROOT, "dlClassifier/models")
    triplet_model_path = os.path.join(model_dir, "triplet_model")
    knn_model_fname = "knn_model"
    UakNN_model_fname = "UakNN_model"

    model = keras.models.load_model(triplet_model_path, compile=False)

    train_fp = model.predict(train_data)

    reg_label_dict = {}
    reg_label_dict[mac] = train_label_dict[mac]
    #UakNN_params = UakNN_train(data=train_fp, labels=train_int_labels, label_dict=train_label_dict) # initial
    #UakNN_params['alpha'] = alpha
    UakNN_params = load_knn_models(model_dir=model_dir, model_fname=UakNN_model_fname)

    UakNN_params = UakNN_update(train_fp, train_int_labels, reg_label_dict, UakNN_params) #it is faster
    nbrs = NearestNeighbors(n_neighbors=neighbor_count, algorithm='ball_tree', metric='euclidean').fit(train_fp)


    save_knn_models(model=nbrs, model_dir=model_dir, model_fname=knn_model_fname)
    save_knn_models(model=UakNN_params, model_dir=model_dir, model_fname=UakNN_model_fname)

    return nbrs, UakNN_params



def forget_mac_npz(mac, old_npz_path, new_npz_path):
    old_npz_file = np.load(old_npz_path, allow_pickle=True)
    old_data, old_labels = old_npz_file['data'], old_npz_file['labels']
    old_label_dict = dict(enumerate(old_npz_file['label_dict'].flatten(), 1))[1]
    int_labels = [old_label_dict[key] for key in old_label_dict.keys()]

    # integer labels must be ordered
    assert len(int_labels) == np.max(int_labels) + 1 
    new_data, new_labels = old_data[old_labels != mac], old_labels[old_labels != mac]
    new_label_dict_temp = old_label_dict.copy()
    del new_label_dict_temp[mac]

    old_int_label= old_label_dict[mac]
    new_label_dict = {}

    # arrange label dict so that int labels are ordered after mac address is deleted
     
    for key in new_label_dict_temp.keys():
        temp_int_label = new_label_dict_temp[key]
        
        if temp_int_label > old_int_label:
            new_label_dict[key] = temp_int_label - 1
        else:
            new_label_dict[key] = temp_int_label

    np.savez(new_npz_path, data=new_data, labels=new_labels, label_dict=new_label_dict)


def clean_npz_label_names(npz_path):
    """
    old_label_names: mac_channel
    new_label_names: mac

    saves data with modified Label names
    """
    npz_file = np.load(npz_path, allow_pickle=True)
    data, labels = npz_file['data'], npz_file ['labels']
    label_dict = dict(enumerate(npz_file ['label_dict'].flatten(), 1))[1]

    new_label_dict = {}

    new_labels = labels.copy()
    
    for key in label_dict.keys():
        new_key= key.split('_')[0]
        new_labels[new_labels == key] = new_key
        new_label_dict[new_key] = label_dict[key]
    
    np.savez(npz_path, data=data, labels=new_labels, label_dict=new_label_dict)

def save_knn_models (model, model_dir, model_fname):
    model_path = os.path.join(model_dir, model_fname)
    with open(model_path + ".p", 'wb') as f:
        pickle.dump (model, file=f)


def UakNN_train(data, labels, label_dict):
    UakNN_params = {}

    for key in label_dict.keys():
        i = label_dict[key]
        data_part = data[labels == i]

        comb_inds = np.arange(data_part.shape[0])
        comb_inds = combinations(comb_inds, 2)
        comb_inds = np.array(list(comb_inds))

        data_part_p, data_part_q = data_part[comb_inds[:, 0]], data_part[comb_inds[:, 1]]
        dists = np.sqrt(np.sum(np.square(data_part_p - data_part_q), axis=1))

        mu_i, sigma_i = np.mean(dists), np.std(dists)
        UakNN_params[i] = {"mu": mu_i, 'sigma': sigma_i}

    return UakNN_params


def UakNN_update(data, labels, reg_label_dict, UakNN_params):
    for key in reg_label_dict.keys():
        i = reg_label_dict[key]
        data_part = data[labels == i]

        comb_inds = np.arange(data_part.shape[0])
        comb_inds = combinations(comb_inds, 2)
        comb_inds = np.array(list(comb_inds))

        data_part_p, data_part_q = data_part[comb_inds[:, 0]], data_part[comb_inds[:, 1]]
        dists = np.sqrt(np.sum(np.square(data_part_p - data_part_q), axis=1))

        mu_i, sigma_i = np.mean(dists), np.std(dists)
        UakNN_params[i] = {"mu": mu_i, 'sigma': sigma_i}

    return UakNN_params


if __name__ == '__main__':
    print("as")
    data_dir= "/home/esen-baha/esen_iot/iot-app/register" 
    mac = "F08A76FE6868"
    old_npz_path = "/home/esen-baha/esen_iot/iot-app/dlClassifier/data/train.npz"
    new_npz_path="/home/esen-baha/esen_iot/iot-app/dlClassifier/data/train_data_registration.npz" 
    register_add_npz(data_dir, mac, old_npz_path, new_npz_path)
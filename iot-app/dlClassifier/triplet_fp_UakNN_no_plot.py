import json
from re import I
from matplotlib.font_manager import json_dump
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt


import pyinotify
import os
import sys

from datetime import datetime
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

import redis

load_dotenv()

r = redis.StrictRedis(host="redis-server", port=6379,charset="utf-8", decode_responses=True)

DECISION_THRESHOLD = 5
noPlotMode = True
rfs = {}

#check whether plot exists for given mac
#decision Accepted ya da Rejected
def check_plot(mac, decision):
    obj = r.hgetall("rfs:" + mac)

    if not obj:
        return False

    if decision != 'Unknown':
        decision, a = decision_confidence(int(obj['ApprovedCount'])+1, int(obj['RejectedCount']))
    else:
        decision, a = decision_confidence(int(obj['ApprovedCount']), int(obj['RejectedCount'])+1)

    return False

def decision_confidence (approve_count, reject_count):
    if approve_count >= reject_count:
        decision = "Approved"
        confidence = 100* approve_count / (approve_count + reject_count)

    else:
        decision = "Rejected"
        confidence = 100* reject_count/ (approve_count + reject_count)

    return decision, str(confidence)

def redis_controller(decision, mac):
    obj = r.hgetall('rfs:' + mac)

    if not obj:
        obj = {
            'MAC': mac,
            'Label': '',
            'SignalDecision': '',
            'Confidence': '',
            'RejectInfo': '',
            'LastUpdateTime': 'Şimdi Güncellendi',
            'ApprovedCount': 0,
            'RejectedCount': 0,
            'ImagePathApproved': 'abbas', #plot*
            'ImagePathRejected': 'abbas',
            'HistoricalResults': json.dumps({})
        }

    obj['HistoricalResults'] = json.loads(obj['HistoricalResults'])

    print()

    if decision != 'Unknown':
        obj['SignalDecision'] = decision
        obj['ApprovedCount'] = int(obj['ApprovedCount']) + 1
        obj['RejectedCount'] = int(obj['RejectedCount'])
    else:
        obj['SignalDecision'] = 'Unknown'
        obj['ApprovedCount'] = int(obj['ApprovedCount'])
        obj['RejectedCount'] = int(obj['RejectedCount']) + 1

    
    if decision in obj['HistoricalResults']:
        obj['HistoricalResults'][decision] = int(obj['HistoricalResults'][decision]) + 1
    else:
        obj['HistoricalResults'][decision] = 1

    
    if int(obj['ApprovedCount']) < int(obj['RejectedCount']) \
        and int(obj['ApprovedCount']) + int(obj['RejectedCount']) >= DECISION_THRESHOLD:
        obj['SignalDecision'] = 'Unknown'


    rejectStatus, confidence = decision_confidence(int(obj['ApprovedCount']), int(obj['RejectedCount']))

    obj['RejectInfo'] = rejectStatus
    obj['Confidence'] = confidence
    obj['Label'] = max(obj['HistoricalResults'], key=obj['HistoricalResults'].get)
    obj['HistoricalResults'] = json.dumps(obj['HistoricalResults'])

    
    print(obj['HistoricalResults'])

    print(obj['HistoricalResults'], type(obj['HistoricalResults']))
    
    rfs['rfs:' + obj['MAC']] = obj
    r.hmset('rfs:' + obj['MAC'], obj)
    r.publish('app:notifications', json.dumps(obj))


def load_knn_models(model_dir, model_fname):
    model_path = os.path.join(model_dir, model_fname)

    with open(model_path + ".p", 'rb') as f:
        model = pickle.load(open(model_path + ".p", "rb"))
    return model


def get_data_fingerprint(data, model):
    seglen = int(os.getenv('SEGLEN'))
    data_i, data_q = data[::2], data[1::2]
    data_i, data_q = data_i[:seglen], data_q[:seglen]
    data_i, data_q = np.expand_dims(data_i, axis=1), np.expand_dims(data_q, axis=1)
    data_i_q = np.concatenate([data_i, data_q], axis=1)
    data_i_q = np.expand_dims(data_i_q, axis=0)
    data_i_q = (data_i_q - np.mean(data_i_q, axis=(1, 2), keepdims=True)) / np.std(data_i_q, axis=(1, 2), keepdims=True)
    data_fp = model.predict(data_i_q)
    return data_fp

def UakNN_predict(data, knn_model, UakNN_model, alpha, train_labels, train_label_dict):
    data_distances, data_indices = knn_model.kneighbors(data)
    data_pred_labels = train_labels[data_indices]

    data_pred_list, data_mean_list = [], []
    for data_pred_k_label, data_dist in zip(data_pred_labels, data_distances):
        weight_dists = 1 / np.array(data_dist)
        label_ind = np.bincount(data_pred_k_label, minlength=len(np.unique(train_labels))+1, weights=weight_dists).argmax()
        data_pred_list.append(label_ind)

        dec_dist = data_dist[data_pred_k_label == label_ind]
        data_mean_list.append(np.mean(dec_dist))

    data_pred_list = np.array(data_pred_list)
    data_mean_list = np.array(data_mean_list)

    lbl = data_pred_list[0]
    dist = data_mean_list[0]
    mu_lbl, sigma_lbl = UakNN_model[lbl]['mu'], UakNN_model[lbl]['sigma']

    #train_label_key_list = list(train_label_dict.keys())
    train_label_value_list = list(train_label_dict.values())

    print("TRAIN LABEL DICT", train_label_dict)


    dec = train_label_value_list[lbl]
    return dec


def plot_TSNE(train_data, train_labels, train_label_dict, data_fp, decision, mac, class_size, save_plot_dir):
    if decision == 'Unknown':
        new_data_fp = np.random.normal(data_fp, np.array([3] * data_fp.shape[1]), size=class_size)
    else:
        new_data_fp = data_fp

    dt_now = datetime.now()
    str_now_date = dt_now.strftime("%d%m%Y")

    str_now_time = str(dt_now).split(" ")[1]
    str_now_time = str_now_time.split(".")[0]
    str_now_time = str_now_time.replace(":", "")

    image_fname = '_'.join([mac, decision, str_now_date, str_now_time]) + ".png"

    unknown_int_label = np.max(train_labels) + 1
    all_fp = np.concatenate([train_data, new_data_fp], axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
    tsne_results = tsne.fit_transform(all_fp)
    
    label_dict_all = train_label_dict.copy()
    label_dict_all['test data'] = unknown_int_label

    all_labels = train_labels
    all_labels = all_labels.tolist()
    all_labels.extend([unknown_int_label] * new_data_fp.shape[0])
    all_labels = np.array(all_labels)

    fig = plt.figure(figsize=(12, 10))

    cmap = plt.cm.get_cmap('nipy_spectral', len(label_dict_all))
    for label_name in label_dict_all.keys():
        int_label = label_dict_all[label_name]
        if int_label == unknown_int_label:
            x, y = tsne_results[all_labels == int_label, 0], tsne_results[all_labels == int_label, 1]
            x, y = np.mean(x), np.mean(y)
            plt.plot(x, y, color=cmap(int_label), marker='*', markersize=24)
        else:
            plt.scatter(tsne_results[all_labels == int_label, 0], tsne_results[all_labels == int_label, 1], c=cmap(int_label),
                        label=label_name, alpha=0.5)

    plt.legend(fontsize='small')
    image_path = os.path.join(save_plot_dir, image_fname)
    plt.savefig(image_path)
    plt.close()

    return image_path

def get_model_decision(bin_dir, bin_fname, alpha, UakNN_params, knn_model, 
        triplet_model, train_int_labels, train_label_dict, train_data,
        train_labels):
    # load test data
    unit_type = np.int16
    bin_path = os.path.join(bin_dir, bin_fname)
    data = np.fromfile(bin_path, dtype=unit_type)

    # raw data to fingerprint
    data_fp = get_data_fingerprint(data=data, model=triplet_model)


    train_label_dict_rev = {y:x for x,y in train_label_dict.items()}

    
    train_int_labels = np.array([train_label_dict_rev[x] for x in train_int_labels])
    
    decision = UakNN_predict(data=data_fp, knn_model=knn_model, UakNN_model=UakNN_params, alpha=alpha,
                         train_labels=train_int_labels, train_label_dict=train_label_dict)
 
    print("decision is", decision)

    mac = bin_fname.split('_')[1]

    if noPlotMode:
        redis_controller(decision, mac)
        return decision
    else:
        save_plot_dir = os.path.join(working_dir, "plots")
        class_size = (len(train_int_labels[train_int_labels == 0]), data_fp.shape[1])
        train_data = (train_data - np.mean(train_data, axis=(1, 2), keepdims=True)) / np.std(train_data, axis=(1, 2),
                                                                                             keepdims=True)
        train_fp = triplet_model.predict(train_data)
        image_path = plot_TSNE(train_data=train_fp, train_labels=train_int_labels, train_label_dict=train_label_dict,
                  data_fp=data_fp, decision=decision, mac=mac, class_size=class_size, save_plot_dir=save_plot_dir)
        
        redis_controller(decision, mac)
        return decision, image_path



PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)
print("project root is", PROJECT_ROOT)

PROJECT_ROOT = os.getcwd()
working_dir = PROJECT_ROOT
model_dir = os.path.join(working_dir, "models")
triplet_model_path = os.path.join(model_dir, "triplet_model")
knn_model_fname = "knn_model"
UakNN_model_fname = "UakNN_model"
data_dir = os.path.join(working_dir, "data")
train_npz_fname = "train.npz"
train_data_path = os.path.join(data_dir, train_npz_fname)

# load data and models
train_npz_file = np.load(train_data_path, allow_pickle=True)
train_data, train_labels = train_npz_file['data'], train_npz_file['labels']
train_label_dict = train_npz_file['label_dict'].item()
train_int_labels = train_labels
train_int_labels = np.array(train_int_labels)

#train_int_labels = LabelEncoder().fit_transform(train_npz_file['labels'])
#train_label_dict = {x:y for x,y in zip(train_int_labels,train_labels)}

triplet_model = keras.models.load_model(triplet_model_path, compile=False)
knn_model = load_knn_models(model_dir=model_dir, model_fname=knn_model_fname)
UakNN_params = load_knn_models(model_dir=model_dir, model_fname=UakNN_model_fname)
alpha = UakNN_params['alpha']



if __name__ == '__main__':
    bin_dir = sys.argv[1]
    print(bin_dir)


    wm = pyinotify.WatchManager()  # Watch Manager
    mask = pyinotify.IN_CREATE  # watched events

    import time 
    class EventHandler(pyinotify.ProcessEvent):
        def process_IN_CREATE(self, event):
            bin_fname = os.path.split((event.pathname))[1]
            print(bin_fname)
            time.sleep(.2)
            get_model_decision(bin_dir, bin_fname, alpha, UakNN_params, knn_model, 
        triplet_model, train_int_labels, train_label_dict, train_data,
        train_labels)

    handler = EventHandler()
    notifier = pyinotify.Notifier(wm, handler)
    wdd = wm.add_watch(bin_dir, mask, rec=True)

    notifier.loop()

    print(rfs)

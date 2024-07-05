import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from scipy.spatial.distance import squareform
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import random
import os


brain_region = 116
eps = 1e-6

# abide
deleted_data_list_abide = [1,2,3,4,5,6,7,8,9,10,12,15,16,17,21,22]

# mdd
deleted_data_list_mdd = [1,8,12,14,17]


class ms_Dataset(Dataset):
    
    def __init__(self, dataset_name):
        self.data_list, self.label_list, self.oneperson_batch_list = self.load_dataset(dataset_name)


    def __getitem__(self, index):
        graph = self.data_list[index]
        label = self.label_list[index]
        oneperson_batch_list = self.oneperson_batch_list[index]

        return graph, label, oneperson_batch_list


    def __len__(self):
        return len(self.label_list)   


    def load_dataset(self, dataset_name):
        print('loading data...')

        if (dataset_name == 'ABIDE'):
            label_filepath = './data/ABIDE_labels.mat'
            label = loadmat(label_filepath)
            labels = label['labels']
            site_filepath = './data/ABIDE_site_information.mat'
            site_information = loadmat(site_filepath)
            site_information = site_information['site_information'][:,0]
            deleted_data_list = deleted_data_list_abide

        if (dataset_name == 'MDD'):
            label_filepath = './data/MDD_labels.mat'
            label = loadmat(label_filepath)
            labels = label['labels']
            site_filepath = './data/MDD_site_information.mat'
            site_information = loadmat(site_filepath)
            site_information = site_information['site']
            deleted_data_list = deleted_data_list_mdd

        if (dataset_name == 'SRPBS'):
            label_filepath = './data/SRPBS_labels.mat'
            label = loadmat(label_filepath)
            labels = label['labels']

        sample_num = len(labels)
        data_list = []
        label_list = []
        oneperson_batch_list = []

        final_num = 0
        for i in range(sample_num):

            flag = 1
            data_filepath = './data/' + str(dataset_name) + '/sample_' + str(i+1) + '.mat'
            print("loading ", data_filepath)
            m1 = loadmat(data_filepath)
            m2 = m1['dataset']
            graph_num = len(m2[0,0])
            node_oneperson = torch.empty(0)
            dense_edge_oneperson = torch.empty(0)

            if (dataset_name == 'MDD' or dataset_name == 'ABIDE'):
                for m in range(len(deleted_data_list)):
                    if deleted_data_list[m] == site_information[i]:
                        flag = 0
                        print('the sample has been discarded')
                        break
                
                if(flag == 0):
                    continue

            k = 0
            for j in range(0, graph_num):
                index = 'sample' + str(i+1) + '_' + str(k+1).rjust(4,'0') + '_' + str(k+50).rjust(4,'0')
                graph = m2[0,0][index]
                node_features = torch.FloatTensor(graph)

                if(torch.isinf(node_features).any() or torch.isnan(node_features).any()):
                    flag = 0
                    print("nan or inf was found in the data, the sample has been discarded")
                    break

                node_oneperson = torch.cat((node_oneperson, node_features), dim = 0)
                tepk = node_features.reshape(-1,1)
                tepk, indices = torch.sort(abs(tepk), dim=0, descending=True)
                mk = tepk[int(node_features.shape[0]**2 * 0.2 - 1)]
                edge = torch.Tensor(np.where(node_features > mk, 1, 0))
                dense_edge = dense_to_sparse(edge)[0]
                dense_edge = dense_edge.add(brain_region * j)
                dense_edge_oneperson = torch.cat((dense_edge_oneperson, dense_edge), dim = 1).long()

                k = k+5

            if(flag):
                data_one_sample = Data(x = node_oneperson, edge_index = dense_edge_oneperson)
                data_list.append(data_one_sample)
                label_list.append(torch.tensor(labels[i]))
                oneperson_batch_list.append(int(node_oneperson.size(0)/brain_region))
                final_num = final_num + 1
        

        print("got", final_num, "samples")
        
        return data_list, label_list, oneperson_batch_list


def separate_data(data, edge, label,  seed, fold_idx, k_fold):
    
    assert 0 <= fold_idx and fold_idx < k_fold, "fold_idx out of index"
    skf = StratifiedKFold(n_splits=k_fold, shuffle = True, random_state = seed)

    idx_list = []
    for idx in skf.split(np.zeros(len(label)), label):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_node_list = [data[i] for i in train_idx]
    train_edge_list = [edge[i] for i in train_idx]
    train_label_list = [label[i] for i in train_idx]

    test_node_list = [data[i] for i in test_idx]
    test_edge_list = [edge[i] for i in test_idx]
    test_label_list = [label[i] for i in test_idx]

    return train_node_list, train_edge_list, train_label_list, test_node_list, test_edge_list, test_label_list


def joint_entropy(x,y,s_x,s_y):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow) + eps)

    return entropy


def calculate_MI(x,y):
    s_x = calculate_sigma(x)
    s_y = calculate_sigma(y)
    Hx = reyi_entropy(x,s_x**2)
    Hy = reyi_entropy(y,s_y**2)
    Hxy = joint_entropy(x,y,s_x**2,s_y**2)
    Ixy = Hx + Hy - Hxy
    
    return Ixy



def calculate_sigma(Z):
    dist_matrix = torch.cdist(Z, Z, p=2)
    sigma = torch.mean(torch.sort(dist_matrix[:, :10], 1)[0])
    if sigma < 0.1:
        sigma = 0.1
    return sigma  


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x,sigma)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow) + eps)
    return entropy


def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    return torch.exp(-dist /sigma)


def pairwise_distances(x):
    #x should be two dimensional
    if x.dim()==1:
        x = x.unsqueeze(1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()


def calculate_entropy(x):
    sigma_x = calculate_sigma(x)
    Hx = reyi_entropy(x, sigma_x**2)

    return Hx


def calculate_bottleneck(x):
    H_sum = 0
    for ib_index in range(x.shape[1]):
        H_sum += calculate_entropy(x[:, ib_index].reshape(-1,1))
    
    out = H_sum - torch.var(x.t())

    return out


def calc_performance_statistics(y_pred, y):
    y_pred = y_pred.cpu()
    y = y.cpu()
    TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    acc = (TN + TP) / N
    sen = TP / (TP + FN)
    spc = TN / (TN + FP)
    prc = TP / (TP + FP)
    f1s = 2 * (prc * sen) / (prc + sen)
    mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))

    return acc, sen, spc, prc, f1s, mcc


def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_value(value, average=True):
    world_size = get_world_size()

    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
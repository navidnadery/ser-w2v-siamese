from random import random, randint, choice
import torch
import numpy as np
import itertools
from tqdm import tqdm
from config import generate_label
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict


def read_EMODB(audio_indexes, is_training, filter_num, timesteps):
    sample_num = 2000
    
    sample_label = torch.empty((sample_num, 1), dtype=torch.int8)
    sample_data = torch.empty((sample_num, timesteps, filter_num), dtype=torch.float32)
    
    snum = 0
    sample_num = 0
    sample_emt = {'F':0, 'W':0, 'N':0, 'T':0 }

    for filename in tqdm(audio_indexes):
        emotion = filename.split('/')[-1][5]
        if(emotion in ['F','W','N','T']):
            featv1 = torch.load(filename)
            if (featv1.shape[1] <= timesteps):
                feat1_ = featv1.clone()
                while featv1.shape[1] < timesteps:
                    featv1 = torch.cat([featv1, feat1_], dim = 1)
            em = generate_label[emotion].value
            sample_label[sample_num] = em
            sample_data[sample_num,:,:] = featv1[0, :timesteps, :]
            sample_emt[emotion] += 1
            sample_num += 1
            snum += 1
    
    sample_label = sample_label[:sample_num]
    sample_data = sample_data[:sample_num, :, :]
    
    if is_training:
        arr = np.arange(sample_num)
        np.random.shuffle(arr)
        sample_data = sample_data[arr]
        sample_label = sample_label[arr]

    return sample_data, sample_label, sample_emt

class data_emo(Dataset):
    def __init__(self, data, label, trdata=None, trlabel=None) -> None:
        super().__init__()
        label_dict = defaultdict(list)
        self.data, self.label = data, label
        if trlabel is not None:
            self.trdata = trdata
            self.trlabel = trlabel
        else:
            self.trdata = None
            self.trlabel = None
        for i, item in enumerate(label):
            item = int(item)
            label_dict[item].append(i)
        self.label_dict = label_dict

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        if self.trlabel is not None:
            return self.data[index[0]], self.trdata[index[1]], int(self.label[index[0]]==self.trlabel[index[1]])
        return self.data[index[0]], self.data[index[1]], int(self.label[index[1]] == self.label[index[0]])


class balanced_sampler(Sampler):
    def __init__(self, data_source):

        self.label_dict         = data_source.label_dict
        
    def __iter__(self):
        
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()

        iter_list = []

        ## Data for each class
        length = lambda values : len(values)
        max_idx = max(length(value) for value in self.label_dict.values())
        
        for i in range(max_idx):
            pIdx = []
            for key in dictkeys:
                pIdx0 = [self.label_dict[key][i%len(self.label_dict)]]
                rnd = random()
                if rnd < 0.4:
                    pIdx0 += [self.label_dict[key][(i+randint(1, len(self.label_dict)))%len(self.label_dict)]]
                else:
                    pIdx0 += [self.label_dict[choice([i for i in range(0,len(dictkeys)) if i not in [key]])][(i+randint(1, len(self.label_dict)))%len(self.label_dict)]]
                pIdx.append(pIdx0)
            iter_list.append(np.random.permutation(pIdx))
        
        return iter(itertools.chain.from_iterable([iter for iter in iter_list]))
    
    def __len__(self):
        return len(self.label_dict)


class eval_sampler(Sampler):
    def __init__(self, data_source, trdata_source):
        self.label_dict         = data_source.label_dict
        self.train_data         = trdata_source.label_dict
        
    def __iter__(self):
        
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()

        iter_list = []

        for key in dictkeys:
            for idx in self.label_dict[key]:
                for trk in self.train_data.values():
                    iter_list.append([idx, np.random.choice(trk)])
        
        return iter([iter for iter in iter_list])
    
    def __len__(self):
        return len(self.label_dict)
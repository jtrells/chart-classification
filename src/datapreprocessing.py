# TODO: Pass the classes as a parameter

from __future__ import print_function
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

import torch
import random
import pandas as pd
import skimage
import os
import numpy as np


class ChartsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_fpath, set_keys, transform=None):        
        self.root_dir = root_dir
        self.transform = transform        
        
        # from the CSV, select the subset for train or val
        df_all = pd.read_csv(csv_fpath)
        self.dataframe = df_all[df_all['id'].isin(set_keys)]
        self.dataframe = self.dataframe[:1000]
        
        # one hot encoder for the chart type
        self.codec = LabelEncoder()
        unique_labels = list(self.dataframe['class'].unique())
        self.codec.fit(unique_labels)        
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.read_image(idx)
        #one_hot_chart_type = self.to_one_hot(self.codec, self.dataframe.iloc[idx, 1])                
        
        if self.transform:
            image = self.transform(image)
        
        label = self.codec.transform([self.dataframe.iloc[idx, 1]])
        return (image, label[0])
    
    def read_image(self, idx):
        img_id = self.dataframe.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, str(self.dataframe.iloc[idx, 0]) + '.png')
        # CHART synthetic images have 4 dimensions, get rid of the last one
        image = skimage.io.imread(img_name)[:,:,:3]
        return image
    
    def to_one_hot(self, codec, val):
        idxs = codec.transform([val])
        #return torch.eye(len(codec.classes_))[idxs][0]        
        return torch.tensor([idxs][0][0])


class DataProprocessing():
    def __init__(self, train_img_dir, test_img_dir, csv_path, seed=443, val_size=0.25):
        # train hosts training and validation images
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.csv_path = csv_path        
        self.train_keys, self.val_keys = self._create_validation_set(val_size)
        
    def _create_validation_set(self, val_size, seed=443):
        ''' get the keys of items that belong to the training and validation 
            sets in a stratified manner '''
        random.seed(seed)
        np.random.seed(seed)
        
        data_df = pd.read_csv(self.csv_path)
        labels_dict = data_df.set_index('id').T.to_dict('list')
        
        X = list(labels_dict.keys())
        y = [labels_dict[x][0] for x in X]
        X = np.array(X)        
        train_keys, val_keys, _, _ = train_test_split(X, y, stratify=y, test_size=val_size)
        
        return train_keys, val_keys
    
    def get_train_dataset(self, normalized=True):        
        return self._get_dataset(self.train_keys, normalized=normalized)
    
    def get_val_dataset(self, normalized=True):        
        return self._get_dataset(self.val_keys, normalized=normalized)

    def _get_dataset(self, keys, normalized=True):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        
        if normalized:
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            
        transform = transforms.Compose(transform_list)
        return ChartsDataset(self.train_img_dir, self.csv_path, keys, transform=transform)

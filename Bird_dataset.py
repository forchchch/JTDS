import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pickle
from torchvision import transforms
import torchvision.datasets as dataset

class Bird_dataset_naive(Dataset):
    def __init__(self, image_path_file, image_label_path, image_root,transform= None, finegrain = False, corrupted = False, num_class=200, cor_rate = 0.0):
        
        self.image_file = json.load( open(image_path_file,'r') )
        self.name_to_id = {}
        n = 0
        for name in self.image_file:
            self.name_to_id[name] = n
            n = n+1
        self.image_label = json.load( open(image_label_path, 'r') )
        self.transform = transform
        self.image_root =  image_root
        self.need_fine_label = finegrain
        self.num_classes = num_class
        self.corruption_ratio = cor_rate
        self.c_matrix = self.bulid_corrupted_matrix()
        self.corrupted = corrupted
        self.used_image_labels = {}
        for key in self.image_label.keys():
            self.used_image_labels[key] = self.image_label[key][1] 
            if self.corrupted:
                self.used_image_labels[key] = np.random.choice(self.num_classes, p=self.c_matrix[self.image_label[key][1]])
        print("self.use_aux:",finegrain)

    def __getitem__(self,index):
        image_id = self.image_file[index]
        image_path = os.path.join( self.image_root, self.image_label[image_id][0] )
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        class_label = self.used_image_labels[image_id]
        if self.need_fine_label:
            fine_label = self.image_label[image_id][2]
            return image, class_label, fine_label, self.name_to_id[image_id]
        else:
            return image, class_label,1
    
    def __len__(self):
        return len(self.image_file)
    
    def bulid_corrupted_matrix(self):
        eye = np.eye(self.num_classes)
        noise = np.full((self.num_classes, self.num_classes), 1/self.num_classes)
        corruption_matrix = eye * (1 - self.corruption_ratio) + noise * self.corruption_ratio
        return corruption_matrix


class Bird_dataset(Dataset):
    def __init__(self, image_path_file, image_label_path, image_root,transform= None, finegrain = False, corrupted = False, num_class=200, cor_rate = 0.0, aux_file=None):
        
        self.image_file = json.load( open(image_path_file,'r') )
        self.image_label = json.load( open(image_label_path, 'r') )
        self.corrupted = corrupted
        if self.corrupted:
            self.auxset = json.load(open(aux_file, 'r'))
        self.transform = transform
        self.image_root =  image_root
        self.need_fine_label = finegrain
        self.num_classes = num_class
        self.corruption_ratio = cor_rate
        self.c_matrix = self.bulid_corrupted_matrix()
        self.used_image_labels = {}
        for key in self.image_file:
            self.used_image_labels[key] = self.image_label[key][1]
            if self.corrupted:
                if key not in self.auxset:
                    self.used_image_labels[key] = np.random.choice(self.num_classes, p=self.c_matrix[self.image_label[key][1]])

    def __getitem__(self,index):
        image_id = self.image_file[index]
        image_path = os.path.join( self.image_root, self.image_label[image_id][0] )
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        class_label = self.used_image_labels[image_id]
        if self.need_fine_label:
            fine_label = self.image_label[image_id][2]
            return image, class_label, fine_label
        else:
            return image, class_label,1
    
    def __len__(self):
        return len(self.image_file)
    
    def bulid_corrupted_matrix(self):
        eye = np.eye(self.num_classes)
        noise = np.full((self.num_classes, self.num_classes), 1/self.num_classes)
        corruption_matrix = eye * (1 - self.corruption_ratio) + noise * self.corruption_ratio
        return corruption_matrix




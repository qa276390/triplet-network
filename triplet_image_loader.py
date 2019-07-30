from PIL import Image
import os
import os.path
import pandas as pd
import json

import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

def default_image_loader(path):
    return Image.open(path).convert('RGB')



"""
filenames_filename: path to "index.csv"
triplets_file_name: train.json (set)
* emb_path: path to "embedding.pt"
* emb_tensor: tensor from "embedding.pt"
* outfit_data: set info in json format
"""

class TripletEmbedLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, triplets_file_name, mode, emb_file, transform=None):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.is_train = mode == 'train' 
        self.base_path = base_path  
        ######################################
        self.emb_path = os.path.join(self.base_path, emb_file)
        self.emb_tensor = torch.load(self.emb_path)
        print(type(self.emb_tensor[0][-1]))
        data_json = os.path.join(base_path, triplets_file_name) 
        outfit_data = json.load(open(data_json, 'r'))
        ######################################
        
        self.index_path = os.path.join(self.base_path, filenames_filename)
        self.indexlist = pd.read_csv(self.index_path)
        #for line in open(filenames_filename):
        #    self.filenamelist.append(line.rstrip('\n'))
        
        self.transform = transform
        #self.loader = loader
        ###################################### ######################################
        im2type = {}
        category2ims = {}
        imnames = set()
        id2im = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']
                #print(im)
                #print(self.indexlist.image==im)
                category = self.indexlist[self.indexlist.image==int(im)].type.values[0]
                im2type[im] = category

                if category not in category2ims:
                    category2ims[category] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []

                category2ims[category][outfit_id].append(im)
                id2im['%s_%i' % (outfit_id, int(item['index']))] = im
                imnames.add(im)
        self.category2ims = category2ims
        pos_pairs = []
        max_items = 0
        for outfit in outfit_data:
            items = outfit['items']
            cnt = len(items)
            max_items = max(cnt, max_items)
            outfit_id = outfit['set_id']
            for j in range(cnt-1):
                for k in range(j+1, cnt):
                    pos_pairs.append([outfit_id, items[j]['item_id'], items[k]['item_id']])

        self.pos_pairs = pos_pairs
        ###################################### ######################################

    def sample_negative(self, outfit_id, item_id, item_type):
        """ Returns a randomly sampled item from a different set
            than the outfit at data_index, but of the same type as
            item_type
        
            data_index: index in self.data where the positive pair
                        of items was pulled from
            item_type: the coarse type of the item that the item
                       that was paired with the anchor
        """
        item_out = item_id
        candidate_sets = self.category2ims[item_type].keys()
        attempts = 0
        while item_out == item_id and attempts < 100:
            choice = np.random.choice(list(candidate_sets))
            items = self.category2ims[item_type][choice]
            item_index = np.random.choice(range(len(items)))
            item_out = items[item_index]
            attempts += 1
                
        return item_out
    

    def __getitem__(self, index):
        
        #if self.transform is not None:
        #    img1 = self.transform(img1)
        #    img2 = self.transform(img2)
        #    img3 = self.transform(img3)

        if self.is_train:
            outfit_id, anchor_im, pos_im = self.pos_pairs[index]
            img1, anchor_type = self.femb_loader(anchor_im)
            img2, item_type = self.femb_loader(pos_im)
            
            neg_im_id = self.sample_negative(outfit_id, pos_im, item_type)
            img3, ntype = self.femb_loader(neg_im_id)
        
            return img1, img2, img3

        anchor = self.imnames[index]
        img1, anchor_type = self.femb_loader(anchor)
            
        return img1

    def __len__(self):
        if self.is_train:
            return len(self.pos_pairs)
        return len(self.imnames)

    def emb_loader(self, index):
        return self.emb_tensor[index][-1]

    def femb_loader(self, imageid):
        imageid = int(imageid)
        return self.emb_tensor[(self.indexlist[self.indexlist.image==imageid].index.values[0])][-1], self.indexlist[self.indexlist.image==imageid].type.values[0]



########################################## below are unuse ###############################################
class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, triplets_file_name, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.base_path = base_path  
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        for line in open(triplets_file_name):
            triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close
        self.triplets = triplets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path1)]))
        img2 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path2)]))
        img3 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path3)]))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)

import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#import h5py

# open questions
# how does index work?
# load with every call of get item only one chunk of one csv file?
# or
# load all time windows of all experiments at once? 
# how to imagine tensor? Array with matrix of cp data in every entry?

class TimeSeriesDataset(Dataset):
    """  TimeSeries dataset"""

    def __init__(self, folder_path: str,  experiments: list, normalize_mvts=False):
        """
        Args:
            folder_path: folder where the data is stored
            normalize_mvts: if the time-series should be normalized
        """
        self.__file_path = folder_path
        self.__normalize = normalize_mvts
        self.__exps = experiments
        self.__dataset_length = len(experiments)*5

    def __len__(self):
        return self.__dataset_length


    def __getitem__(self, idx):
        mvts = torch.tensor([]) # should have a shape of [1, m, n]
        label = torch.tensor([]) # should have a shape of [1, 1]

        # calculate from index the corresponding experiment and the starting point:
        # per experiment five starting points
        # batch size: 5*#experiments

        startingPointIdx = idx%5
        experimentIndex = int((idx-startingPointIdx)/5)
      
        self.mvts_length = 2000
        
        # only get data set corresponding to index

        exp = self.__exps[experimentIndex]
               
        # get correct file name
               
        if exp < 10:
            filename = '/Exp_00'+str(exp)+'_aerosense/1_baros_p.csv'
        elif exp > 9 and exp < 100:
            filename = '/Exp_0'+str(exp)+'_aerosense/1_baros_p.csv'
        else: 
            filename = '/Exp_'+str(exp)+'_aerosense/1_baros_p.csv'

        filepath = self.__file_path+filename

        # discard measurements of sensor 23 (has faulty data)                                     
        temp = pd.read_csv( filepath, delimiter=',', header = None, skiprows = 2500, usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,31,32,33,34,35,36,37,38])
        [rows, columns] = temp.shape
        startingPoints = np.linspace(0,rows-self.mvts_length,5)
        startingPoints_rounded = np.rint(startingPoints)
        
        start = startingPoints_rounded[startingPointIdx]
        start_int = int(start)
        end_int = int(start+self.mvts_length)
        mvts_chunk = temp.iloc[start_int:end_int,:]
        mvts_new = mvts_chunk.transpose()
        #mvts_new = mvts_new.astype()

        # transform shape of mvts_chunk: m = number of channels n = measurement values
                            
        # conversion to tensor correct? 
        mvts = torch.tensor(mvts_new.values) 
        #mvts = mvts.unsqueeze(0)     
        #mvts_chun_tensor = torch.as_tensor(mvts_chunk)
        #mvts.add(mvts_chun_tensor)

        #unsqueeze(0)

        # obtain label

        if exp < 20:                    # healthy probes
            label = 0.0
        elif exp > 19 and exp < 39:     # added mass
            label = 5.0
        elif exp > 38 and exp < 58:     # 5 mm crack
            label = 1.0
        elif exp > 57  and exp < 77:    # 10 mm crack
            label = 2.0
        elif exp > 76  and exp < 96:    # 15 mm crack
            label = 3.0
        else: 
            label = 4.0                 # 20 mm crack

        #label = label.double()
        # folder path
             
        # per channel normalization
        if self.__normalize:
            mean = mvts.mean(dim=1).unsqueeze(1)
            std = mvts.std(dim=1).unsqueeze(1)
            mvts = (mvts - mean) / std

        # conversion to tensor correct?    

        label = torch.tensor(label, dtype=torch.long).double()
        #label = label.unsqueeze(0)
        #label = label.unsqueeze(1)
        return mvts, label


### Example dataset below

class NHCPPDataset(Dataset):
    """  NHCPP dataset"""

    def __init__(self, file_path: str, normalize_mvts=False):
        """
        Args:
            file_path: path to the hdf5 data file
            normalize_mvts: bool indicating if time-series should be normalized
        """
        self.__file_path = file_path
        self.__normalize = normalize_mvts

        with h5py.File(file_path, 'r') as h5_file:
            attrs = dict(h5_file.attrs.items())
        self.__dataset_length = attrs['dataset_len']
        self.num_channels = attrs['num_channels']
        self.channels = attrs['channels']
        self.num_classes = attrs['num_classes']
        self.mvts_length = 100

    def __len__(self):
        return self.__dataset_length

    def __getitem__(self, idx):
        # read the hdf5 file and select the data to load in by index
        with h5py.File(self.__file_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            dset = h5_file[keys[idx]]
            mvts = dset[()]
            label = dset.attrs['degradation_state'].tolist()

        for key in dict:
            if label == key:
                    label = dict[key]

        mvts = torch.tensor(mvts).squeeze().float()[:self.mvts_length,:].permute(1,0)

        # per channel normalization
        if self.__normalize:
            mean = mvts.mean(dim=1).unsqueeze(1)
            std =  mvts.std(dim=1).unsqueeze(1)
            mvts = (mvts - mean) / std

        label = torch.tensor(label, dtype=torch.long)
        return mvts, label
    
class FeatureDataset(Dataset):

    def __init__(self, folder):
        self.files = os.listdir(folder)
        self.folder = folder

    def __len__(self):
       return len(self.files)
    
    # maybe change to give back mvts as well as label
    def __getitem__(self, idx):
       return torch.load(f"{self.folder}/{self.files[idx]}")
    



    # if __name__ == '__main__':
    #     train_exp = [4,5,8,9]
    #     test_exp = [6,10]   


    #     dataset = TimeSeriesDataset('C:/Users/phfra/Desktop/mvts_analysis/AoA_0deg_Cp', train_exp)
    #     #print(len(dataset))
    #     #print(dataset[0])
    #     #print(dataset[3])
    #     #print(dataset[15])
    #     #print(dataset.num_node_output_features)
    #     #print(dataset.num_edge_features)
    #     #print(dataset.num_glob_features)
    #     #print(dataset[0].node_feat_labels)
    #     #print(dataset[0].globals)

    #     loader = DataLoader(dataset, shuffle=True, batch_size=2)
    #     for i in range(2):
    #         testset1 = (next(iter(loader)))
    #         testset1[1].shape[1]
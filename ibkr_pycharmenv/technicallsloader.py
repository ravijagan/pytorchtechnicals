import torch
import torchvision
import numpy as np
import math
import sys
sys.path.extend(["C:\\Users\\ravi\\PycharmProjects\\ibkr-jun2021"])
import dbutils
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''


# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__
from torch.utils.data import Dataset, DataLoader


class TechDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor( x_data, device = 'cuda')# tmpdf[:, 1:])  # size [n_samples, n_features]
        #self.y_data_raw = torch.tensor(tmpdf[ :, -1] , device = 'cuda' )#tmpdf[:, [0]])  # size [n_samples, 1]
        self.y_data = torch.tensor(y_data, device= 'cuda')#.type(torch.IntTensor)
        self.y_data = torch.clip(self.y_data, 0., 1.0)# 1e-9, 1-1e-9) #doc says less than 1 float uncomment if problem
        #print("x shape",self.x_data.shape , self.y_data.shape)
        self.n_samples = x_data.shape[0]
        self.n_features = x_data.shape[1]
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        # tbd index = index % self.n_samples
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


def getdata():
# Initialize data, download, etc.
        # read with numpy or pandas
        q = """
            select 
                --timestampx , sma_2 , close, ema_2, macd_2, -- *, 
                timestampx, sma_2, ema_2, macd_2,
            (lead(close, 1) over (order by timestampx)  - close) as  gain
             from technicals  as AA
         """
        # get it as numpy becauase df to numpy will require more steps and nan can be stripped
        #['timestampx' , 'hour_min' , 'close', 'ema_10', 'macd_10']
        tmpdf = dbutils.get_all_data(query=q, retdf=False, columns=None,
                                     tablename=None, stripnan=True, if_exists='replace')
        # here the first column is the class label, the rest are the features
        x_data = tmpdf[ :, 1:-1]  # last column is Y first is timestamp
        y_data = tmpdf[ :, -1]
        y_data = np.clip(y_data, -5., 5.)
        thresh = 1.
        # convert to boolean for logistic or similar application 
        #bool_thresh_array = y_data > thresh # used for logistic type 
        #y_data = bool_thresh_array
        #
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1234)
        #sc = StandardScaler()
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        """
        scy = MinMaxScaler()
        y_train = y_train.reshape(-1, 1)
        y_train = scy.fit_transform(y_train)
        y_test = y_test.reshape(-1, 1)
        y_test = scy.transform(y_test)
        """

        train_dataset = TechDataset(X_test, y_test)
        return train_dataset


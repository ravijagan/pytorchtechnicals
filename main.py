# This is a sample Python script.
import torch
import sys
import traceback
sys.path.extend(["C:\\Users\\ravi\\PycharmProjects\\ibkr-jun2021"])
import dbutils
from technicallsloader import *
from torch import nn
from samp_models import *

from torchvision.ops import sigmoid_focal_loss
print("cuda available", torch.cuda.is_available())
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

device= "cpu" # "cuda"
# 1) Model
# Linear model f = wx + b , sigmoid at the end

# create dataset
train_dataset, test_dataset = getdata()

neural_model = NeuralNet(input_size = train_dataset.n_features, hidden_size = 4,
                         op_size=1).to(device)
model = logisticModel(train_dataset.n_features).to(device=device)

model = neural_model
model = AutoEncoder(train_dataset.n_features, 4)# train_dataset.n_features/2)
# Training loop

batch_size = 16
total_samples = train_dataset.n_samples
n_iterations = math.ceil(total_samples / batch_size)
print(total_samples, n_iterations)
num_epochs = 30
learning_rate = 1e-5
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
for epoch in range(num_epochs):
    learning_rate *= 0.9
    # Load whole dataset with DataLoader
    # shuffle: shuffle data, good for training
    # num_workers: faster loading with multiple subprocesses
    # !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle= False, #True,
                              num_workers=0)
    # convert to an iterator and look at one random sample
    dataiter_train = iter(train_loader)
    for i in range(n_iterations -1 ): # range(num_epochs):
        traindata = next(dataiter_train)
        X_train, y_train = traindata
        y_train = y_train.to(torch.float32)
        X_train=X_train.to(torch.float32) # TBD hack understand this cuda forces
        y_pred = model(X_train)
        # normalize this or scale this ?
        #y_train to be weighten higher for 1 , less for 0
        #pos_samp = y_train > 2
        #wts = (pos_samp+ 0.001) * 2
        #hack_y_train = y_train + wts
        #y_train = hack_y_train
        #criterion = nn.BCELoss(weight=wts)  # targets should be 0 and 1
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss(reduce='mean')

        #for encoded decoder we want output to look like the X-train
        loss = criterion(y_pred, X_train)
        #loss = sigmoid_focal_loss (y_pred, y_train, alpha =-0.25, gamma = 2 , reduction='mean')
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print(epoch, i, "loss", loss)

        # debug printing to see gradients not 0 or infinity
        if i< 3 or i% 2000==0:
            print(epoch, i, "loss", loss)
            #print("\n \t predicted", y_pred.flatten(), "\n \t actual" , y_train.flatten())
            #print( torch.argsort(y_pred.flatten()) , "\nactual", torch.argsort(y_train.flatten()))
            print(i, "---gradients---")
            for p in model.parameters():
                try:
                    print( "\t grad", p.grad)
                except:
                    traceback.print_exc()
                    pass
            print("-----\n")
        optimizer.zero_grad(set_to_none=True)

test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle= False, #True,
                              num_workers=0)
dataiter_test = iter(test_loader)
# TBD set model to no train
model.eval()
for i in range(10):
    testdata = next(dataiter_test)
    X_test, y_test = testdata
    y_test = y_test.to(torch.float32)
    X_test = X_test.to(torch.float32)  # TBD hack understand this cuda forces
    y_pred = model(X_test)  # .flatten()
    # print if anomaly or not, and
    # what is the distance between pred an train ideally should be zero unless anomaly
    for t in range(10):
        print(i,  y_test[t], criterion(y_pred[t], X_train[t])) #"pred" , y_pred,  "xtest", X_test, " \n ytest", y_test)


"""

WITH ABS(DIFF)
for logistic the gradients not zero but the model games even focal loss 
by putting all the predicted to the same value . error is like 0.01 
without scaler every gradient every loop was zero 
Adam seems to do something other than making all the same but still far from 1,0

strange cuda errors running ibkr and this project in parallel  in debugger 
standard scalaer gives grads 1e-4 but minmax gives 0 

#loss = sigmoid_focal_loss (y_pred, y_train, alpha =-0.25, gamma = 2 , reduction='mean') goves grads zero 
high weighted bce 4 also goes to 0 
mse wt4 is ok at 8 zero gradient 

WITHOUT ABS(DIFF)
same the wt cant take 4 

"""
# This is a sample Python script.
import torch
import sys
import traceback
sys.path.extend(["C:\\Users\\ravi\\PycharmProjects\\ibkr-jun2021"])
import dbutils
from technicallsloader import *
from torch import nn
from torchvision.ops import sigmoid_focal_loss
print(torch.cuda.is_available())
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

device="cuda"
# 1) Model
# Linear model f = wx + b , sigmoid at the end
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.siz2  = int(hidden_size/2)
        self.l2 = nn.Linear(hidden_size, self.siz2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(self.siz2, num_classes)


    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        out = torch.clamp(out, 0.0, 1.0)
        return out


class logisticModel(nn.Module):
    def __init__(self, n_input_features):
        super(logisticModel, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# create dataset
dataset = getdata()

neural_model = NeuralNet(input_size = dataset.n_features, hidden_size = 4,
                         num_classes=1).to(device)
model = logisticModel(dataset.n_features).to(device='cuda')

model = neural_model
# Training loop
num_epochs = 2
batch_size = 16
total_samples = dataset.n_samples
n_iterations = math.ceil(total_samples / batch_size)
print(total_samples, n_iterations)
num_epochs = 100
learning_rate = 1e-3

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
for epoch in range(0,10):
    learning_rate *= 0.98
    # Load whole dataset with DataLoader
    # shuffle: shuffle data, good for training
    # num_workers: faster loading with multiple subprocesses
    # !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle= False, #True,
                              num_workers=0)
    # convert to an iterator and look at one random sample
    dataiter = iter(train_loader)
    for i in range(n_iterations -1 ): # range(num_epochs):
        learning_rate *= 0.99
        traindata = next(dataiter)
        X_train, y_train = traindata
        X_train=X_train.to(torch.float32) # TBD hack understand this cuda forces
        y_pred = model(X_train).flatten()

        # normalize this or scale this ?
        #y_train to be weighten higher for 1 , less for 0
        pos_samp = y_train > 0
        wts = (pos_samp+ 0.001) * 2
        criterion = nn.BCELoss(weight=wts)  # targets should be 0 and 1
        loss = criterion(y_pred, y_train)
        #loss = sigmoid_focal_loss (y_pred, y_train, alpha =-0.25, gamma = 2 , reduction='mean')
        loss.backward()
        optimizer.step()
        if i< 10 or i%50==0:
            print( i, "loss", loss)
            print("\n \t predicted", y_pred, "\n \t actual" , y_train, y_train.sum())
            print("---gradients---")
            for p in model.parameters():
                try:
                    print(p)
                    print( "\t grad", p.grad)
                except:
                    traceback.print_exc()
                    pass
            print("-----\n")
        optimizer.zero_grad(set_to_none=True)

    print(f""" EPOCH {epoch}, {loss}  learning_rate Epoch: {epoch + 1}/{num_epochs}, Step {epoch + 1}/{n_iterations}""")


"""
for logistic the gradients not zero but the model games even focal loss 
by putting all the predicted to the same value . error is like 0.01 
without scaler every gradient every loop was zero 
Adam seems to do something other than making all the same but still far from 1,0

strange cuda errors running ibkr and this project in parallel  in debugger 
standard scalaer gives grads 1e-4 but minmax gives 0 

#loss = sigmoid_focal_loss (y_pred, y_train, alpha =-0.25, gamma = 2 , reduction='mean') goves grads zero 
high weighted bce 4 also goes to 0 
"""
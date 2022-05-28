import torch
import torch.nn as nn

class NVIDIA_ORIGIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 24,
            kernel_size = (5,5), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels = 36,
            kernel_size = (5,5), stride = (2,2))
        self.conv3 = nn.Conv2d(in_channels = 36, out_channels = 48,
            kernel_size = (5,5), stride = (2,2))
        self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.fc1 = nn.Linear(1152,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
        self.fc4 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.conv3(X)
        X = self.relu(X)
        X = self.conv4(X)
        X = self.relu(X)
        X = self.conv5(X)
        X = self.relu(X)
        X = torch.flatten(X, start_dim = 1, end_dim = -1)
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.relu(X)
        X = self.fc3(X)
        X = self.relu(X)
        X = self.fc4(X)
        return X
class NVIDIA_ORIGIN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 24,
            kernel_size = (5,5), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels = 36,
            kernel_size = (5,5), stride = (2,2))
        self.conv3 = nn.Conv2d(in_channels = 36, out_channels = 48,
            kernel_size = (5,5), stride = (2,2))
        self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.fc1 = nn.Linear(1152,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
        self.fc4 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.conv3(X)
        X = self.relu(X)
        X = self.conv4(X)
        X = self.relu(X)
        X = self.conv5(X)
        X = self.relu(X)
        X = torch.flatten(X, start_dim = 1, end_dim = -1)
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.relu(X)
        X = self.fc3(X)
        X = self.relu(X)
        X = self.fc4(X)
        self.tanh = nn.Tanh()
        return X

class NVIDIA_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 24,
            kernel_size = (5,5), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels = 36,
            kernel_size = (5,5), stride = (2,2))
        self.conv3 = nn.Conv2d(in_channels = 36, out_channels = 48,
            kernel_size = (5,5), stride = (2,2))
        self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.fc1 = nn.Linear(1152,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
        self.fc4 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.relu(X)
        
        X = self.conv3(X)
        X = self.relu(X)
        
        X = self.dropout(X)
        
        X = self.conv4(X)
        X = self.relu(X)
        
        X = self.conv5(X)
        X = self.relu(X)
        X = torch.flatten(X, start_dim = 1, end_dim = -1)
        X = self.fc1(X)
        X = self.relu(X)
        
        X = self.dropout(X)
        
        X = self.fc2(X)
        X = self.relu(X)
        
        X = self.dropout(X)
        
        X = self.fc3(X)
        X = self.relu(X)
        
        X = self.dropout(X)
        
        X = self.fc4(X)
        X = self.tanh(X)
        return X

class NVIDA_V2(nn.Module):
    def __init__(self):
        super().__init__() #3@66*200
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 24,
            kernel_size = (5,5), stride = (1,1))
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels = 36,
            kernel_size = (5,5), stride = (1,1))
        self.conv3 = nn.Conv2d(in_channels = 36, out_channels = 48,
            kernel_size = (5,5), stride = (1,1))
        self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64,
            kernel_size = (3,3), stride = (1,1))
        self.fc1 = nn.Linear(1152,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,10)
        self.fc4 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
    def forward(self, X):
        X = self.conv1(X)
        X = nn.MaxPool2d(kernel_size=(2,2),padding = 0)(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = nn.MaxPool2d(kernel_size=(2,2),padding = (1,0))(X)
        X = self.relu(X)
        
        X = self.conv3(X)
        X = nn.MaxPool2d(kernel_size=(2,2),padding = (0,1))(X)
        X = self.relu(X)
        
        X = self.dropout(X)        
        X = self.conv4(X)
        X = self.relu(X)
        X = self.conv5(X)
        X = self.relu(X)
        X = torch.flatten(X, start_dim = 1, end_dim = -1)
        
        X = self.fc1(X)
        X = self.relu(X)
        
        X = self.dropout(X)
        
        X = self.fc2(X)
        X = self.relu(X)
        
        X = self.dropout(X)
        
        X = self.fc3(X)
        X = self.relu(X)
        
        X = self.dropout(X)
        
        X = self.fc4(X)
        X = self.tanh(X)
        return X

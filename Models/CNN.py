import torch.nn as nn
from Utils.models import pool_block

#Creo la clase del modelo 
class CNN(nn.Module):
    def __init__(self, num_classes, activation=None):
        super(CNN, self).__init__()
        
        self.block1 = pool_block(3, 16, activation = activation) 
        self.block2 = pool_block(16, 32, activation = activation) 
        self.block3 = pool_block(32, 64, activation = activation) 
        self.block4 = pool_block(64, 128, activation = activation) 
        self.block5 = pool_block(128, 256, activation = activation) 
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)

        self.fc2 = nn.Linear(512, num_classes)
        
        self.activation = activation

        self.bn = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p = 0.4)

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        # Pasar por las capas convolucionales + activación ReLU + pooling + batch normalization
        x = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
        
        # Aplanar las características para pasar a la capa totalmente conectada
        x = self.bn(self.dropout(self.activation(self.fc1(self.flatten(x)))))
        x = self.fc2(x) #no hace falta definir Softmax por que ya se incluye en CrossEntropy los logaritmos d las probabilidades.
        
        return x
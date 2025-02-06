# Funciones utiles en la construcion de los modelos

import torch
import torch.nn as nn
import torch.nn.functional as F 

# Función de activación personalizada Mish.
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    

#Bloque residual.
# Se espera de activation que sea un nn.Module. (nn.ReLU(), nn.LeakyReLU(0.2)).
# En caso de ser Mish por ser personalizada solo con poner Mish() valdría.   
import torch
import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, change_size=True, activation=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.change_size = change_size
        self.activation = activation

        # Ajustar la conexión residual para igualar dimensiones si cambia el tamaño
        if change_size:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x) if self.change_size else x

        y = self.activation(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        # Asegurar que las dimensiones coinciden
        if y.shape != identity.shape:
            print(f"Error de tamaño: y={y.shape}, identity={identity.shape}")
            raise RuntimeError(f"Tamaño no coincide: y={y.shape}, identity={identity.shape}")

        y += identity
        return self.activation(y)


    
#Bloque de n bloques residuales característico de Resnet.
class n_block(nn.Module):
    def __init__(self, n, in_channel, out_channel, stride=1, change_size=True, activation=None):
        super().__init__()
        # Crear el bloque de n bloques residuales
        self.block = self.create_block(n, in_channel, out_channel, stride, change_size, activation)

    def create_block(self, n, in_channel, out_channel, stride, change_size=True, activation=None):
        # El primer bloque tiene los parámetros completos
        block = [residual_block(in_channel, out_channel, stride, change_size=change_size, activation=activation)]
        
        # Los bloques siguientes son bloques residuales con stride=1 y sin cambiar el tamaño
        for i in range(n - 1):
            block.append(residual_block(out_channel, out_channel, stride=1, change_size=False, activation=activation))
        
        # Devolver un secuencial con los bloques
        return nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
    

#Bloque de pooling
class pool_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=3, pool_stride=2, activation=None):
        super().__init__()
        
        # Definir la capa de convolución
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # Capa de normalización por lotes
        
        # Definir la capa de pooling (MaxPooling por defecto)
        #self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=padding)
        
        # Activación (ReLU por defecto si no se especifica)
        self.activation = activation
    
    def forward(self, x):
        # Aplicar la convolución, normalización y activación
        x = self.activation(self.bn(self.conv(x)))
        
        # Aplicar el pooling
        x = self.pool(x)
        
        return x

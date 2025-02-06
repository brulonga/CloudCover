import torch
import ptflops
import os
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Adadelta, Adagrad, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from Models.CNN import CNN
from Models.ResNet36 import ResNet36
from Models.ResNet45 import ResNet45
from Utils.models import Mish
from Utils.findLR import find_lr
from Utils.plot import plot_LR_accuracy, plot_LR_losses

def create_model(opt, rank, world_size):
    torch.cuda.set_device(rank)  # Asignar la GPU correspondiente
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    model_name = opt['network']['name'] 

    # Seleccionar el modelo basado en el nombre
    if model_name == 'CNN':
        activation = nn.Mish() if opt['network']['activation'] == 'Mish' else nn.ReLU()
        model = CNN(num_classes=opt['network']['num_classes'], activation=activation)

    elif model_name == 'ResNet36':
        activation = nn.Mish() if opt['network']['activation'] == 'Mish' else nn.ReLU()
        model = ResNet36(n=opt['network']['n'], num_classes=opt['network']['num_classes'], activation=activation)

    elif model_name == 'ResNet45':
        activation = nn.Mish() if opt['network']['activation'] == 'Mish' else nn.ReLU()
        model = ResNet45(n=opt['network']['n'], num_classes=opt['network']['num_classes'], activation=activation)

    else:
        raise NotImplementedError(f'La red {model_name} no está implementada')

    # Estimación de la complejidad y el número de operaciones
    if rank == 0:
        print(f'Usando la red {model_name}')
        input_size = tuple(opt['datasets']['input_size'])
        flops, params = ptflops.get_model_complexity_info(model, input_size, print_per_layer_stat=False)
        print(f'Complejidad computacional con entrada de tamaño {input_size}: {flops}')
        print('Número de parámetros: ', params)    
    else:
        flops, params = None, None

    model.to(device)

    # Si estamos en un entorno distribuido, envolver el modelo en DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    return model, flops, params

def create_optimizer_scheduler(opt, model, loader, rank, world_size):
    optname = opt['train']['optimizer']
    scheduler = opt['train']['lr_scheduler']

    # Crear el optimizador
    if optname == 'Adam':
        optimizer = Adam(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
        print('Optimizer Adam')
    elif optname == 'SGD':
        optimizer = SGD(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
    elif optname == 'Adadelta':
        optimizer = Adadelta(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
    elif optname == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
    else:
        optimizer = Adam(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
        print(f"Advertencia: Optimizer {optname} no reconocido. Usando Adam por defecto.")

    # Crear el scheduler
    if scheduler == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt['train']['epochs'], eta_min=opt['train']['eta_min'])
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, eta_min=opt['train']['eta_min'])
    elif scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=opt['train']['max_lr'], steps_per_epoch=len(loader), epochs=opt['train']['epochs'], pct_start=0.43, div_factor=opt['train']['div_factor'], final_div_factor=opt['train']['final_div_factor'], three_phase=True, anneal_strategy='linear')
        print('Scheduler: OneCycle')

    return optimizer, scheduler

def save_weights(model, optimizer, scheduler=None, filename="model_weights.pth", rank=0):
    if rank != 0:
        return  # Solo el proceso con rank 0 guarda los pesos

    if not filename.endswith(".pt"):
        filename += ".pt"

    Weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Weights")
    full_path = os.path.join(Weights_dir, filename)
    
    if not os.path.exists(Weights_dir):
        os.makedirs(Weights_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, full_path)
    print(f"Pesos guardados exitosamente en {full_path}")

__all__ = ['create_model', 'create_optimizer_scheduler', 'save_weights']

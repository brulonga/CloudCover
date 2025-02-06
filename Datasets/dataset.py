import os
import torch.distributed as dist
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from Utils.transform import AllSky
from Utils.plot import *

def dataloader(opt, rank, world_size):

    root_path = opt['root_path']
    entrenamiento_path = opt['datasets']['entrenamiento']['entrenamiento_path']
    validacion_path = opt['datasets']['validacion']['validacion_path']
    test_path = opt['datasets']['test']['test_path']

    full_path_entrenamiento = os.path.join(root_path, entrenamiento_path)
    full_path_validacion = os.path.join(root_path, validacion_path)
    full_path_test = os.path.join(root_path, test_path)

    transform = None

    samplers = []

    if opt['datasets']['entrenamiento']['transform'] == 'AllSky':
        transform = AllSky()

    if transform is None:
        print("Advertencia: No se ha definido una transformación. Usando una transformación por defecto.")
        transform = transforms.ToTensor()

    dataset_ent = datasets.ImageFolder(root=full_path_entrenamiento, transform=transform)

    dataset_val = datasets.ImageFolder(root=full_path_validacion, transform=transform)

    dataset_test = datasets.ImageFolder(root=full_path_test, transform=transform)

    print('Dataset info:')
    print('\t Imágenes train:', len(dataset_ent))
    print('\t Imágenes val:', len(dataset_val))
    print('\t Imágenes test:', len(dataset_test))
    print('world size:', world_size)

    if world_size > 1:

        sampler_entrenamiento = DistributedSampler(dataset_ent, num_replicas=world_size, shuffle= True, rank=rank)

        sampler_validacion = DistributedSampler(dataset_val, num_replicas=world_size, shuffle= True, rank=rank)

        sampler_test = DistributedSampler(dataset_test, num_replicas=world_size, shuffle= True, rank=rank)

        LOADER_ENTRENAMIENTO = DataLoader(dataset_ent, batch_size=opt['datasets']['entrenamiento']['batch_size_entrenamiento'], shuffle=False, pin_memory=True, sampler=sampler_entrenamiento)

        #if rank == 0: plot_entrenamiento(opt, LOADER_ENTRENAMIENTO)
        #print('samplers')

        LOADER_VALIDACION = DataLoader(dataset_val, batch_size=opt['datasets']['validacion']['batch_size_validacion'], shuffle=False, pin_memory=True, sampler=sampler_validacion)

        #if rank==0: plot_validacion(opt, LOADER_VALIDACION)

        LOADER_TEST = DataLoader(dataset_test, batch_size=opt['datasets']['test']['batch_size_test'], shuffle=False, pin_memory=True, sampler=sampler_test)

        #if rank==0: plot_test(opt, LOADER_TEST)

        samplers.append(sampler_entrenamiento)
        samplers.append(sampler_validacion)

        
    else:        
        
        LOADER_ENTRENAMIENTO = DataLoader(dataset_ent, batch_size=opt['datasets']['entrenamiento']['batch_size_entrenamiento'], shuffle=True, pin_memory=True)

        plot_entrenamiento(opt, LOADER_ENTRENAMIENTO)

        LOADER_VALIDACION = DataLoader(dataset_val, batch_size=opt['datasets']['validacion']['batch_size_validacion'], shuffle=True, pin_memory=True)

        plot_validacion(opt, LOADER_VALIDACION)

        LOADER_TEST = DataLoader(dataset_test, batch_size=opt['datasets']['test']['batch_size_test'], shuffle=False, pin_memory=True)

        plot_test(opt, LOADER_TEST)

        samplers = None

    return LOADER_ENTRENAMIENTO, LOADER_VALIDACION, LOADER_TEST, samplers

def shuffle_sampler(samplers, epoch):
    '''
    A function that shuffles all the Distributed samplers in the loaders.
    '''
    if not samplers: # if they are none
        return
    for sampler in samplers:
        sampler.set_epoch(epoch)
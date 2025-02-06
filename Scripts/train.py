import torch    
import time
import wandb
import torch.nn
import torch.distributed as dist
from Utils.loss import CustomCrossEntropyLoss
from Utils.plot import *
from Models._init_ import save_weights
from Utils.init_wandb import init_wandb
from Utils.setup_distributed import cleanup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Datasets.dataset import shuffle_sampler

def AllSky_entrenamiento(opt, model, criterion, optimizer, scheduler, LOADER_ENTRENAMIENTO, LOADER_VALIDACION, samplers, rank, world_size):

    best_accuracy = 0  # O la mejor pérdida si prefieres usarla
    num_epochs = opt['train']['epochs']

    train_losses = []
    val_losses = []
    train_acuraccies = []
    val_acuraccies = []

    inicio = time.time()

    # Determinar la función de pérdida
    if opt['train']['loss'] == 'CustomCrossEntropyLoss':
        criterion = CustomCrossEntropyLoss()  # Instanciamos la clase
    elif opt['train']['loss'] == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()  # Usamos la función CrossEntropyLoss estándar de PyTorch
    else:
        raise ValueError(f"Función de pérdida no reconocida: {opt['train']['loss']}")

    # Inicia el seguimiento de los experimentos con wandb, solo en rank 0
    if rank == 0:
        init_wandb(opt)

    # Ciclo de entrenamiento
    for epoch in range(num_epochs):

        print('Epoca actual:', epoch)
        shuffle_sampler(samplers, epoch)
        print('Samplers distribuidos')
        
        model.train()  # Establecer el modelo en modo de entrenamiento
        
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in LOADER_ENTRENAMIENTO:

            # Mover los datos a la GPU
            inputs, labels = inputs.to(f'cuda:{rank}'), labels.to(f'cuda:{rank}')

            #print('Datos en las gpu')
            
            # Inicializar los gradientes
            optimizer.zero_grad()

            #print('gradientes inicializados')

            # Pasar los datos por el modelo
            outputs = model(inputs)

            #print('Forward completado', outputs)

            # Calcular la pérdida
            loss_train = criterion(outputs, labels)

            #print('perdida completada', loss_train)

            # Hacer la retropropagación
            loss_train.backward()

            #print('backpropagation ejecutado')

            # Actualizar los parámetros del modelo
            optimizer.step()

            #print('Paso del optimizador')

            # Si estás usando ReduceLROnPlateau, pasa la pérdida de validación
            # De hecho si es otro scheduler hay que moverlo fuera del batch
            scheduler.step()

            #print('Paso del scheduler')

            # Obtener las predicciones
            _, predicted = torch.max(outputs, 1)

            #print('Predicciones', predicted)

            # Actualizar las métricas
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            running_loss_train += loss_train.item()

        print('Entrenamiento completado de la época:', epoch)

        correct_train_tensor = torch.tensor(correct_train, dtype=torch.float32, device=f'cuda:{rank}')
        total_train_tensor = torch.tensor(total_train, dtype=torch.float32, device=f'cuda:{rank}')
        running_loss_train_tensor = torch.tensor(running_loss_train, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(correct_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(total_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(running_loss_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        if rank == 0:
            correct_train = correct_train_tensor.item()
            total_train = total_train_tensor.item()
            running_loss_train = running_loss_train_tensor.item()

        dist.barrier()

        # Validación
        model.eval()  # Poner el modelo en modo evaluación (sin actualizar pesos)

        correct_val = 0
        total_val = 0
        running_loss_val = 0.0

        with torch.no_grad():  # No necesitamos calcular gradientes durante la validación
            for inputs, labels in LOADER_VALIDACION:
                
                # Mover los datos a la GPU
                inputs, labels = inputs.to(f'cuda:{rank}'), labels.to(f'cuda:{rank}')
                outputs = model(inputs)

                loss_val = criterion(outputs, labels)
                
                _, predictedval = torch.max(outputs, 1)
                
                total_val += labels.size(0)
                correct_val += (predictedval == labels).sum().item()
                running_loss_val += loss_val.item()

        # Sincronización de métricas entre procesos
        correct_val_tensor = torch.tensor(correct_val, dtype=torch.float32, device=f'cuda:{rank}')
        total_val_tensor = torch.tensor(total_val, dtype=torch.float32, device=f'cuda:{rank}')
        running_loss_val_tensor = torch.tensor(running_loss_val, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(correct_val_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(total_val_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(running_loss_val_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        if rank == 0:
            correct_val = correct_val_tensor.item()
            total_val = total_val_tensor.item()
            running_loss_val = running_loss_val_tensor.item()

            avg_train_loss = running_loss_train / len(LOADER_ENTRENAMIENTO)
            avg_val_loss = running_loss_val / len(LOADER_VALIDACION)

            train_acuraccy = 100 * correct_train / total_train
            val_acuraccy = 100 * correct_val / total_val

            train_acuraccies.append(train_acuraccy)
            val_acuraccies.append(val_acuraccy)

            # Almacenar las pérdidas de entrenamiento y validación
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            current_lr = scheduler.get_last_lr()[0]

            perdida_entrenamiento = running_loss_train / len(LOADER_ENTRENAMIENTO)
            perdida_validacion = running_loss_val / len(LOADER_VALIDACION)

            # Log de wandb solo en rank 0
            wandb.log({'Precisión entrenamiento': train_acuraccy, 'Precisión validación': val_acuraccy, 
                        'Perdida entrenamiento': avg_train_loss, 'Perdida validacion': avg_val_loss})

            # Imprimir las métricas de la época
            print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {perdida_entrenamiento:.4f}, Precisión: {train_acuraccy:.2f}% , "
                  f"Precision Validacion: {val_acuraccy:.2f}%, Perdida validacion: {perdida_validacion:.4f}, Current LR: {current_lr:.8f}")

            # Comprobar si la precisión es mejor que la anterior
            if val_acuraccy > best_accuracy:
                best_accuracy = val_acuraccy
                save_weights(model, optimizer, scheduler, opt['network']['save_weights'], rank=0)
                print("Mejor modelo guardado")

        dist.barrier()

    fin = time.time()

    tiempo_entrenamiento = fin - inicio
    print(f"Tiempo total de entrenamiento: {tiempo_entrenamiento:.2f} segundos")

    plot_accuracy(opt, train_acuraccies, val_acuraccies)
    plot_loss(opt, train_losses, val_losses)

    cleanup()

    return perdida_entrenamiento, perdida_validacion, train_acuraccy, val_acuraccy


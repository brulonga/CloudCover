import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(opt, predicciones_tensor, etiquetas_tensor):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    predicciones_tensor_cpu = predicciones_tensor.cpu()
    etiquetas_tensor_cpu = etiquetas_tensor.cpu()

    # Calcular la matriz de confusión
    cm = confusion_matrix(etiquetas_tensor_cpu.numpy(), predicciones_tensor_cpu.numpy())

    # Convertir la matriz de confusión en porcentajes
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Visualizar la matriz de confusión con porcentajes
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
                xticklabels=['0', '1', '2','3', '4', '5','6', '7', '8'], yticklabels=['0', '1', '2','3', '4', '5','6', '7', '8'])
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión con Porcentajes')
    plot_filename = os.path.join(path, 'Matriz de Confusión.png')
    plt.savefig(plot_filename, bbox_inches='tight')

def plot_loss(opt, train_losses, val_losses):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    plt.plot(train_losses, label='Entrenamiento', color='blue')
    plt.plot(val_losses, label='Validación', color='green')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de Entrenamiento y Validación')
    plt.legend()
    plt.xscale('linear')
    plt.xlim(0, len(train_losses)) 
    plot_filename = os.path.join(path, 'Pérdida de Entrenamiento y Validación.png')
    plt.savefig(plot_filename, bbox_inches='tight')

def plot_accuracy(opt, train_acuraccies, val_acuraccies):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    plt.plot(train_acuraccies, label='Entrenamiento', color='blue')
    plt.plot(val_acuraccies, label='Validación', color='green')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Precisión de Entrenamiento y Validación')
    plt.legend()
    plt.xscale('linear')
    plt.xlim(0, len(train_acuraccies)) 
    plot_filename = os.path.join(path, 'Precisión de Entrenamiento y Validación.png')
    plt.savefig(plot_filename, bbox_inches='tight')

def plot_entrenamiento(opt, LOADER_ENTRENAMIENTO):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    # Inicializa un contador para cada clase
    class_counts = [0] * 9  

    # Recorrer el conjunto de datos de entrenamiento
    for inputs, labels in LOADER_ENTRENAMIENTO:
        for label in labels:
            class_counts[label] += 1

    # Mostrar el histograma
    plt.figure(figsize=(10, 6))
    plt.bar(range(9), class_counts, tick_label=[str(i) for i in range(9)])
    plt.xlabel('Clase')
    plt.ylabel('Número de ejemplos')
    plt.title('Distribución de ejemplos por clase en el conjunto de entrenamiento')
    plot_filename = os.path.join(path, 'Distribución de ejemplos por clase en el conjunto de entrenamiento.png')
    plt.savefig(plot_filename, bbox_inches='tight')

def plot_validacion(opt, LOADER_VALIDACION):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    class_counts_validacion = [0] * 9  

    # Recorrer el conjunto de datos de entrenamiento
    for inputs, labels in LOADER_VALIDACION:
        for label in labels:
            class_counts_validacion[label] += 1

    # Mostrar el histograma
    plt.figure(figsize=(10, 6))
    plt.bar(range(9), class_counts_validacion, tick_label=[str(i) for i in range(9)])
    plt.xlabel('Clase')
    plt.ylabel('Número de ejemplos')
    plt.title('Distribución de ejemplos por clase en el conjunto de validacion')
    plot_filename = os.path.join(path, 'Distribución de ejemplos por clase en el conjunto de validacion.png')
    plt.savefig(plot_filename, bbox_inches='tight')

def plot_test(opt, LOADER_TEST):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    class_counts_test = [0] * 9  

    # Recorrer el conjunto de datos de entrenamiento
    for inputs, labels in LOADER_TEST:
        for label in labels:
            class_counts_test[label] += 1

    # Mostrar el histograma
    plt.figure(figsize=(10, 6))
    plt.bar(range(9), class_counts_test, tick_label=[str(i) for i in range(9)])
    plt.xlabel('Clase')
    plt.ylabel('Número de ejemplos')
    plt.title('Distribución de ejemplos por clase en el conjunto de test')
    plot_filename = os.path.join(path, 'Distribución de ejemplos por clase en el conjunto de test.png')
    plt.savefig(plot_filename, bbox_inches='tight')

def plot_LR_losses(opt, lg_lr, losses):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    f1, ax1 = plt.subplots(figsize=(20,10))
    # ax1.plot(lr[60:-2], losses[60:-2])
    ax1.plot(lg_lr, losses)
    ax1.set_xscale('log')
    ax1.set_xticks([1e-4, 1e-3, 1e-2, 1e-1,2e-1, 1, 10])
    ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
    plot_filename = os.path.join(path, 'OneCycleLR_losses.png')
    plt.savefig(plot_filename, bbox_inches='tight')


def plot_LR_accuracy(opt, lg_lr, accuracies):

    root_path = opt['root_path']
    save_path = opt['network']['save_path']

    path = os.path.join(root_path, save_path)

    f1, ax1 = plt.subplots(figsize=(20,10))
    # ax1.plot(lr[60:-2], losses[60:-2])
    ax1.plot(lg_lr, accuracies)
    ax1.set_xscale('log')
    # ax1.set_xticks([1e-1, 2e-1,5e-1, 7e-1, 1, 10])
    ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
    plot_filename = os.path.join(path, 'OneCycleLR_accuracy.png')
    plt.savefig(plot_filename, bbox_inches='tight')

__all__ = ['plot_confusion_matrix', 'plot_loss', 'plot_accuracy', 'plot_entrenamiento', 'plot_validacion', 'plot_test', 'plot_LR_losses', 'plot_LR_accuracy']

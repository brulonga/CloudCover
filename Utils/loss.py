import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels):
        # Obtener las predicciones (clases predichas) usando los logits
        _, predicted = torch.max(outputs, 1)  # Esto te da los índices de las clases predichas

        # Calcular la distancia entre la etiqueta real y todas las clases (distancia promedio)
        distancia_promedio = self.calculate_class_distances(predicted, labels)
        
        # Calcular la pérdida estándar de CrossEntropy
        ce_loss = nn.CrossEntropyLoss()(outputs, labels)  # Asegúrate de usar los logits y las etiquetas correctas
        
        # Si quieres combinar la pérdida de CrossEntropy con la distancia, puedes hacerlo
        total_loss = distancia_promedio * ce_loss

        return total_loss

    def calculate_class_distances(self, predicted, labels):
        # Calculando la diferencia cuadrada (distancia Euclidiana)
        distancia = 0.0
        for i in range(len(predicted)):
            distancia += (predicted[i] - labels[i]) ** 2  # Diferencia cuadrada

        # Promedio de las distancias
        distancia_promedio = torch.sqrt(distancia / len(predicted))  # Raíz cuadrada de la media
        return distancia_promedio
    

__all__ = ['CustomCrossEntropyLoss']

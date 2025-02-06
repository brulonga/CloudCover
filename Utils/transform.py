from torchvision import transforms
import numpy as np

# Definir las probabilidades para las rotaciones
# 0°: 25%, 90°: 25%, 180°: 25%, 270°: 25%
opciones_rotaciones = [0, 90, 180, 270]

# Definir la clase de transformación AllSky
class AllSky:
    def __init__(self):
        pass
    
    def __call__(self, x):
        # Seleccionar aleatoriamente un ángulo para la rotación cada vez que se aplica la transformación
        angulo_aleatorio = np.random.choice(opciones_rotaciones, p=[0.25, 0.25, 0.25, 0.25])

        # Aplicar la transformación: rotación, flip horizontal y flip vertical
        x = transforms.RandomHorizontalFlip(p=0.5)(x)
        x = transforms.RandomVerticalFlip(p=0.2)(x)
        x = transforms.Lambda(lambda img: img.rotate(angulo_aleatorio))(x)
        
        # Convertir la imagen a tensor (sin normalizar)
        x = transforms.ToTensor()(x)
        
        return x

# Asegúrate de que esta clase esté accesible desde el módulo
__all__ = ['AllSky']


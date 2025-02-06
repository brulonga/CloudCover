# Convolutional Neural Networks for Cloud Cover classification in all sky images

Bruno Longarela

> **Abstract:**
> 
This work employs Convolutional Neural Networks (CNN) to automate the detection of cloud coverage in all-sky images. Data was collected from cameras installed in various locations across Spain, combining real and synthetic images generated by diffusion networks. Different CNN architectures are analyzed, optimizing their training to avoid overfitting and improve accuracy. The results show that CNNs outperform traditional color-based segmentation methods in cloud classification. The importance of a large and diverse dataset is highlighted to enhance model performance. The study contributes to the automatic monitoring of cloud cover, which can improve weather models. Improvements in network architecture and the use of transfer learning are proposed. In the future, this system could be applied to real-time climate prediction.


Real all sky images 
 
<img src="assets/imagen1.jpg" alt="add" width="150">  <img src="assets/imagen2.jpg" alt="add" width="150">  <img src="assets/imagen3.jpg" alt="add" width="150">  <img src="assets/imagen4.jpg" alt="add" width="150"> 
                                             
Syntetic all sky images 
 
<img src="assets/imagen5.png" alt="add" width="150">  <img src="assets/imagen6.png" alt="add" width="150">  <img src="assets/imagen7.png" alt="add" width="150">  <img src="assets/imagen8.png" alt="add" width="150"> 

&nbsp;

## Network Architecture

Another model, ResNet45, has been implemented in the code; however, the weights for this model are not available.

The weights for the CNN and ResNet36 models can be found here: [Pesos_CloudCover](https://drive.google.com/drive/folders/1KZWFAToNkq5duZJEkr1tDn150-vRDBGM?usp=drive_link)

The model weights should be located in the "Weights" folder.

<img src="assets/residual.jpg" alt="add" width="450">  <img src="assets/cnn.jpg" alt="add" width="450">

## Dependencies and Installation

- Python == 3.12.3
- PyTorch == 2.4.1
- CUDA == 12.6
- Other required packages in `requirements.txt`

```
# git clone this repository
git clone https://github.com/brulonga/CloudCover.git
cd CloudCover

# create python environment
python3 -m venv venv_CloudCover
source venv_CloudCover/bin/activate

# install python dependencies
pip install -r requirements.txt
```
The code is implemented for parallelized training using multiprocessing with the spawn method. It should work on a single GPU.

## Datasets

Currently, the dataset is not public. If you are interested in the real images, please contact longarela.bruno@gmail.com. 

The images must be placed in their respective folders:

```
CloudCover
  ├──Datasets
      ├──entrenamiento/    (Real images + syntetic images, 99900)
      ├──validacion/  (Real images, 9000)
      ├──test/       (Real images, 9000)
```

The synthetic images can be obtained through the pre-trained weights and the corresponding Jupyter notebook. You must change the paths to the images.

You can download the pretrained U-Net weights for diffusion here: [Pesos_Difusion](https://drive.google.com/drive/folders/18Uida-rjl7EKlqdIhzDRHJMQwDY0YvHK?usp=drive_link)

```
CloudCover
  ├──Difusion
      ├──difussion.ipynb
      ├──Pesos_Difusion/  (Classes 1,2,3,4,5,6,7)
```

Here you can watch a quick demo of how we obtain a syntetic image from class 4 oktas from gaussian noise.

![add](/assets/output.gif)


## Training

Network can be trained from scratch running 

```python3 main.py```

Configuration file for this training can be found in `/Options/Baseline.yml`. There you can select the Baseline for the model that you want to train with. 

For running the code, you just have to change in each ```Baseline.yaml```  the   ```root_path```.

Only evaluation code isn´t implemented yet, you can run just on evaluation by modifying the ```main.py```

## Contact

If you have any questions, please contact longarela.bruno@gmail.com

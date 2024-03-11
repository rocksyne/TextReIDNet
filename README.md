# TextReIDNet
This is the codebase for TextReIDNet (Text-based Person Re-identification Network). The details of TextReIDNet are found in our DCOSS-IoT 2024 paper titled "Resource-Efficient Text-based Person
Re-identification on Embedded Devices".

&nbsp;

# Summary
TextReIDNet is a lightweight person re-identification model designed explicitly for embedded devices. TextReIDNet has a total of 32.29 million parameters only, but a achieves an
impressive 52.76% and 35.71% top-1 accuracy on the CUHK-PEDES and RSTPReid datasets respectively.

![](docs/TextReIDNet_architetcure.png)
Figure 1. The architure of TextReIDNet.

&nbsp;
## Requirements (dependencies)
- Operating System: Ubuntu 20.04.6 LTS (irrelevant but worth mentioning)
- CUDA Version: 11.7
- python version: 3.9.18
- pytorch version: 1.13.1
- torchvision version: 0.14.1
- pillow version: 9.5.0
- opencv-python version: 4.8.1.78
- tqdm version: 4.66.1
- numpy version: 1.26.2
- natsort version: 8.4.0


The remaining requirements are specified in [requirements.txt](requirements.txt)

&nbsp;
## Resources
- CUHK-PEDES dataset: please see [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)
- RSTPReid dataset: please see [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)
- Pretrained model (30 epochs): [TextReIDNet_State_Dicts.pth.tar](https://drive.google.com/file/d/1Clry-_oJcQXDbA92H0ARUerlzw-9hdrI/view?usp=sharing)


&nbsp;
## Setup
- Install dependencies. Complete list of dependencies is in ```requirements.txt```
- Download or clone TextReIDNet repository
- Navigate into TextReIDNet directory: ```cd /path/to/TextReIDNet```
- Download the pre-trained model [TextReIDNet_State_Dicts.pth.tar](https://drive.google.com/file/d/1Clry-_oJcQXDbA92H0ARUerlzw-9hdrI/view?usp=sharing) and put it into ```TextReIDNet/data/checkpoints/TextReIDNet_State_Dicts.pth.tar```
- Read and modify the ```config.py``` to suit your system resources and requirements

&nbsp;
## Test (using pretrained model - 30 epochs)
- Please make sure the pre-trained model [TextReIDNet_State_Dicts.pth.tar](https://drive.google.com/file/d/1Clry-_oJcQXDbA92H0ARUerlzw-9hdrI/view?usp=sharing) is downloaded put it into ```TextReIDNet/data/checkpoints/TextReIDNet_State_Dicts.pth.tar```
- Also read and modify the ```config.py``` to suit your system parameters
- ```python test.py```. The test results are logged to the console only.

&nbsp;
## Inference on sample images
10 sample images are provided in ```TextReIDNet/data/samples/``` for inference purposes. ```boy_with_bag.jpg``` and ```girl_student.jpg``` are taken from the internet, while the remaining 8 are sampled from the CUHK-PEDES dataset. You may download and crop more single-person images from the internet and add them to ```TextReIDNet/data/samples/```.

- Please make sure the pre-trained model [TextReIDNet_State_Dicts.pth.tar](https://drive.google.com/file/d/1Clry-_oJcQXDbA92H0ARUerlzw-9hdrI/view?usp=sharing) is downloaded put it into ```TextReIDNet/data/checkpoints/TextReIDNet_State_Dicts.pth.tar```
- Also read and modify the ```config.py``` to suit your system parameters
- modify the value of ```textual_description``` in ```TextReIDNet/inference/search_person.py``` with the description of the person you would like to retrieve from ```TextReIDNet/data/samples/```.
- run ```TextReIDNet/inference/search_person.py```. The ranking results are logged to the console, while the top-1 image is saved to ```TextReIDNet/inference/retrieved_image.jpg```.


&nbsp;
## Train
- First read and modify the ```config.py``` to suit your system parameters
- ```python train.py```. The training progress and values are logged into ```TextReIDNet/data/logs/train.log```

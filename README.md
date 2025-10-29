# Project Name : Retinal Blindness (Diabetic Retinopathy) detection   

# Problem statement :    
Diabetic Retinopathy is a disease with an increasing prevalence and the main cause of blindness among working-age population.  
The risk of severe vision loss can be significantly reduced by timely diagnosis and treatment. Systematic screening for DR has been identified as a cost-effective way to save health services resources. Automatic retinal image analysis is emerging as an important screening tool for early DR detection, which can reduce the workload associated to manual grading as well as save diagnosis costs and time. Many research efforts in the last years have been devoted to developing automated tools to help in the detection and evaluation of DR lesions.
We are interested in automating this predition using deep learning models.

# Motivation : 
Early detection through regular retinal screening can drastically reduce vision loss — yet, in many regions, access to skilled ophthalmologists remains limited.   
This project aims to leverage deep learning to assist hospitals and diagnostic centers in detecting Diabetic Retinopathy from retinal fundus images.  
Our motivation was to build an AI-driven, scalable, and cost-effective screening tool that can and by open-sourcing this work, the goal is to empower hospitals, clinics, and NGOs to:

- Support ophthalmologists in identifying DR at an early stage.
- Improve screening efficiency in under-resourced hospitals.
- Enable large-scale, real-time retinal analysis through automation.

By combining data-driven insights with medical imaging, the project demonstrates how AI can bridge the gap between healthcare accessibility and diagnostic accuracy, contributing toward the broader goal of preventing avoidable blindness.   

**Note:** This project was inspired by the mission of _**Aravind Eye Hospital (India)**_ and the _**Asia Pacific Tele-Ophthalmology Society (APTOS)**_, which aim to bring AI-assisted screening to remote areas.


# Dataset : [APOTS Kaggle Blindness dataset](https://www.kaggle.com/c/aptos2019-blindness-detection)      

# Solution :   
I am proposing Deep Learning classification technique using CNN pretrained model [resnet152](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) to classify severity levels of DR ranging from 0 (NO DR) to 4 (Proliferative DR).   
This is a collaborative project of team of five where my main work is on developing, training and testing various CNN models along with some secondary work.
Deep learning looks promising because already various types of image classification tasks has been performed by various CNN's so, we can rely on DL pretrained models or we can modify some layers if we wish to :)     

# Summary of Technologies used in this project :       
| Dev Env. | Framework/ library/ languages |
| ------------- | ------------- |
| Backend development  | PyTorch (Deep learning framework) |
| Frontend development | Tkinter (Python GUI toolkit) |
| Programming Languages | Python| 


# Resnet152 model summary :     
The backbone of this project is a ResNet152 model, a deep convolutional neural network developed by Microsoft Research.
ResNet152 is part of the Residual Network family, which introduced the concept of skip connections — allowing extremely deep networks to be trained efficiently without suffering from vanishing gradients.

In this project, the ResNet152 model was fine-tuned on a labeled diabetic retinopathy dataset consisting of retinal fundus images. Each image belongs to one of five diagnostic categories based on disease severity.

Model Architecture

The pre-trained ResNet152 architecture (originally trained on ImageNet) was adapted for our classification task by modifying its fully connected layer.
The structure after modification is as follows:

Layer	Description
Base Network	ResNet152 (pretrained on ImageNet)
Frozen Layers	Conv1, Layer1
Unfrozen Layers	Layer2, Layer3, Layer4, Fully Connected Layer
Modified Fully Connected Block	Linear(2048 → 512) → ReLU → Linear(512 → 5) → LogSoftmax(dim=1)
Output Classes	5 (No DR, Mild, Moderate, Severe, Proliferative DR)
Fine-Tuning Strategy

To adapt the model effectively for retinal image classification:

Transfer Learning was used — the pretrained ResNet152 weights were retained for feature extraction.

Last three residual blocks (Layer2, Layer3, Layer4) were unfrozen to allow the model to learn DR-specific features.

The fully connected (FC) layer was replaced to match the number of output classes (5).

The model was optimized using Adam optimizer with a very low learning rate (1e-6) to preserve useful pretrained features while adapting to the new task.

The Negative Log-Likelihood Loss (NLLLoss) function was used since the final activation is LogSoftmax.

Training Details

Framework: PyTorch

Optimizer: Adam

Scheduler: StepLR (step size = 5, gamma = 0.1)

Loss Function: NLLLoss

Input Image Size: 224 × 224

Normalization:

Mean: (0.485, 0.456, 0.406)

Std: (0.229, 0.224, 0.225)

The fine-tuning was performed using the Diabetic Retinopathy 224x224 2019 Dataset available on Kaggle, which contains preprocessed retinal images categorized by severity levels.

After training, the best model weights were saved as best_model.pth, which achieved consistent and accurate predictions on unseen test images.      

[Note : The training files in this repo is only shown after final training as it took around more than 100 epochs to reach 97% accuracy and a lot of compute power and time.]     


⭐️ this Project if you liked it !

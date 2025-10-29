import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import preprocess_image

# Load fine-tuned model
model = models.resnet152(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 5),
    torch.nn.LogSoftmax(dim=1)
)
model.load_state_dict(torch.load("models/best_model.pth", map_location='cpu'))
model.eval()

# Define your image
image_path = "sampleimages/eye6.jpg"
img = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])
input_tensor = transform(img).unsqueeze(0)

# Grad-CAM setup
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM

target_layers = [model.layer4[-1]]  # final conv layer
cam = GradCAM(model=model, target_layers=target_layers)

# Generate heatmap for predicted class
preds = model(input_tensor)
pred_class = preds.argmax(dim=1).item()
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0, :]

# Overlay CAM on image
rgb_img = np.float32(img.resize((224, 224))) / 255
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.title(f"Grad-CAM for Class: {pred_class}")
plt.axis('off')
plt.show()

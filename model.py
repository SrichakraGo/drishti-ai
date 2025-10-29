# model.py — final version for local fine-tuned model
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

print(" Model script started")

# 1️ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2️ Load architecture
model = models.resnet152(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, 5),
    nn.LogSoftmax(dim=1)
)

# 3️ Load your trained weights
model_path = os.path.join("models", "best_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Could not find model file at {model_path}")

# Load the saved model (it might be saved as full model or checkpoint)
checkpoint = torch.load(model_path, map_location=device)

# Try to detect save format
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Loaded model_state_dict from checkpoint")
else:
    model.load_state_dict(checkpoint)
    print("✅ Loaded entire model weights directly")

model.to(device)
model.eval()

# ------------------------------------------------------------
# 4️ Label names
classes = ['Mild', 'Moderate', 'No DR', 'Proliferative DR', 'Severe']

# ------------------------------------------------------------
# 5️ Transform pipeline
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# ------------------------------------------------------------
# Inference function
def inference(model, file, transform, classes):
    image = Image.open(file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.exp(outputs)
        top_p, top_class = probs.topk(1, dim=1)
        value = top_class.item()

    # print(f" Predicted Severity Value: {value}")
    print(f"Class: {classes[value]}")
    return value, classes[value]

# ------------------------------------------------------------
# 7️⃣ Main callable function (used in blindness.py)
def main(path):
    y = inference(model, path, test_transforms, classes)
    return y

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same CNN model used in training
class DeepFakeCNN(nn.Module):
    def __init__(self):
        super(DeepFakeCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Define the image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load model
model = DeepFakeCNN().to(device)
model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))
model.eval()

# 🔥 Paste your image path here
image_path = r"S:/Documents/6th sem/PR/Celeb-DF Preprocessed/test/fake/id0_id1_0002_frame240_face5.jpg"  # ⬅️ CHANGE THIS

# Prediction function
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image path not found.")
        return

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_names = ["real", "fake"]
        print(f"🧠 Prediction: {class_names[predicted.item()].upper()}")

# Call the prediction function
predict_image(image_path)

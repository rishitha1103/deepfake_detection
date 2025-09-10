import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_path = "S:/Documents/6th sem/PR/Celeb-DF Preprocessed"
img_size = (64, 64)  # Smaller size reduces memory usage

# Function to load images efficiently
def load_images(folder, max_images=None):
    classes = ["real", "fake"]
    images, labels = [], []

    for label, subfolder in enumerate(classes):
        folder_path = os.path.join(folder, subfolder)
        file_list = os.listdir(folder_path)[:max_images]  # Load a subset

        for filename in file_list:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)  # More memory-efficient
                images.append(img.flatten().astype(np.uint8))  # Use uint8
                labels.append(label)

    return np.array(images), np.array(labels)

# Load only a subset first to prevent memory errors
print("Loading training data...")
train_images, train_labels = load_images(os.path.join(dataset_path, "train"), max_images=5000)
print("Loading validation data...")
val_images, val_labels = load_images(os.path.join(dataset_path, "val"), max_images=1000)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduce trees to save memory
model.fit(train_images, train_labels)

# Validate model
val_preds = model.predict(val_images)
val_acc = accuracy_score(val_labels, val_preds)
print(f"✅ Validation Accuracy: {val_acc:.4f}")

# Save trained model
joblib.dump(model, "deepfake_detector.pkl")
print("✅ Model saved as deepfake_detector.pkl!")

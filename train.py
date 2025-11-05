import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image

# ------------------------------
# 1. Load dataset
# ------------------------------
print("Loading dataset...")
dataset = load_dataset("aryadytm/vehicle-classification")

# Check available splits
print("Available splits:", dataset.keys())

# Use only 'train' and split manually
full_train_dataset = dataset["train"]

# Check class info
class_names = full_train_dataset.features["label"].names
num_classes = len(class_names)
print("Classes:", class_names)
print("Number of classes:", num_classes)

# ------------------------------
# 2. Image transforms
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def transform_batch(example_batch):
    images = [transform(Image.open(img).convert("RGB")) for img in example_batch["image"]]
    labels = [int(l) for l in example_batch["label"]]  # Ensure integer labels
    return {"pixel_values": images, "labels": labels}

full_train_dataset = full_train_dataset.with_transform(transform_batch)

# ------------------------------
# 3. Train/Validation Split
# ------------------------------
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ------------------------------
# 4. Model setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 5. Training loop
# ------------------------------
epochs = 3
print("Starting training...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs = torch.stack(batch["pixel_values"]).to(device)
        labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

        # FIX: clamp labels to valid range (0 to num_classes-1)
        labels = torch.clamp(labels, 0, num_classes - 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training complete!")

# ------------------------------
# 6. Save model
# ------------------------------
torch.save(model.state_dict(), "vehicle_model.pth")
print("💾 Model saved as vehicle_model.pth")

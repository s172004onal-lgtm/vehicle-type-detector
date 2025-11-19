import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO

# CLASS NAMES (same as training)
class_names = ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar',
               'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile',
               'Tank', 'Taxi', 'Truck', 'Van']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("vehicle_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = outputs.max(1)

    return class_names[predicted.item()]

# Run prediction
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    result = predict_image(image_path)
    print(f"\n🔍 Predicted Vehicle Type: {result}\n")

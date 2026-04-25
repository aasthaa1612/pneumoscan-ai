import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Config ──
MODEL_PATH = "../outputs/models/best_model.pt"  # change to your model path
TEST_DIR   = "../data/raw/chest_xray/test"      # change to your test folder path
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load Test Data ──
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names  = test_dataset.classes
print(f"Classes: {class_names}")

# ── Load Model ──
import sys
sys.path.append('c:\\project\\backend')
from models.cnn_model import get_model_for_checkpoint

model, meta, variant = get_model_for_checkpoint(MODEL_PATH, DEVICE)

# ── Evaluate ──
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# ── Results ──
accuracy = accuracy_score(all_labels, all_preds)
f1       = f1_score(all_labels, all_preds, average='weighted')

print(f"\n{'='*40}")
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  F1 Score : {f1:.4f}")
print(f"{'='*40}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
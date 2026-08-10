import os
import argparse
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# ========================== SETTINGS ==========================
DEVICE = "cpu"                    # Change to "cuda" if you have GPU
torch.set_num_threads(4)
print(f"[GrowBotOS] Running on {DEVICE.upper()}")

STRESS_CLASSES = [
    "Healthy", "Water Stress", "Nitrogen Deficiency", "Phosphorus Deficiency",
    "Potassium Deficiency", "Temperature Stress", "pH Imbalance"
]
NUM_CLASSES = len(STRESS_CLASSES)
SENSOR_DIM = 7

# ========================== IMAGE PROCESSING ==========================
def grey_world_white_balance(bgr):
    img = bgr.astype(np.float32)
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]

    mb = np.mean(B) + 1e-6
    mg = np.mean(G) + 1e-6
    mr = np.mean(R) + 1e-6

    gm = (mb + mg + mr) / 3.0

    return np.stack([
        np.clip(B * (gm / mb), 0, 255),
        np.clip(G * (gm / mg), 0, 255),
        np.clip(R * (gm / mr), 0, 255)
    ], axis=2).astype(np.uint8)


def exg_segment(bgr_frame):
    working = grey_world_white_balance(bgr_frame.copy())

    img = working.astype(np.float32)
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]

    total = R + G + B + 1e-6

    exg = (2.0 * G / total - R / total - B / total) * 255
    mask_exg = (exg > 10).astype(np.uint8)

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, np.array([30,30,30]), np.array([90,255,255]))
    m2 = cv2.inRange(hsv, np.array([20,40,40]), np.array([35,255,255]))
    m3 = cv2.inRange(hsv, np.array([155,30,50]), np.array([180,255,255]))

    mask_hsv = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3)) > 0

    combined = cv2.bitwise_or(mask_exg, mask_hsv.astype(np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    masked = bgr_frame.copy()
    masked[combined == 0] = 0

    return masked


# ========================== DATASET ==========================
class PlantDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224, apply_exg=True):
        self.apply_exg = apply_exg
        self.dataset = ImageFolder(os.path.join(root_dir, split))
        self.img_size = img_size

        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.35,
                    contrast=0.35,
                    saturation=0.35,
                    hue=0.2
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_pil, label = self.dataset[idx]

        np_img = np.array(img_pil)

        if self.apply_exg:
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            bgr_masked = exg_segment(bgr)
            np_img = cv2.cvtColor(bgr_masked, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(np_img)

        tensor = self.transform(img_pil)

        return tensor, label, torch.tensor([0.5], dtype=torch.float32)


# ========================== MODEL ==========================
class MultiHeadPlantCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, sensor_dim=SENSOR_DIM, dropout=0.3):
        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(576, 256),
            nn.Hardswish(),

            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(576, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.decision_head = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, image, sensor):
        feat = self.pool(self.backbone(image)).flatten(1)

        cls_logits = self.cls_head(feat)

        intensity = self.reg_head(feat)

        sensor_feat = self.sensor_encoder(sensor)

        fused_logits = self.decision_head(
            torch.cat([feat, sensor_feat], dim=1)
        )

        return cls_logits, fused_logits, intensity


# ========================== TRAINING ==========================
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=1e-4,
    save_path="growbotos_model.pth"
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_acc = 0.0

    # METRIC STORAGE
    train_losses = []
    val_accuracies = []

    print("\n🚀 Starting Training...\n")

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        for imgs, labels, intensities in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}"
        ):

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            intensities = intensities.to(DEVICE)

            sensor_vec = torch.zeros(
                (imgs.size(0), SENSOR_DIM),
                device=DEVICE
            )

            cls, fused, inten = model(imgs, sensor_vec)

            loss = (
                0.4 * ce_loss(cls, labels)
                + 0.5 * ce_loss(fused, labels)
                + 0.1 * mse_loss(inten, intensities)
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_loss += loss.item()

        # ================= VALIDATION =================
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels, _ in val_loader:

                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                sensor_vec = torch.zeros(
                    (imgs.size(0), SENSOR_DIM),
                    device=DEVICE
                )

                _, fused, _ = model(imgs, sensor_vec)

                preds = fused.argmax(dim=1)

                correct += (preds == labels).sum().item()

                total += labels.size(0)

        val_acc = correct / max(total, 1)

        avg_loss = train_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save(model.state_dict(), save_path)

            print(
                f"   → Best model saved "
                f"(Val Acc: {val_acc:.4f})"
            )

        gc.collect()

    # ================= SAVE TRAINING CURVES =================
    plt.figure(figsize=(12,5))

    # LOSS GRAPH
    plt.subplot(1,2,1)

    plt.plot(
        train_losses,
        marker='o',
        linewidth=2
    )

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # ACCURACY GRAPH
    plt.subplot(1,2,2)

    plt.plot(
        val_accuracies,
        marker='o',
        linewidth=2
    )

    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("training_curves.png")

    plt.close()

    print("Saved: training_curves.png")

    print(
        f"\n✅ Training Finished! "
        f"Best Validation Accuracy: {best_acc:.4f}"
    )

    return model


# ========================== FINAL EVALUATION ==========================
def final_evaluation(model_path, test_dir):

    print("\n📊 Running Final Evaluation on Test Set...")

    model = MultiHeadPlantCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    test_ds = PlantDataset(
        test_dir,
        split="test",
        apply_exg=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for imgs, labels, _ in test_loader:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            sensor_vec = torch.zeros(
                (imgs.size(0), SENSOR_DIM),
                device=DEVICE
            )

            _, fused, _ = model(imgs, sensor_vec)

            preds = fused.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=STRESS_CLASSES,
        yticklabels=STRESS_CLASSES
    )

    plt.title('GrowBotOS - Confusion Matrix')

    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig("confusion_matrix.png")

    plt.close()

    # ================= CLASSIFICATION REPORT =================
    report = classification_report(
        all_labels,
        all_preds,
        target_names=STRESS_CLASSES,
        digits=4
    )

    with open("classification_report.txt", "w") as f:
        f.write(report)

    accuracy = accuracy_score(all_labels, all_preds)

    # ================= BAR GRAPH FOR PER-CLASS ACCURACY =================
    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)

    class_acc = class_correct / np.maximum(class_total, 1)

    plt.figure(figsize=(12,6))

    plt.bar(STRESS_CLASSES, class_acc)

    plt.title("Per-Class Accuracy")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=20)

    plt.ylim(0, 1)

    plt.grid(axis='y')

    plt.tight_layout()

    plt.savefig("per_class_accuracy.png")

    plt.close()

    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)

    print(f"\nOverall Test Accuracy: {accuracy:.4f}\n")

    print(report)

    print("\nSaved Files:")
    print(" - confusion_matrix.png")
    print(" - per_class_accuracy.png")
    print(" - classification_report.txt")


# ========================== MAIN ==========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to Balanced_Dataset folder"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--model",
        default="growbotos_model.pth"
    )

    args = parser.parse_args()

    print(f"Using dataset: {args.data}\n")

    train_ds = PlantDataset(
        args.data,
        "train",
        apply_exg=True
    )

    val_ds = PlantDataset(
        args.data,
        "val",
        apply_exg=True
    )

    test_ds = PlantDataset(
        args.data,
        "test",
        apply_exg=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    model = MultiHeadPlantCNN().to(DEVICE)

    train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        save_path=args.model
    )

    final_evaluation(args.model, args.data)

    print("\n🎉 Training & Evaluation Completed!")

    print(f"Model saved as: {args.model}")

    print("Generated evaluation files:")
    print(" - training_curves.png")
    print(" - confusion_matrix.png")
    print(" - per_class_accuracy.png")
    print(" - classification_report.txt")

    print("\nCopy this model to your Raspberry Pi 4 for inference.")
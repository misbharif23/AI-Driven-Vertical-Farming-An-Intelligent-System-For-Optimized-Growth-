# ===================================================================
# FILE 1: pseudo_labeling.py  (CPU + Memory Optimized Version)
# ===================================================================
# Optimized for your current system (80% memory used, running on CPU)

import os
import shutil
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
import gc  # garbage collector

# ── Device & Memory Optimization ─────────────────────────────────────
DEVICE = "cpu"  # Force CPU since you have no CUDA
print(f"[GrowBotOS Labeling] Running on {DEVICE.upper()} - Memory Optimized Mode")

torch.set_num_threads(4)  # Limit CPU threads

# ── SimCLR Augmentation ───────────────────────────────────────────────
class SimCLRAugmentation:
    def __init__(self, img_size: int = 224):
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        return self.transform(image), self.transform(image)


class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_features: int = 576, hidden: int = 256, out: int = 128):  # smaller hidden
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.tau
        mask = torch.eye(2 * N, dtype=torch.bool, device=DEVICE)
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(DEVICE)
        return F.cross_entropy(sim, labels)


def pretrain_simclr(unlabeled_loader, encoder, epochs=5, lr=1e-3):
    proj_head = SimCLRProjectionHead().to(DEVICE)
    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=lr)

    encoder.train()
    proj_head.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (v1, v2) in enumerate(unlabeled_loader):
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)

            z1 = proj_head(encoder(v1))
            z2 = proj_head(encoder(v2))

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(unlabeled_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(len(unlabeled_loader), 1)
        print(f"[SimCLR] Epoch {epoch+1}/{epochs}  Avg Loss = {avg_loss:.4f}")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return encoder


# ── Encoder ───────────────────────────────────────────────────────────
def get_encoder():
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    return encoder.to(DEVICE)


def extract_embeddings(encoder, loader):
    encoder.eval()
    all_emb = []
    all_paths = []

    print("Extracting embeddings (this may take time)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                imgs, paths = batch[0], batch[1]
            else:
                imgs, paths = batch, [None] * len(batch)

            imgs = imgs.to(DEVICE)
            emb = encoder(imgs)
            all_emb.append(emb.cpu().numpy())
            all_paths.extend(paths)

            if batch_idx % 20 == 0:
                print(f"  Processed {batch_idx * loader.batch_size} images...")

    embeddings = np.vstack(all_emb)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings, all_paths


# ── Clustering ────────────────────────────────────────────────────────
def kmeans_pseudo_labels(embeddings, n_clusters=7):
    print("Running K-Means clustering...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=10)
    cluster_ids = kmeans.fit_predict(emb_scaled)
    return cluster_ids, emb_scaled


def evaluate_clustering(embeddings_scaled, cluster_ids):
    print("\n=== CLUSTERING EVALUATION METRICS (for your report) ===")
    try:
        sil = silhouette_score(embeddings_scaled, cluster_ids)
        db = davies_bouldin_score(embeddings_scaled, cluster_ids)
        ch = calinski_harabasz_score(embeddings_scaled, cluster_ids)
        print(f"Silhouette Score      : {sil:.4f}  (higher = better)")
        print(f"Davies-Bouldin Index  : {db:.4f}  (lower  = better)")
        print(f"Calinski-Harabasz     : {ch:.2f}   (higher = better)")
    except Exception as e:
        print(f"Evaluation metrics failed: {e}")


# ── Datasets ──────────────────────────────────────────────────────────
class UnlabeledSimCLRDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
        self.aug = SimCLRAugmentation(img_size)
        print(f"Found {len(self.image_paths)} images for SimCLR pre-training")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.aug(image)


class UnlabeledEmbeddingDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        tensor = self.transform(image)
        return tensor, self.image_paths[idx]


# ========================== MAIN ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_dir", type=str, 
                        default=r"C:\Users\Misbha\Desktop\FYP\CNN\model\data",
                        help="Path to unlabeled leaf images")
    parser.add_argument("--output_dir", type=str, default="pseudo_labeled_clusters")
    parser.add_argument("--epochs", type=int, default=5, help="SimCLR epochs (keep low on CPU)")
    parser.add_argument("--n_clusters", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8, help="Small batch for low memory")
    args = parser.parse_args()

    print("=== GrowBotOS Pseudo-Labeling Started (Memory Optimized) ===\n")

    # Step 1: SimCLR Pre-training
    print("STEP 1: SimCLR Pre-training...")
    simclr_dataset = UnlabeledSimCLRDataset(args.unlabeled_dir)
    simclr_loader = DataLoader(simclr_dataset, batch_size=args.batch_size, 
                               shuffle=True, num_workers=0, pin_memory=False)

    encoder = get_encoder()
    encoder = pretrain_simclr(simclr_loader, encoder, epochs=args.epochs)
    torch.save(encoder.state_dict(), "simclr_encoder.pth")
    print("✓ SimCLR encoder saved as simclr_encoder.pth\n")

    # Step 2: Extract Embeddings
    print("STEP 2: Extracting embeddings...")
    emb_dataset = UnlabeledEmbeddingDataset(args.unlabeled_dir)
    emb_loader = DataLoader(emb_dataset, batch_size=16, shuffle=False, 
                            num_workers=0, pin_memory=False)

    embeddings, image_paths = extract_embeddings(encoder, emb_loader)

    # Step 3: K-Means
    print("\nSTEP 3: K-Means Pseudo-Labeling...")
    cluster_ids, emb_scaled = kmeans_pseudo_labels(embeddings, n_clusters=args.n_clusters)
    evaluate_clustering(emb_scaled, cluster_ids)

    # Step 4: Create Folders
    print("\nSTEP 4: Creating cluster folders...")
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.n_clusters):
        os.makedirs(os.path.join(args.output_dir, f"cluster_{i}"), exist_ok=True)

    for path, cid in zip(image_paths, cluster_ids):
        dest = os.path.join(args.output_dir, f"cluster_{cid}", os.path.basename(path))
        shutil.copy2(path, dest)

    # Save CSV
    df = pd.DataFrame({"image_path": image_paths, "cluster_id": cluster_ids})
    df.to_csv("cluster_assignments.csv", index=False)

    print(f"\n✅ SUCCESS! {len(image_paths)} images organized into '{args.output_dir}'")
    print("   • Check the evaluation metrics above for your report")
    print("   • Next step: Manually review images in each cluster_{0-6} folder")
    print("     and rename folders to: Healthy, Water_Stress, etc.")

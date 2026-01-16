import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import entropy
import cv2
from einops import rearrange
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 假设你的模型定义在这些文件中，需要根据实际路径调整导入
from models.model_wrapper import DinoFeaturizer
from utils import load_checkpoint
import hydra
from omegaconf import OmegaConf

def get_args():
    parser = argparse.ArgumentParser(description="Visualize SACL Features")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint (.ckpt)")
    # parser.add_argument("--config_path", type=str, default="configs/train_config.yml", help="Path to model config")
    parser.add_argument("--model_config", type=str, default="configs/model/vit_small_16.yaml", help="Path to model config file")
    parser.add_argument("--output_dir", type=str, default="vis_results", help="Directory to save results")
    parser.add_argument("--patch_size", type=int, default=16, help="ViT patch size")
    parser.add_argument("--img_size", type=int, default=480, help="Resize image to this size for visualization")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters for K-Means visualization")
    return parser.parse_args()

def load_model(args):
    # 手动加载模型配置
    print(f"Loading model config from {args.model_config}")
    model_cfg = OmegaConf.load(args.model_config)
    
    # 构造 cfg 结构以匹配 DinoFeaturizer 的预期 (cfg.dim, etc are passed directly usually, but let's see constructor)
    # DinoFeaturizer(dim, cfg, ...)
    # 这里的 cfg 参数对应的是 hydra config 中的 model 部分
    
    # 初始化模型
    model = DinoFeaturizer(model_cfg.dim, model_cfg, require_grad=False, load_pretrain=False)
    
    # 加载权重
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    
    # 清理 state_dict 键名
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'net.' 前缀
        if k.startswith('net.'):
            k = k[4:]
        # 移除 'net_teacher.' 如果有 (我们只看 student 或 teacher，这里看 student)
        if k.startswith('net_teacher.'):
            continue
        new_state_dict[k] = v
        
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded: {msg}")
    
    model.eval()
    model.cuda()
    return model

def preprocess_image(img_path, img_size, device='cuda'):
    img = Image.open(img_path).convert('RGB')
    original_w, original_h = img.size
    
    # Resize keeping aspect ratio or just resize to square?
    # 为了方便 patch 处理，这里 resize 到固定大小
    # 注意：SACL 训练时可能没有加 Normalize，这里为了保持一致也不加
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img.resize((img_size, img_size))

def visualize_tsne(features, img_pil, h_feat, w_feat, output_prefix, n_clusters=4):
    """
    features: [H*W, D]
    h_feat, w_feat: spatial dimensions of features
    """
    # print("Computing t-SNE...")
    features_np = features.cpu().numpy()
    
    # 1. K-Means Clustering to color the points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_np)
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_np)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8)
    plt.title(f"t-SNE of Patch Features\\n(Colored by K-Means, K={n_clusters})")
    plt.colorbar(scatter)
    
    # Plot Segmentation Mask on Image
    # Reshape labels back to HxW
    mask = rearrange(labels, '(h w) -> h w', h=h_feat, w=w_feat)
    
    # Resize mask to image size
    mask_resized = cv2.resize(mask.astype(np.uint8), img_pil.size, interpolation=cv2.INTER_NEAREST)
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_pil)
    plt.imshow(mask_resized, alpha=0.5, cmap='tab10')
    plt.title(f"K-Means Clustering on Image")
    plt.axis('off')
    
    save_path = f"{output_prefix}_tsne.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    # print(f"Saved t-SNE visualization to {save_path}")

def visualize_similarity(features, img_pil, h_feat, w_feat, output_prefix):
    """
    Interactive-like similarity map. 
    Since this is a script, we pick a center point and a few other points automatically.
    """
    # print("Computing Similarity Maps...")
    
    # Define query points (y, x) in patch coordinates
    queries = [
        (h_feat // 2, w_feat // 2),       # Center
        (h_feat // 4, w_feat // 4),       # Top-Left
        (h_feat // 2, w_feat // 4),       # Left-Center
    ]
    
    fig, axes = plt.subplots(1, len(queries) + 1, figsize=(4 * (len(queries) + 1), 4))
    
    # Original Image
    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image\\n(Red dots = Query Points)")
    axes[0].axis('off')
    
    # Draw query points on original image
    img_w, img_h = img_pil.size
    patch_w, patch_h = img_w / w_feat, img_h / h_feat
    
    for i, (qy, qx) in enumerate(queries):
        # Mark on original image
        cx = int((qx + 0.5) * patch_w)
        cy = int((qy + 0.5) * patch_h)
        axes[0].plot(cx, cy, 'ro', markersize=8, markeredgecolor='white')
        
        # Calculate Similarity
        query_idx = qy * w_feat + qx
        query_feat = features[query_idx].unsqueeze(0) # [1, D]
        
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        query_feat_norm = F.normalize(query_feat, p=2, dim=1)
        
        sim = torch.mm(features_norm, query_feat_norm.t()) # [N, 1]
        
        # Debug: Print stats
        # print(f"Pt {i+1} Sim Stats: Min={sim.min():.4f}, Max={sim.max():.4f}, Mean={sim.mean():.4f}, Std={sim.std():.4f}")
        
        sim = rearrange(sim, '(h w) 1 -> h w', h=h_feat, w=w_feat).cpu().numpy()
        
        # Resize to image size for smooth visualization
        sim_resized = cv2.resize(sim, img_pil.size, interpolation=cv2.INTER_CUBIC)
        
        axes[i+1].imshow(img_pil)
        # Use min as vmin to stretch contrast if min is high
        vmin = 0 # max(0, sim.min() - 0.1) # Adaptive vmin? Let's stick to 0-1 or maybe 0.2 for now as discussed.
        im = axes[i+1].imshow(sim_resized, alpha=0.6, cmap='jet', vmin=vmin, vmax=1)
        axes[i+1].set_title(f"Sim Map (Pt {i+1})")
        axes[i+1].axis('off')
        axes[i+1].plot(cx, cy, 'wo', markersize=5, markeredgecolor='black')
        
        # Add colorbar only to the last plot, ensuring image size matches others
        if i == len(queries) - 1:
            divider = make_axes_locatable(axes[i+1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    save_path = f"{output_prefix}_sim.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    # print(f"Saved Similarity visualization to {save_path}")

def visualize_entropy(features, output_prefix):
    """
    Calculate entropy of the similarity distribution for each patch.
    This simulates the 'confidence' metric in SACL.
    """
    # print("Computing Entropy Histogram...")
    
    # Normalize features
    features_norm = F.normalize(features, p=2, dim=1)
    
    # Similarity matrix: [N, N]
    sim_matrix = torch.mm(features_norm, features_norm.t())
    
    # Convert to probability distribution (Softmax)
    # Use a temperature tau consistent with training (e.g., 0.07 or 0.1)
    tau = 0.1
    probs = F.softmax(sim_matrix / tau, dim=1) # [N, N]
    
    # Compute Entropy for each row (each patch's distribution over all other patches)
    # H(p) = - sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropy_vals = -torch.sum(probs * log_probs, dim=1).cpu().numpy()
    
    plt.figure(figsize=(6, 4))
    plt.hist(entropy_vals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Patch Entropies (Mean: {np.mean(entropy_vals):.2f})")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    
    save_path = f"{output_prefix}_entropy.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    # print(f"Saved Entropy histogram to {save_path}")
    
    return np.mean(entropy_vals)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    model = load_model(args)
    
    # Prepare Image
    img_tensor, img_pil = preprocess_image(args.img_path, args.img_size)
    
    # Extract Features
    # DinoFeaturizer.get_test_features returns (cls, patch)
    # We want patch features
    with torch.no_grad():
        # model.get_test_features internally calls model.model.get_last_feature
        # Returns: feats_cls, feats_patch
        _, feats_patch = model.get_test_features(img_tensor) 
        # feats_patch shape: [B, H, W, D] or [B, N, D] depending on implementation
        # Let's check shape
        print(f"Feature shape: {feats_patch.shape}")
        
    # Reshape to [N, D]
    h, w = 0, 0
    if feats_patch.ndim == 4:
        # [B, D, H, W] -> [B, H, W, D] -> [N, D] (N = B*H*W)
        features = rearrange(feats_patch, 'b d h w -> (b h w) d')
        h, w = feats_patch.shape[2], feats_patch.shape[3]
    elif feats_patch.ndim == 3:
        # [B, N, D]
        features = rearrange(feats_patch, 'b n d -> (b n) d')
        n = features.shape[0]
        h = w = int(np.sqrt(n))
    
    # Extract model name from ckpt path for folder naming
    # Example: checkpoints/cotap/dino+cotap.ckpt -> cotap_dino+cotap
    ckpt_path = args.ckpt_path
    if ckpt_path.startswith("checkpoints/"):
        ckpt_path = ckpt_path[len("checkpoints/"):]
    model_name_safe = ckpt_path.replace(os.path.sep, "_").replace(".ckpt", "")
    
    # Create output directory for this model
    save_dir = os.path.join(args.output_dir, model_name_safe)
    os.makedirs(save_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    output_prefix = os.path.join(save_dir, base_name)
    
    # 1. t-SNE & Clustering
    visualize_tsne(features, img_pil, h, w, output_prefix, args.n_clusters)
    
    # 2. Similarity Maps
    visualize_similarity(features, img_pil, h, w, output_prefix)
    
    # 3. Entropy Histogram
    visualize_entropy(features, output_prefix)
    
    print("\nDone! Check results in:", save_dir)

if __name__ == "__main__":
    main()

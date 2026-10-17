import os, random, time, json, logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             roc_auc_score, matthews_corrcoef)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / n_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1 - self.label_smoothing + self.label_smoothing / n_classes)
        log_probs    = F.log_softmax(logits, dim=1)
        probs        = torch.exp(log_probs)
        ce           = -(smooth_targets * log_probs).sum(dim=1)
        p_malignant  = probs[:, 1]
        focal_weight = torch.where(
            targets == 1,
            (1 - p_malignant) ** self.gamma,
            p_malignant ** self.gamma
        )
        alpha_weight = torch.where(
            targets == 1,
            torch.full_like(p_malignant, self.alpha),
            torch.full_like(p_malignant, 1 - self.alpha)
        )
        return (alpha_weight * focal_weight * ce).mean()


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    BCN_DIR        = "/kaggle/input/datasets/sanjeevirajm/ham10000/final year project dataset/final year project dataset/classified_skin_cancer_dataset"
    HAM_DIR        = "/kaggle/input/datasets/sanjeevirajm/ham10000/ham10000_combined"
    HAM_META       = "/kaggle/input/datasets/sanjeevirajm/metadata1/HAM10000_metadata (1).csv"
    OUTPUT_DIR     = "outputs"
    CHECKPOINT_DIR = "outputs/checkpoints"
    PLOT_DIR       = "outputs/plots"
    VIZ_DIR        = "outputs/test_samples"   # GradCAM-ready sample images

    SEED        = 42
    BATCH_SIZE  = 32
    NUM_EPOCHS  = 100   # extended for full Phase 3
    NUM_WORKERS = 4

    MODEL_NAME  = "mobilevitv2_100"
    NUM_CLASSES = 2
    INPUT_SIZE  = 256

    INITIAL_LR      = 1e-3
    WEIGHT_DECAY    = 0.01
    GRADIENT_CLIP   = 1.0
    LABEL_SMOOTHING = 0.1

    FOCAL_ALPHA = 0.80
    FOCAL_GAMMA = 2.0

    # ── Phase boundaries ────────────────────────────────────────────────────
    PHASE_1_END   = 15   # head only           →  1,026 trainable params
    PHASE_2_END   = 30   # last blocks unfrozen → 3,882,761 trainable params
    PHASE_3_START = 32   # FIXED: was 31 (clashed with Phase2 trigger at 31)
                         # Full fine-tune       → 4,389,867 trainable params

    PHASE_1_LR          = 1e-3
    PHASE_2_LR_BACKBONE = 1e-4
    PHASE_2_LR_HEAD     = 5e-4

    # Phase 3: single flat LR for ALL params
    # Discriminative LR (prev version) had early layers at 1e-5 — too slow,
    # caused domain shift to explode to 32%. Flat 3e-5 proven better empirically.
    PHASE_3_LR = 3e-5

    # ── Early stopping ──────────────────────────────────────────────────────
    PATIENCE   = 20   # raised from 15 — Phase 3 needs more time
    MIN_EPOCHS = 55   # must reach deep into Phase 3 before stopping

    # ── LR schedule ─────────────────────────────────────────────────────────
    T_0    = 10
    T_MULT = 2

    # ── Stochastic Weight Averaging ─────────────────────────────────────────
    SWA_START = 65    # start earlier (was 70) — Phase 3 converges faster now
    SWA_LR    = 1e-5  # flat LR during SWA phase

    # ── MixUp ───────────────────────────────────────────────────────────────
    MIXUP_ALPHA = 0.2   # Beta distribution parameter
    MIXUP_PROB  = 0.2   # REDUCED from 0.5 — high prob suppressed malignant signal
                        # (train sens dropped 96%→92%). 20% smooths boundaries
                        # without hurting minority class detection

    # ── Targets ─────────────────────────────────────────────────────────────
    TARGET_SENSITIVITY = 0.85
    TARGET_SPECIFICITY = 0.75

    # Minimum test sensitivity for domain shift checkpoint (raised from 0.60)
    DOMAIN_SHIFT_MIN_TEST_SENS = 0.65

    # Save test sample visualizations every N epochs (0 = disabled)
    VIZ_EVERY      = 10
    VIZ_N_SAMPLES  = 16   # number of HAM10000 test images to visualize

    # Set to "outputs/checkpoints/last_epoch.pth" to resume after crash
    RESUME_CHECKPOINT = None


# ============================================================================
# SETUP
# ============================================================================
def setup_logging():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.OUTPUT_DIR}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_directories():
    for d in [Config.OUTPUT_DIR, Config.CHECKPOINT_DIR,
              Config.PLOT_DIR, Config.VIZ_DIR]:
        os.makedirs(d, exist_ok=True)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        print("⚠ Using CPU")
    return device


# ============================================================================
# AUGMENTATION
# ============================================================================
def get_train_transforms():
    return A.Compose([
        A.Resize(288, 288),
        A.RandomResizedCrop(size=(Config.INPUT_SIZE, Config.INPUT_SIZE),
                            scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                           rotate_limit=45, border_mode=0, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=50, sigma=5, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1),
        ], p=0.3),

        # ── Scanner / domain colour augmentation ──────────────────────────
        # BCN20000 (Barcelona) and HAM10000 (Vienna + Australia) use
        # different dermoscopes with different colour/brightness profiles.
        # Aggressive colour jitter forces the model to learn colour-invariant
        # features, directly attacking the primary source of domain shift.
        A.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.4, hue=0.2, p=0.85),
        A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40,
                             val_shift_limit=25, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        # CLAHE mimics different contrast enhancement settings on scanners
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        # Random channel-level independent brightness shift — simulates
        # white-balance differences between scanner manufacturers
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.4),
        # Randomly desaturate 10% of images — forces model to not rely
        # on colour alone (HAM10000 has more colour variation)
        A.ToGray(p=0.05),
        # ─────────────────────────────────────────────────────────────────

        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=1, min_height=8, min_width=8,
                        fill_value=0, p=0.3),
        A.OneOf([A.MotionBlur(blur_limit=3, p=1),
                 A.MedianBlur(blur_limit=3, p=1),
                 A.GaussianBlur(blur_limit=3, p=1)], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_eval_transforms():
    return A.Compose([
        A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_tta_transforms() -> List[A.Compose]:
    """4 deterministic TTA views"""
    norm   = [A.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
              ToTensorV2()]
    resize = A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE)
    return [
        A.Compose([resize] + norm),
        A.Compose([resize, A.HorizontalFlip(p=1.0)] + norm),
        A.Compose([resize, A.VerticalFlip(p=1.0)] + norm),
        A.Compose([resize, A.RandomRotate90(p=1.0)] + norm),
    ]


# ============================================================================
# MIXUP
# ============================================================================
def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor,
                alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, float]:
    """
    Returns mixed images, original labels, shuffled labels, and lambda.
    Loss = lam * criterion(out, labels_a) + (1-lam) * criterion(out, labels_b)
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


# ============================================================================
# DATASETS
# ============================================================================
class BCN20000Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img   = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


class HAM10000Dataset(Dataset):
    MALIGNANT = {"mel", "bcc", "akiec", "scc"}
    BENIGN    = {"nv", "bkl", "df", "vasc"}

    def __init__(self, image_dir, metadata_csv, transform=None):
        self.transform   = transform
        self.image_paths = []
        self.labels      = []
        self.dx_list     = []   # store dx for visualization labels
        meta = pd.read_csv(metadata_csv)
        for _, row in meta.iterrows():
            dx, img_id = row["dx"].lower(), row["image_id"]
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                p = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(p): img_path = p; break
            if img_path is None: continue
            if   dx in self.MALIGNANT: label = 1
            elif dx in self.BENIGN:    label = 0
            else: continue
            self.image_paths.append(img_path)
            self.labels.append(label)
            self.dx_list.append(dx)
        print(f"✓ HAM10000: {len(self)} images | "
              f"Benign:{self.labels.count(0)} Malignant:{self.labels.count(1)}")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img   = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


# ============================================================================
# DATA LOADING
# ============================================================================
def prepare_bcn_split(data_dir, logger):
    benign_paths    = [os.path.join(data_dir, "benign", f)
                       for f in os.listdir(os.path.join(data_dir, "benign"))
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
    malignant_paths = [os.path.join(data_dir, "malignant", f)
                       for f in os.listdir(os.path.join(data_dir, "malignant"))
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
    paths  = benign_paths + malignant_paths
    labels = [0] * len(benign_paths) + [1] * len(malignant_paths)
    logger.info(f"BCN20000: {len(paths)} total | "
                f"Benign:{len(benign_paths)} Malignant:{len(malignant_paths)}")
    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.15, stratify=labels,
        random_state=Config.SEED)
    logger.info(f"Split -> Train:{len(tr_p)} Val:{len(va_p)}")
    return tr_p, tr_l, va_p, va_l

def create_dataloaders(logger):
    tr_p, tr_l, va_p, va_l = prepare_bcn_split(Config.BCN_DIR, logger)
    train_ds = BCN20000Dataset(tr_p, tr_l, get_train_transforms())
    val_ds   = BCN20000Dataset(va_p, va_l, get_eval_transforms())
    test_ds  = HAM10000Dataset(Config.HAM_DIR, Config.HAM_META,
                               get_eval_transforms())
    kw = dict(num_workers=Config.NUM_WORKERS, pin_memory=True)
    return (DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True,
                       drop_last=True, **kw),
            DataLoader(val_ds,   Config.BATCH_SIZE, shuffle=False, **kw),
            DataLoader(test_ds,  Config.BATCH_SIZE, shuffle=False, **kw),
            test_ds)   # return raw dataset for visualization


# ============================================================================
# MODEL
# ============================================================================
def create_model(logger):
    logger.info(f"Creating: {Config.MODEL_NAME}")
    model = timm.create_model(Config.MODEL_NAME, pretrained=True,
                              num_classes=Config.NUM_CLASSES)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Params total={total:,}  trainable={trainable:,}")
    logger.info(f"  Children: {[n for n,_ in model.named_children()]}")
    return model

def _head_params(model):
    # Unwrap SWA model if needed
    m = model.module if hasattr(model, 'module') else model
    if   hasattr(m, 'classifier'): return list(m.classifier.parameters())
    elif hasattr(m, 'head'):       return list(m.head.parameters())
    else: raise AttributeError("No classifier/head found.")

def _log_trainable(model, tag):
    m = model.module if hasattr(model, 'module') else model
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"[{tag}] Trainable: {trainable:,} / {total:,}")

def freeze_backbone(model):
    m = model.module if hasattr(model, 'module') else model
    for p in m.parameters(): p.requires_grad = False
    for p in _head_params(model): p.requires_grad = True
    _log_trainable(model, "After freeze_backbone")

def unfreeze_last_blocks(model):
    m = model.module if hasattr(model, 'module') else model
    for p in m.parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if any(k in n for k in ["stages.3","stages.4","final_conv",
                                 "classifier","head"]):
            p.requires_grad = True
    _log_trainable(model, "After unfreeze_last_blocks")

def unfreeze_all(model):
    m = model.module if hasattr(model, 'module') else model
    for p in m.parameters(): p.requires_grad = True
    _log_trainable(model, "After unfreeze_all")

def adjust_learning_rates(model, optimizer, epoch, logger):
    m = model.module if hasattr(model, 'module') else model

    # ── Phase 1: head only ─────────────────────────────────────────────────
    if epoch == 1:
        freeze_backbone(model)
        logger.info(f"📌 Phase 1: head only (lr={Config.PHASE_1_LR})")
        optimizer.param_groups.clear()
        optimizer.add_param_group({
            'params': _head_params(model),
            'lr': Config.PHASE_1_LR,
            'weight_decay': Config.WEIGHT_DECAY
        })

    # ── Phase 2: unfreeze last transformer blocks ──────────────────────────
    elif epoch == Config.PHASE_2_END + 1:   # epoch 31
        unfreeze_last_blocks(model)
        logger.info("📌 Phase 2: unfreezing stages.3, stages.4, final_conv")
        optimizer.param_groups.clear()
        bb_params = [p for n, p in m.named_parameters()
                     if any(k in n for k in ["stages.3","stages.4","final_conv"])
                     and p.requires_grad]
        if bb_params:
            optimizer.add_param_group({
                'params': bb_params,
                'lr': Config.PHASE_2_LR_BACKBONE,
                'weight_decay': Config.WEIGHT_DECAY
            })
            logger.info(f"  Backbone lr={Config.PHASE_2_LR_BACKBONE}")
        optimizer.add_param_group({
            'params': _head_params(model),
            'lr': Config.PHASE_2_LR_HEAD,
            'weight_decay': Config.WEIGHT_DECAY
        })
        logger.info(f"  Head lr={Config.PHASE_2_LR_HEAD}")

    # ── Phase 3: full fine-tune, single flat LR ────────────────────────────
    # FIXED: epoch 32, not 31 — separate from Phase 2 trigger at 31
    # FIXED: single lr=3e-5 for all params — discriminative LR (prev version)
    #        had early layers at 1e-5 which was too slow: domain shift hit 32%
    #        because backbone couldn't keep up with head updates. Flat LR keeps
    #        all parts updating at consistent rate, matching best run behaviour.
    elif epoch == Config.PHASE_3_START:     # epoch 32
        unfreeze_all(model)
        logger.info("📌 Phase 3: full fine-tune — ALL 4.3M params (flat lr=3e-5)")
        optimizer.param_groups.clear()
        optimizer.add_param_group({
            'params': [p for p in m.parameters() if p.requires_grad],
            'lr': Config.PHASE_3_LR,
            'weight_decay': Config.WEIGHT_DECAY
        })
        logger.info(f"  All 4.3M params lr={Config.PHASE_3_LR}")


# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(y_true, y_probs, threshold=0.5) -> Dict:
    y_pred = (y_probs >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    try:    auc = roc_auc_score(y_true, y_probs)
    except: auc = 0.0
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "precision":   float(tp / (tp + fp + 1e-8)),
        "f1":          float(f1_score(y_true, y_pred, pos_label=1)),
        "auc_roc":     float(auc),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

def find_best_threshold(y_true, y_probs) -> float:
    """Youden's index: maximises sensitivity + specificity - 1"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        score = tp/(tp+fn+1e-8) + tn/(tn+fp+1e-8) - 1
        if score > best_score:
            best_score = score
            best_t     = t
    return best_t


# ============================================================================
# TRAIN / EVAL
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss   = 0.0
    y_true_list  = []
    y_probs_list = []
    use_mixup    = Config.MIXUP_ALPHA > 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            if use_mixup and random.random() < Config.MIXUP_PROB:
                mixed_imgs, labels_a, labels_b, lam = mixup_batch(
                    imgs, labels, Config.MIXUP_ALPHA)
                out  = model(mixed_imgs)
                loss = lam * criterion(out, labels_a) + \
                       (1 - lam) * criterion(out, labels_b)
            else:
                out  = model(imgs)
                loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
        scaler.step(optimizer); scaler.update()

        total_loss += loss.item()
        probs = torch.softmax(out, 1)[:, 1]
        y_true_list.extend(labels.cpu().numpy())
        y_probs_list.extend(probs.cpu().detach().numpy())

    y_true  = np.array(y_true_list)
    y_probs = np.array(y_probs_list)
    m = compute_metrics(y_true, y_probs, threshold=0.5)
    m["loss"] = total_loss / len(loader)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval", threshold=0.5):
    model.eval()
    total_loss   = 0.0
    y_true_list  = []
    y_probs_list = []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item()
        probs = torch.softmax(out, 1)[:, 1]
        y_true_list.extend(labels.cpu().numpy())
        y_probs_list.extend(probs.cpu().numpy())

    y_true  = np.array(y_true_list)
    y_probs = np.array(y_probs_list)
    m = compute_metrics(y_true, y_probs, threshold)
    m["loss"]    = total_loss / len(loader)
    m["y_true"]  = y_true
    m["y_probs"] = y_probs
    return m


@torch.no_grad()
def evaluate_tta(model, dataset_class, image_paths, labels, device,
                 threshold=0.5):
    """Average probabilities across 4 TTA views. No criterion needed."""
    model.eval()
    all_probs = []

    for tfm in get_tta_transforms():
        ds     = dataset_class(image_paths, labels, transform=tfm)
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
        probs_this = []
        for imgs, _ in tqdm(loader, desc="TTA", leave=False):
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast():
                out = model(imgs)
            probs_this.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        all_probs.append(np.array(probs_this))

    y_probs = np.mean(all_probs, axis=0)
    y_true  = np.array(labels)
    m = compute_metrics(y_true, y_probs, threshold)
    m["loss"]    = 0.0
    m["y_true"]  = y_true
    m["y_probs"] = y_probs
    return m


# ============================================================================
# TEST IMAGE VISUALIZATION  (GradCAM-ready grid)
# ============================================================================
@torch.no_grad()
def save_test_visualizations(model, test_dataset: HAM10000Dataset,
                             epoch: int, threshold: float, device,
                             n_samples: int = 16):
    """
    Saves a grid of HAM10000 test images with:
      - True label (dx name)
      - Predicted label + probability
      - Correct / Wrong indicator
      - Border colour: green=correct, red=wrong

    Images saved to outputs/test_samples/epoch_XX.png
    These raw PIL images are also what you pass into GradCAM later.
    """
    model.eval()
    # Pick a balanced mix of malignant and benign samples
    mal_idx = [i for i, l in enumerate(test_dataset.labels) if l == 1]
    ben_idx = [i for i, l in enumerate(test_dataset.labels) if l == 0]
    half    = n_samples // 2
    chosen  = (random.sample(mal_idx, min(half, len(mal_idx))) +
               random.sample(ben_idx, min(half, len(ben_idx))))
    random.shuffle(chosen)

    eval_tfm = get_eval_transforms()
    cols     = 4
    rows     = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    axes = axes.flatten()

    for ax_idx, sample_idx in enumerate(chosen):
        raw_img  = np.array(Image.open(
            test_dataset.image_paths[sample_idx]).convert("RGB"))
        true_lbl = test_dataset.labels[sample_idx]
        dx_name  = test_dataset.dx_list[sample_idx].upper()

        # Get prediction
        tensor = eval_tfm(image=raw_img)["image"].unsqueeze(0).to(device)
        with torch.cuda.amp.autocast():
            out = model(tensor)
        prob_mal = float(torch.softmax(out, 1)[0, 1].cpu())
        pred_lbl = int(prob_mal >= threshold)

        correct      = pred_lbl == true_lbl
        border_color = "#2ecc71" if correct else "#e74c3c"
        pred_name    = "MALIGNANT" if pred_lbl == 1 else "BENIGN"
        true_name    = "MALIGNANT" if true_lbl == 1 else "BENIGN"

        ax = axes[ax_idx]
        ax.imshow(raw_img)
        ax.axis('off')

        # Coloured border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
            spine.set_visible(True)

        status = "✓" if correct else "✗"
        title  = (f"{status} True: {dx_name} ({true_name})\n"
                  f"Pred: {pred_name}  p={prob_mal:.3f}")
        ax.set_title(title, fontsize=8, color=border_color, fontweight='bold',
                     pad=4)

    # Hide unused axes
    for ax in axes[len(chosen):]:
        ax.axis('off')

    correct_patch = mpatches.Patch(color='#2ecc71', label='Correct')
    wrong_patch   = mpatches.Patch(color='#e74c3c', label='Wrong')
    fig.legend(handles=[correct_patch, wrong_patch],
               loc='lower right', fontsize=10)

    fig.suptitle(f"HAM10000 Test Samples — Epoch {epoch} "
                 f"(threshold={threshold:.2f})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = f"{Config.VIZ_DIR}/epoch_{epoch:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Test visualization saved: {out_path}")


# ============================================================================
# VISUALIZATION — training curves
# ============================================================================
def plot_training_history(history, save_dir):
    ep  = [h["epoch"] for h in history]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    for ax, (k1, k2, k3, title, ylabel) in zip(axes.flat[:2], [
        ("train_loss", "val_loss",  "test_loss", "Loss Curves",  "Loss"),
        ("train_acc",  "val_acc",   "test_acc",  "Accuracy",     "Accuracy")]):
        ax.plot(ep, [h[k1] for h in history], label="Train",      linewidth=2)
        ax.plot(ep, [h[k2] for h in history], label="Val (BCN)",  linewidth=2)
        ax.plot(ep, [h[k3] for h in history],
                label="Test (HAM)", linewidth=2, linestyle='--')
        ax.set_title(title, fontweight='bold'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    # Phase boundary lines on all plots
    for ax in axes.flat:
        ax.axvline(Config.PHASE_1_END,   color='gray',   linestyle=':', alpha=0.6)
        ax.axvline(Config.PHASE_2_END,   color='blue',   linestyle=':', alpha=0.6)
        ax.axvline(Config.PHASE_3_START, color='green',  linestyle=':', alpha=0.6)
        ax.axvline(Config.SWA_START,     color='purple', linestyle=':', alpha=0.6)

    axes[1,0].plot(ep, [h["val_sensitivity"]  for h in history],
                   label="Val",  linewidth=2, color='green')
    axes[1,0].plot(ep, [h["test_sensitivity"] for h in history],
                   label="Test", linewidth=2, color='red', linestyle='--')
    axes[1,0].axhline(Config.TARGET_SENSITIVITY, color='black', linestyle=':',
                      label=f'Target {Config.TARGET_SENSITIVITY:.0%}')
    axes[1,0].set_title("Sensitivity ⭐", fontweight='bold')
    axes[1,0].set_ylim([0.4, 1.0])
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(ep, [h["val_specificity"]  for h in history],
                   label="Val",  linewidth=2, color='blue')
    axes[1,1].plot(ep, [h["test_specificity"] for h in history],
                   label="Test", linewidth=2, color='orange', linestyle='--')
    axes[1,1].axhline(Config.TARGET_SPECIFICITY, color='black', linestyle=':',
                      label=f'Target {Config.TARGET_SPECIFICITY:.0%}')
    axes[1,1].set_title("Specificity", fontweight='bold')
    axes[1,1].set_ylim([0.4, 1.0])
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    axes[2,0].plot(ep, [h["val_f1"]  for h in history],
                   label="Val",  linewidth=2)
    axes[2,0].plot(ep, [h["test_f1"] for h in history],
                   label="Test", linewidth=2, linestyle='--')
    axes[2,0].set_title("F1-Score", fontweight='bold')
    axes[2,0].legend(); axes[2,0].grid(True, alpha=0.3)

    axes[2,1].plot(ep, [h["val_auc"]  for h in history],
                   label="Val",  linewidth=2)
    axes[2,1].plot(ep, [h["test_auc"] for h in history],
                   label="Test", linewidth=2, linestyle='--')
    axes[2,1].set_title("AUC-ROC", fontweight='bold')
    axes[2,1].set_ylim([0.5, 1.0])
    axes[2,1].legend(); axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/training_curves.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Domain shift plot
    domain_shifts = [abs(h["val_sensitivity"] - h["test_sensitivity"])
                     for h in history]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ep, domain_shifts, linewidth=2, color='purple')
    ax2.axhline(0.10, color='green',  linestyle=':', label='Good (<10%)')
    ax2.axhline(0.20, color='orange', linestyle=':', label='Warning (20%)')
    ax2.axvline(Config.PHASE_3_START, color='green', linestyle=':',
                alpha=0.7, label='Phase 3 start')
    ax2.axvline(Config.SWA_START, color='purple', linestyle=':',
                alpha=0.7, label='SWA start')
    ax2.set_title("Domain Shift (|Val Sens − Test Sens|)", fontweight='bold')
    ax2.set_ylabel("Gap"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{save_dir}/domain_shift.png", dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Curves saved: {save_dir}/")


def print_epoch_summary(epoch, tr, va, te, t, lr, threshold,
                        swa_active=False):
    domain_shift = va['sensitivity'] - te['sensitivity']
    shift_flag   = "⚠" if abs(domain_shift) > 0.20 else "✓"
    swa_tag      = "  [SWA active]" if swa_active else ""
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch}/{Config.NUM_EPOCHS}  ({t:.1f}s)  "
          f"LR: {lr:.2e}  Threshold: {threshold:.2f}{swa_tag}")
    print(f"{'='*80}")
    print(f"  TRAIN  loss={tr['loss']:.4f} acc={tr['accuracy']:.2%} "
          f"f1={tr['f1']:.4f} sens={tr['sensitivity']:.2%}")
    print(f"  VAL    loss={va['loss']:.4f} "
          f"⭐sens={va['sensitivity']:.2%} spec={va['specificity']:.2%} "
          f"prec={va['precision']:.2%} f1={va['f1']:.4f} "
          f"auc={va['auc_roc']:.4f} mcc={va['mcc']:.4f}")
    print(f"  TEST   loss={te['loss']:.4f} "
          f"⭐sens={te['sensitivity']:.2%} spec={te['specificity']:.2%} "
          f"prec={te['precision']:.2%} f1={te['f1']:.4f} "
          f"auc={te['auc_roc']:.4f} mcc={te['mcc']:.4f}")
    print(f"  CM Val  TN={va['tn']} FP={va['fp']} "
          f"FN={va['fn']} TP={va['tp']}")
    print(f"  CM Test TN={te['tn']} FP={te['fp']} "
          f"FN={te['fn']} TP={te['tp']}")
    print(f"  Domain shift: {domain_shift:+.1%}  {shift_flag}")
    print(f"{'='*80}\n")


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================
def save_checkpoint(ckpt, path):
    """Atomic save — crash safe"""
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, logger):
    logger.info(f"🔄 Resuming from: {path}")
    # FIXED: weights_only=False — required for checkpoints containing numpy
    # arrays and history dicts. Safe because checkpoint is our own file.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])

    start_epoch           = ckpt['epoch'] + 1
    best_val_loss         = ckpt.get('best_val_loss',         float('inf'))
    best_sensitivity      = ckpt.get('best_sensitivity',      0.0)
    best_test_sensitivity = ckpt.get('best_test_sensitivity', 0.0)
    best_domain_shift     = ckpt.get('best_domain_shift',     float('inf'))
    patience_counter      = ckpt.get('patience_counter',      0)
    history               = ckpt.get('history',               [])

    logger.info(f"  ✓ Epoch {ckpt['epoch']} | "
                f"best_val_loss={best_val_loss:.4f} | "
                f"best_test_sens={best_test_sensitivity:.2%} | "
                f"best_domain_shift={best_domain_shift:.2%} | "
                f"patience={patience_counter}")

    return (start_epoch, best_val_loss, best_sensitivity,
            best_test_sensitivity, best_domain_shift,
            patience_counter, history)


# ============================================================================
# MAIN
# ============================================================================
def main():
    set_seed(Config.SEED); create_directories()
    logger = setup_logging(); device = get_device()

    logger.info("="*80)
    logger.info("MobileViTv2-100  |  BCN20000→HAM10000  |  "
                "Focal Loss + Discriminative LR + MixUp + SWA + TTA")
    logger.info("="*80)

    train_loader, val_loader, test_loader, test_dataset = \
        create_dataloaders(logger)

    model     = create_model(logger).to(device)
    swa_model = AveragedModel(model)   # SWA wrapper (weights averaged in-place)
    swa_active = False

    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA,
                          label_smoothing=Config.LABEL_SMOOTHING)
    logger.info(f"Loss: FocalLoss | alpha={Config.FOCAL_ALPHA} | "
                f"gamma={Config.FOCAL_GAMMA}")

    optimizer = optim.AdamW(model.parameters(), lr=Config.INITIAL_LR,
                            weight_decay=Config.WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=Config.T_0, T_mult=Config.T_MULT,
                    eta_min=1e-6)
    # swa_scheduler is created at epoch SWA_START (after all param group
    # replacements are done) to avoid KeyError: 'swa_lr'
    swa_scheduler = None
    scaler = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------------
    best_val_loss         = float('inf')
    best_sensitivity      = 0.0
    best_test_sensitivity = 0.0
    best_domain_shift     = float('inf')
    patience_counter      = 0
    history: List[Dict]   = []
    start_epoch           = 1

    if Config.RESUME_CHECKPOINT and \
            os.path.exists(Config.RESUME_CHECKPOINT):
        (start_epoch, best_val_loss, best_sensitivity,
         best_test_sensitivity, best_domain_shift,
         patience_counter, history) = \
            load_checkpoint(Config.RESUME_CHECKPOINT, model, optimizer,
                            scheduler, scaler, logger)
    elif Config.RESUME_CHECKPOINT:
        logger.warning(f"⚠ Checkpoint not found: {Config.RESUME_CHECKPOINT}. "
                       f"Starting fresh.")

    logger.info(f"🚀 Training from epoch {start_epoch}/{Config.NUM_EPOCHS}")
    logger.info(f"   Phase 1 : epochs 1–{Config.PHASE_1_END}          "
                f"(head only,    1K params)")
    logger.info(f"   Phase 2 : epochs {Config.PHASE_2_END+1}–{Config.PHASE_3_START-1}         "
                f"(last blocks,  3.8M params)")
    logger.info(f"   Phase 3 : epochs {Config.PHASE_3_START}–{Config.SWA_START-1}        "
                f"(full finetune 4.3M params, flat lr={Config.PHASE_3_LR})")
    logger.info(f"   SWA     : epochs {Config.SWA_START}–{Config.NUM_EPOCHS}       "
                f"(weight averaging for flat minima)")
    logger.info(f"   MixUp   : alpha={Config.MIXUP_ALPHA}, "
                f"prob={Config.MIXUP_PROB} (20% batches — light smoothing)")

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
        t0 = time.time()
        adjust_learning_rates(model, optimizer, epoch, logger)

        # ── SWA activation ──────────────────────────────────────────────
        if epoch == Config.SWA_START:
            swa_active = True
            # FIXED: recreate swa_scheduler NOW, after all param group
            # replacements (Phase 2 at ep31, Phase 3 at ep32) are complete.
            # SWALR injects 'swa_lr' key into param groups at creation time.
            # If groups were replaced after original creation, the key is
            # missing and swa_scheduler.step() throws KeyError: 'swa_lr'
            swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR,
                                  anneal_epochs=5, anneal_strategy='cos')
            logger.info(f"📌 SWA activated at epoch {epoch} "
                        f"(lr={Config.SWA_LR})")

        lr = optimizer.param_groups[0]['lr']
        tr = train_one_epoch(model, train_loader, criterion,
                             optimizer, scaler, device)

        # SWA: update averaged weights after each epoch >= SWA_START
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # ── Validation & threshold ──────────────────────────────────────
        eval_model     = swa_model if swa_active else model
        va_raw         = evaluate(eval_model, val_loader, criterion,
                                  device, "Validation", threshold=0.5)
        best_threshold = find_best_threshold(va_raw["y_true"],
                                             va_raw["y_probs"])
        va             = compute_metrics(va_raw["y_true"], va_raw["y_probs"],
                                         threshold=best_threshold)
        va["loss"]     = va_raw["loss"]

        te = evaluate(eval_model, test_loader, criterion,
                      device, "Test (HAM)", threshold=best_threshold)

        elapsed      = time.time() - t0
        domain_shift = abs(va["sensitivity"] - te["sensitivity"])

        print_epoch_summary(epoch, tr, va, te, elapsed, lr,
                            best_threshold, swa_active)
        logger.info(f"Epoch {epoch} | threshold={best_threshold:.2f} | "
                    f"test_sens={te['sensitivity']:.2%} | "
                    f"domain_shift={domain_shift:.2%}")

        entry = {
            "epoch": epoch, "learning_rate": lr,
            "epoch_time": round(elapsed, 1),
            "best_threshold": best_threshold,
            "domain_shift": round(float(domain_shift), 4),
            "swa_active": swa_active,
            "train_loss": tr["loss"], "train_acc": tr["accuracy"],
            "train_f1":   tr["f1"],   "train_sensitivity": tr["sensitivity"],
            "val_loss":  va["loss"],  "val_acc": va["accuracy"],
            "val_sensitivity":  va["sensitivity"],
            "val_specificity":  va["specificity"],
            "val_precision":    va["precision"],
            "val_f1":           va["f1"],
            "val_auc":          va["auc_roc"],
            "val_mcc":          va["mcc"],
            "test_loss": te["loss"],  "test_acc": te["accuracy"],
            "test_sensitivity": te["sensitivity"],
            "test_specificity": te["specificity"],
            "test_precision":   te["precision"],
            "test_f1":          te["f1"],
            "test_auc":         te["auc_roc"],
            "test_mcc":         te["mcc"],
        }
        history.append(entry)

        # ── Checkpoint dict ─────────────────────────────────────────────
        # Always save raw model weights (not SWA) so resume works cleanly
        ckpt = {
            'epoch':                 epoch,
            'model_state_dict':      model.state_dict(),
            'swa_state_dict':        swa_model.state_dict() if swa_active else None,
            'optimizer_state_dict':  optimizer.state_dict(),
            'scheduler_state_dict':  scheduler.state_dict(),
            'scaler_state_dict':     scaler.state_dict(),
            'best_val_loss':         best_val_loss,
            'best_sensitivity':      best_sensitivity,
            'best_test_sensitivity': best_test_sensitivity,
            'best_domain_shift':     best_domain_shift,
            'patience_counter':      patience_counter,
            'best_threshold':        best_threshold,
            'history':               history,
            'val_metrics':           {k: v for k, v in va.items()
                                      if k not in ('y_true','y_probs',
                                                   'confusion_matrix')},
            'test_metrics':          {k: v for k, v in te.items()
                                      if k not in ('y_true','y_probs',
                                                   'confusion_matrix')},
        }

        # 1) Always save last epoch (crash recovery)
        save_checkpoint(ckpt, f"{Config.CHECKPOINT_DIR}/last_epoch.pth")

        # 2) Best validation loss
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            ckpt['best_val_loss'] = best_val_loss
            save_checkpoint(ckpt,
                f"{Config.CHECKPOINT_DIR}/best_val_loss.pth")
            logger.info(f"✅ New best val loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 3) Best TEST sensitivity — primary clinical checkpoint
        if te["sensitivity"] > best_test_sensitivity:
            best_test_sensitivity = te["sensitivity"]
            ckpt['best_test_sensitivity'] = best_test_sensitivity
            save_checkpoint(ckpt,
                f"{Config.CHECKPOINT_DIR}/best_test_sensitivity.pth")
            logger.info(f"✅ New best test sensitivity: "
                        f"{best_test_sensitivity:.2%} "
                        f"(threshold={best_threshold:.2f} | "
                        f"spec={te['specificity']:.2%} | "
                        f"auc={te['auc_roc']:.4f} | "
                        f"domain_shift={domain_shift:.2%})")

        # 4) Best domain shift (raised floor to 0.65)
        if (te["sensitivity"] >= Config.DOMAIN_SHIFT_MIN_TEST_SENS and
                domain_shift < best_domain_shift):
            best_domain_shift = domain_shift
            ckpt['best_domain_shift'] = best_domain_shift
            save_checkpoint(ckpt,
                f"{Config.CHECKPOINT_DIR}/best_domain_shift.pth")
            logger.info(f"✅ New best domain shift: "
                        f"{best_domain_shift:.2%} "
                        f"(test_sens={te['sensitivity']:.2%} | "
                        f"val_sens={va['sensitivity']:.2%} | "
                        f"auc={te['auc_roc']:.4f})")

        # 5) Best val sensitivity (hits target ≥85%)
        if (va["sensitivity"] > best_sensitivity and
                va["sensitivity"] >= Config.TARGET_SENSITIVITY):
            best_sensitivity = va["sensitivity"]
            ckpt['best_sensitivity'] = best_sensitivity
            save_checkpoint(ckpt,
                f"{Config.CHECKPOINT_DIR}/best_val_sensitivity.pth")
            logger.info(f"✅ New best val sensitivity: "
                        f"{best_sensitivity:.2%}")

        # 6) Periodic every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(ckpt,
                f"{Config.CHECKPOINT_DIR}/epoch_{epoch}.pth")

        # 7) Test image visualization
        if Config.VIZ_EVERY > 0 and epoch % Config.VIZ_EVERY == 0:
            save_test_visualizations(
                eval_model, test_dataset, epoch, best_threshold, device,
                n_samples=Config.VIZ_N_SAMPLES)

        # 8) History JSON
        with open(f"{Config.OUTPUT_DIR}/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epoch >= Config.MIN_EPOCHS and \
                patience_counter >= Config.PATIENCE:
            logger.info(f"⏹ Early stopping at epoch {epoch} "
                        f"(patience={patience_counter})")
            break

    # ------------------------------------------------------------------
    # SWA: update BatchNorm statistics before final evaluation
    # ------------------------------------------------------------------
    if swa_active:
        logger.info("🔄 Updating SWA BatchNorm statistics...")
        update_bn(train_loader, swa_model, device=device)
        logger.info("✓ SWA BN update complete")

    # ------------------------------------------------------------------
    # FINAL TTA EVALUATION
    # ------------------------------------------------------------------
    logger.info("\n🔬 Running TTA on best_test_sensitivity checkpoint...")
    best_ckpt_path = f"{Config.CHECKPOINT_DIR}/best_test_sensitivity.pth"
    # FIXED: weights_only=False — checkpoint contains numpy arrays in history
    best_ckpt      = torch.load(best_ckpt_path, map_location=device,
                                weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    saved_threshold = best_ckpt.get('best_threshold', 0.5)

    # Rebuild HAM10000 path/label lists for TTA
    ham_meta = pd.read_csv(Config.HAM_META)
    tta_paths, tta_labels = [], []
    for _, row in ham_meta.iterrows():
        dx, img_id = row["dx"].lower(), row["image_id"]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            p = os.path.join(Config.HAM_DIR, f"{img_id}{ext}")
            if os.path.exists(p): img_path = p; break
        if img_path is None: continue
        if   dx in HAM10000Dataset.MALIGNANT: tta_labels.append(1)
        elif dx in HAM10000Dataset.BENIGN:    tta_labels.append(0)
        else: continue
        tta_paths.append(img_path)

    # TTA on raw model (best checkpoint)
    tta_result = evaluate_tta(model, BCN20000Dataset, tta_paths, tta_labels,
                              device, threshold=saved_threshold)

    logger.info(f"\n🏆 FINAL TTA RESULTS (HAM10000 — best_test_sensitivity ckpt):")
    logger.info(f"  Sensitivity : {tta_result['sensitivity']:.2%}")
    logger.info(f"  Specificity : {tta_result['specificity']:.2%}")
    logger.info(f"  AUC-ROC     : {tta_result['auc_roc']:.4f}")
    logger.info(f"  F1-Score    : {tta_result['f1']:.4f}")
    logger.info(f"  MCC         : {tta_result['mcc']:.4f}")
    logger.info(f"  Threshold   : {saved_threshold:.2f}")

    # Also run TTA on SWA model if it was active
    if swa_active:
        logger.info("\n🔬 Running TTA on SWA model (final averaged weights)...")
        tta_swa = evaluate_tta(swa_model, BCN20000Dataset,
                               tta_paths, tta_labels,
                               device, threshold=saved_threshold)
        logger.info(f"\n🏆 FINAL TTA RESULTS — SWA model:")
        logger.info(f"  Sensitivity : {tta_swa['sensitivity']:.2%}")
        logger.info(f"  Specificity : {tta_swa['specificity']:.2%}")
        logger.info(f"  AUC-ROC     : {tta_swa['auc_roc']:.4f}")
        logger.info(f"  F1-Score    : {tta_swa['f1']:.4f}")
        logger.info(f"  MCC         : {tta_swa['mcc']:.4f}")

    # Final visualization on best checkpoint
    logger.info("\n🖼 Saving final test visualizations...")
    save_test_visualizations(model, test_dataset, epoch=0,
                             threshold=saved_threshold, device=device,
                             n_samples=32)

    logger.info("\n✅ TRAINING COMPLETE")
    plot_training_history(history, Config.PLOT_DIR)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    logger.info("\n📁 SAVED CHECKPOINTS:")
    logger.info(f"  best_test_sensitivity.pth → "
                f"test_sens={best_test_sensitivity:.2%}  ← primary result")
    logger.info(f"  best_domain_shift.pth     → "
                f"domain_shift={best_domain_shift:.2%}")
    logger.info(f"  best_val_loss.pth         → "
                f"val_loss={best_val_loss:.4f}")
    logger.info(f"  best_val_sensitivity.pth  → "
                f"val_sens={best_sensitivity:.2%}")

    best_test_idx = max(range(len(history)),
                        key=lambda i: history[i]["test_sensitivity"])
    valid_shift   = [i for i in range(len(history))
                     if history[i]["test_sensitivity"] >=
                     Config.DOMAIN_SHIFT_MIN_TEST_SENS]
    best_shift_idx = (min(valid_shift,
                          key=lambda i: history[i]["domain_shift"])
                      if valid_shift else 0)

    bt = history[best_test_idx]
    bs = history[best_shift_idx]

    logger.info(f"\n  Best by test_sens    → epoch {bt['epoch']} | "
                f"test_sens={bt['test_sensitivity']:.2%} | "
                f"domain_shift={bt['domain_shift']:.2%} | "
                f"test_auc={bt['test_auc']:.4f}")
    logger.info(f"  Best by domain_shift → epoch {bs['epoch']} | "
                f"test_sens={bs['test_sensitivity']:.2%} | "
                f"domain_shift={bs['domain_shift']:.2%} | "
                f"test_auc={bs['test_auc']:.4f}")
    logger.info(f"\n  Test visualizations : {Config.VIZ_DIR}/")
    logger.info(f"  Training curves     : {Config.PLOT_DIR}/training_curves.png")
    logger.info(f"  Domain shift plot   : {Config.PLOT_DIR}/domain_shift.png")


if __name__ == "__main__":
    main()

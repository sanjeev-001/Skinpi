# Originally run on Kaggle; dataset paths must be adjusted to run locally.
import os, numpy as np, torch, torch.nn.functional as F
import timm, pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             cohen_kappa_score, average_precision_score,
                             roc_curve)
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT      = "/kaggle/input/models/sanjeevirajm/mobilevitv2-77/pytorch/default/1/best_test_sensitivity.pth"
THRESHOLD = 0.64
IMG_SIZE  = 256

BASE_DIR  = "/kaggle/input/datasets/sanjeevirajm/pad-ufes20/pad-ufes classified"

LABEL_MAP = {
    "benign"    : 0,
    "malignant" : 1,
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── LOAD MODEL ───────────────────────────────────────────────────────────
ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = timm.create_model("mobilevitv2_100", pretrained=False, num_classes=2)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval().to(DEVICE)
print("✓ Model loaded")

# ── BUILD DATASET FROM FOLDERS ───────────────────────────────────────────
records = []

for class_name, label in LABEL_MAP.items():
    folder = os.path.join(BASE_DIR, class_name)

    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            records.append({
                "path"      : os.path.join(folder, f),
                "image_id"  : os.path.splitext(f)[0],
                "class_name": class_name,
                "label"     : label
            })

df = pd.DataFrame(records)

print(f"PAD-UFES dataset loaded: {len(df)} images")
print(df["class_name"].value_counts().to_string())
print(f"\nBenign   : {(df['label']==0).sum()}")
print(f"Malignant: {(df['label']==1).sum()}")

# ── INFERENCE ────────────────────────────────────────────────────────────
print("\nRunning inference...")
all_probs, all_labels, all_ids, all_classes = [], [], [], []

with torch.no_grad():
    for _, row in df.iterrows():
        img  = Image.open(row["path"]).convert("RGB")
        inp  = transform(img).unsqueeze(0).to(DEVICE)
        prob = F.softmax(model(inp), dim=1)[0, 1].item()

        all_probs.append(prob)
        all_labels.append(row["label"])
        all_ids.append(row["image_id"])
        all_classes.append(row["class_name"])

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
preds      = (all_probs >= THRESHOLD).astype(int)

# ── METRICS ──────────────────────────────────────────────────────────────
tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

sensitivity  = tp / (tp + fn)
specificity  = tn / (tn + fp)
ppv          = tp / (tp + fp) if (tp+fp) > 0 else 0
npv          = tn / (tn + fn) if (tn+fn) > 0 else 0
f1           = 2*tp / (2*tp + fp + fn)
auc          = roc_auc_score(all_labels, all_probs)
ap           = average_precision_score(all_labels, all_probs)
mcc          = matthews_corrcoef(all_labels, preds)
kappa        = cohen_kappa_score(all_labels, preds)
bal_acc      = balanced_accuracy_score(all_labels, preds)
accuracy     = (tp + tn) / len(all_labels)

print("\n" + "="*55)
print("  PAD-UFES DATASET -- BATCH EVALUATION RESULTS")
print("  MobileViTv2-100 | BCN20000 trained")
print("="*55)
print(f"  Total images     : {len(df)}")
print(f"  Threshold        : {THRESHOLD}")
print("-"*55)
print(f"  Accuracy         : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Sensitivity      : {sensitivity:.4f}  ({sensitivity*100:.2f}%)")
print(f"  Specificity      : {specificity:.4f}  ({specificity*100:.2f}%)")
print(f"  Precision (PPV)  : {ppv:.4f}  ({ppv*100:.2f}%)")
print(f"  Neg Pred Value   : {npv:.4f}  ({npv*100:.2f}%)")
print(f"  F1-Score         : {f1:.4f}")
print(f"  AUC-ROC          : {auc:.4f}")
print(f"  PR-AUC           : {ap:.4f}")
print(f"  MCC              : {mcc:.4f}")
print(f"  Cohen's Kappa    : {kappa:.4f}")
print(f"  Balanced Accuracy: {bal_acc:.4f}  ({bal_acc*100:.2f}%)")
print("-"*55)
print(f"  Confusion Matrix :")
print(f"    TP={tp}  FP={fp}")
print(f"    FN={fn}  TN={tn}")
print("="*55)

# ── ROC CURVE ────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PAD-UFES")
plt.legend()
plt.grid()
plt.show()

# ── CROSS-DATASET SUMMARY ────────────────────────────────────────────────
print("\n" + "="*55)
print("  CROSS-DATASET GENERALISATION SUMMARY")
print("="*55)
print(f"  {'Dataset':<15} {'Sens':>8} {'Spec':>8} {'AUC':>8} {'N':>6}")
print("  " + "-"*45)
print(f"  {'BCN20000 (val)':<15} {'88.09%':>8} {'81.83%':>8} {'0.9281':>8} {'2047':>6}")
print(f"  {'HAM10000':<15} {'77.38%':>8} {'72.31%':>8} {'0.8318':>8} {'10015':>6}")
print(f"  {'PAD-UFES':<15} {sensitivity*100:>7.2f}% {specificity*100:>7.2f}% {auc:>8.4f} {len(df):>6}")
print("="*55)
print("Model trained on BCN20000 only.")
print("HAM10000 + PAD-UFES = unseen cross-dataset tests.")

import os, glob, numpy as np, torch, torch.nn.functional as F
import timm, pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             cohen_kappa_score, average_precision_score)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT      = "/kaggle/input/models/sanjeevirajm/mobilevitv2-77/pytorch/default/1/best_test_sensitivity.pth"
THRESHOLD = 0.64
IMG_SIZE  = 256
CAT_DIR   = "/kaggle/input/datasets/sanjeevirajm/ph2-dataset/Categories"

LABEL_MAP = {
    "melanoma"      : 1,
    "atypical_nevus": 0,
    "common_nevus"  : 0,
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Load model ────────────────────────────────────────────────────────────
ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = timm.create_model("mobilevitv2_100", pretrained=False, num_classes=2)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval().to(DEVICE)
print("✓ Model loaded")

# ── Build image list ──────────────────────────────────────────────────────
records = []
for class_name, label in LABEL_MAP.items():
    folder = os.path.join(CAT_DIR, class_name)
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".bmp", ".jpg", ".png", ".jpeg")):
            records.append({
                "path"      : os.path.join(folder, f),
                "image_id"  : os.path.splitext(f)[0],
                "class_name": class_name,
                "label"     : label
            })

df = pd.DataFrame(records)
print(f"PH2 dataset loaded: {len(df)} images")
print(df["class_name"].value_counts().to_string())
print(f"\nBenign   : {(df['label']==0).sum()}")
print(f"Malignant: {(df['label']==1).sum()}")

# ── Inference ─────────────────────────────────────────────────────────────
print("\nRunning inference...")
all_probs, all_labels, all_ids, all_classes = [], [], [], []

model.eval()
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

# ── Overall metrics ───────────────────────────────────────────────────────
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
print("  PH2 DATASET — BATCH EVALUATION RESULTS")
print("  MobileViTv2-100 | BCN20000 trained | Run 7")
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

# ── Per class breakdown ───────────────────────────────────────────────────
print("\n── Per-class breakdown ──────────────────────────────────")
print(f"  {'Class':<20} {'N':>4}  {'Correct':>8}  {'Accuracy':>10}  {'Mean Prob':>10}")
print("  " + "-"*58)
for class_name in ["melanoma", "atypical_nevus", "common_nevus"]:
    mask   = np.array(all_classes) == class_name
    lbl    = np.array(all_labels)[mask]
    pred_c = preds[mask]
    prob_c = all_probs[mask]
    correct = (pred_c == lbl).sum()
    n      = mask.sum()
    print(f"  {class_name:<20} {n:>4}  {correct:>7}  {correct/n:>10.1%}  {prob_c.mean():>10.3f}")

# ── Per image results ─────────────────────────────────────────────────────
print("\n── Per-image results ────────────────────────────────────")
print(f"  {'Image':<12} {'Class':<18} {'Prob':>6}  {'Pred':>10}  {'True':>8}  {'':>4}")
print("  " + "-"*65)
correct_count = 0
for i in range(len(all_ids)):
    pred     = preds[i]
    true     = all_labels[i]
    correct  = pred == true
    if correct: correct_count += 1
    print(f"  {all_ids[i]:<12} {all_classes[i]:<18} "
          f"{all_probs[i]:>6.3f}  "
          f"{'MALIGNANT' if pred==1 else 'BENIGN':>10}  "
          f"{'MALIGNANT' if true==1 else 'BENIGN':>8}  "
          f"{'✅' if correct else '❌'}")

print(f"\n  Overall: {correct_count}/{len(all_ids)} correct "
      f"({correct_count/len(all_ids)*100:.1f}%)")

# ── Cross dataset summary ─────────────────────────────────────────────────
print("\n" + "="*55)
print("  CROSS-DATASET GENERALISATION SUMMARY")
print("="*55)
print(f"  {'Dataset':<15} {'Sens':>8} {'Spec':>8} {'AUC':>8} {'N':>6}")
print("  " + "-"*45)
print(f"  {'BCN20000 (val)':<15} {'88.09%':>8} {'81.83%':>8} {'0.9281':>8} {'2047':>6}")
print(f"  {'HAM10000':<15} {'77.38%':>8} {'72.31%':>8} {'0.8318':>8} {'10015':>6}")
print(f"  {'PH2':<15} {sensitivity*100:>7.2f}% {specificity*100:>7.2f}% {auc:>8.4f} {len(df):>6}")
print("="*55)
print("Model trained on BCN20000 only.")
print("HAM10000 + PH2 = unseen cross-dataset tests.")

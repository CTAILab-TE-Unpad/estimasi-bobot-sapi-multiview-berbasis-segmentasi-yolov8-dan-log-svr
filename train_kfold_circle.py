import os
import json
import glob
import math
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Set PyTorch to use all available CPU cores
torch.set_num_threads(8)

def circle_to_polygon(center, point_on_perimeter, num_points=32):
    """Mengubah bentuk circle Labelme (center & radius point) menjadi N-titik polygon."""
    cx, cy = center
    px, py = point_on_perimeter
    radius = math.hypot(px - cx, py - cy)
    
    polygon = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        polygon.append([x, y])
    return polygon

def prepare_yolo_annotations_and_resized_images(image_dir, base_processed_dir, num_circle_points=32, target_dim=480):
    """
    Konversi seluruh file JSON Labelme ke format YOLOv8 segmentation 
    dan lakukan pre-resizing citra ke target_dim agar training CPU berjalan sangat cepat.
    """
    os.makedirs(base_processed_dir, exist_ok=True)
    images_out = os.path.join(base_processed_dir, "images")
    labels_out = os.path.join(base_processed_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    
    json_files = sorted(glob.glob(os.path.join(image_dir, "*.json")))
    print(f"[1/6] Mengonversi {len(json_files)} file JSON Labelme & Pre-resizing citra ke {target_dim}x{target_dim}...")
    
    valid_pairs = []
    for jf in json_files:
        base_name = os.path.splitext(os.path.basename(jf))[0]
        img_path = os.path.join(image_dir, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            continue
            
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        
        # Resize citra ke target_dim x target_dim
        resized_img = cv2.resize(img, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
        out_img_path = os.path.join(images_out, f"{base_name}.jpg")
        cv2.imwrite(out_img_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "").lower()
            if "sticker" not in label:
                continue
            shape_type = shape.get("shape_type", "circle")
            points = shape.get("points", [])
            
            if shape_type == "circle" and len(points) == 2:
                poly_pts = circle_to_polygon(points[0], points[1], num_points=num_circle_points)
            elif shape_type == "polygon" and len(points) >= 3:
                poly_pts = points
            else:
                continue
                
            norm_coords = []
            for pt in poly_pts:
                nx = max(0.0, min(1.0, pt[0] / orig_w))
                ny = max(0.0, min(1.0, pt[1] / orig_h))
                norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
                
            line = "0 " + " ".join(norm_coords)
            lines.append(line)
            
        if lines:
            txt_path = os.path.join(labels_out, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(lines) + "\n")
            valid_pairs.append((out_img_path, txt_path, img_path))
            
    print(f"-> Berhasil memproses {len(valid_pairs)} citra & label.")
    return valid_pairs

def setup_kfold_datasets(valid_pairs, kfold_base_dir, n_splits=5, seed=42):
    """Membagi dataset menjadi N-Fold dan membuat struktur direktori YOLO serta data.yaml."""
    print(f"[2/6] Membagi dataset menjadi {n_splits}-Fold Cross Validation (seed={seed})...")
    if os.path.exists(kfold_base_dir):
        shutil.rmtree(kfold_base_dir, ignore_errors=True)
    os.makedirs(kfold_base_dir, exist_ok=True)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_configs = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(valid_pairs)):
        fold_dir = os.path.join(kfold_base_dir, f"fold_{fold_idx}")
        train_img_dir = os.path.join(fold_dir, "images", "train")
        train_lbl_dir = os.path.join(fold_dir, "labels", "train")
        val_img_dir = os.path.join(fold_dir, "images", "val")
        val_lbl_dir = os.path.join(fold_dir, "labels", "val")
        
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Copy train files
        for idx in train_indices:
            img_p, lbl_p, _ = valid_pairs[idx]
            shutil.copy2(img_p, train_img_dir)
            shutil.copy2(lbl_p, train_lbl_dir)
            
        # Copy val files
        val_files = []
        for idx in val_indices:
            img_p, lbl_p, orig_img_p = valid_pairs[idx]
            shutil.copy2(img_p, val_img_dir)
            shutil.copy2(lbl_p, val_lbl_dir)
            val_files.append((os.path.join(val_img_dir, os.path.basename(img_p)),
                              os.path.join(val_lbl_dir, os.path.basename(lbl_p)),
                              orig_img_p))
            
        # Create data.yaml
        yaml_content = f"""path: {os.path.abspath(fold_dir).replace('\\', '/')}
train: images/train
val: images/val

names:
  0: sticker
"""
        yaml_path = os.path.join(fold_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yf.write(yaml_content)
            
        print(f"   Fold {fold_idx + 1}: {len(train_indices)} train images, {len(val_indices)} val images")
        fold_configs.append({
            "fold_idx": fold_idx,
            "yaml_path": yaml_path,
            "fold_dir": fold_dir,
            "val_files": val_files
        })
        
    return fold_configs

def train_kfold(fold_configs, pretrained_model_path, epochs=15, imgsz=480, batch=16):
    """Melatih model YOLOv8-Seg untuk setiap Fold dengan akselerasi CPU caching."""
    print(f"\n[3/6] Memulai Pelatihan 5-Fold (Epochs: {epochs}, Imgsz: {imgsz}, Batch: {batch}, Cache=RAM)...")
    
    results_summary = []
    best_overall_map = -1.0
    best_fold_model_path = None
    best_fold_idx = -1
    
    for cfg in fold_configs:
        fold_idx = cfg["fold_idx"]
        yaml_path = cfg["yaml_path"]
        print(f"\n==========================================")
        print(f"🚀 Training Fold {fold_idx + 1}/{len(fold_configs)}...")
        print(f"==========================================")
        
        # Load fresh pretrained weights for each fold
        model = YOLO(pretrained_model_path)
        
        run_name = f"circle_fold_{fold_idx}"
        train_res = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=0.005,
            device="cpu",
            workers=0,
            cache=True,
            project="runs/kfold_circle",
            name=run_name,
            exist_ok=True,
            verbose=False,
            seed=42
        )
        
        # Run validation
        val_res = model.val(data=yaml_path, imgsz=imgsz, split="val", verbose=False)
        
        # Extract metrics
        # Box metrics
        box_p = float(val_res.box.p[0]) if len(val_res.box.p) > 0 else 0.0
        box_r = float(val_res.box.r[0]) if len(val_res.box.r) > 0 else 0.0
        box_map50 = float(val_res.box.map50)
        box_map = float(val_res.box.map)
        
        # Mask metrics
        mask_p = float(val_res.seg.p[0]) if len(val_res.seg.p) > 0 else 0.0
        mask_r = float(val_res.seg.r[0]) if len(val_res.seg.r) > 0 else 0.0
        mask_map50 = float(val_res.seg.map50)
        mask_map = float(val_res.seg.map)
        
        weights_path = os.path.join("runs/kfold_circle", run_name, "weights", "best.pt")
        
        fold_record = {
            "Fold": f"Fold {fold_idx + 1}",
            "Box_Precision": round(box_p, 4),
            "Box_Recall": round(box_r, 4),
            "Box_mAP50": round(box_map50, 4),
            "Box_mAP50-95": round(box_map, 4),
            "Mask_Precision": round(mask_p, 4),
            "Mask_Recall": round(mask_r, 4),
            "Mask_mAP50": round(mask_map50, 4),
            "Mask_mAP50-95": round(mask_map, 4),
            "weights_path": weights_path,
            "run_dir": os.path.join("runs/kfold_circle", run_name)
        }
        results_summary.append(fold_record)
        
        print(f"📊 Fold {fold_idx + 1} Selesai:")
        print(f"   Box  -> P: {box_p:.4f}, R: {box_r:.4f}, mAP50: {box_map50:.4f}, mAP50-95: {box_map:.4f}")
        print(f"   Mask -> P: {mask_p:.4f}, R: {mask_r:.4f}, mAP50: {mask_map50:.4f}, mAP50-95: {mask_map:.4f}")
        
        if mask_map > best_overall_map:
            best_overall_map = mask_map
            best_fold_model_path = weights_path
            best_fold_idx = fold_idx
            
    print(f"\n🏆 Model Terbaik: Fold {best_fold_idx + 1} dengan Mask mAP50-95 = {best_overall_map:.4f}")
    return results_summary, best_fold_model_path, best_fold_idx

def generate_evaluation_plots(results_summary, output_dir, artifact_dir=None):
    """Membuat visualisasi perbandingan metrik across Folds dan loss curves."""
    print("\n[4/6] Menghasilkan grafik evaluasi performa K-Fold...")
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        
    df = pd.DataFrame(results_summary)
    
    # 1. Bar Chart Metrik Per Folds
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    folds = df["Fold"]
    x = np.arange(len(folds))
    width = 0.2
    
    # Plot Box Metrics
    axes[0].bar(x - 1.5*width, df["Box_Precision"], width, label="Precision", color="#3498db")
    axes[0].bar(x - 0.5*width, df["Box_Recall"], width, label="Recall", color="#2ecc71")
    axes[0].bar(x + 0.5*width, df["Box_mAP50"], width, label="mAP@0.50", color="#f1c40f")
    axes[0].bar(x + 1.5*width, df["Box_mAP50-95"], width, label="mAP@0.50:0.95", color="#e74c3c")
    axes[0].set_title("Bounding Box Detection Metrics across 5 Folds", fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds)
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Plot Mask Metrics
    axes[1].bar(x - 1.5*width, df["Mask_Precision"], width, label="Precision", color="#3498db")
    axes[1].bar(x - 0.5*width, df["Mask_Recall"], width, label="Recall", color="#2ecc71")
    axes[1].bar(x + 0.5*width, df["Mask_mAP50"], width, label="mAP@0.50", color="#f1c40f")
    axes[1].bar(x + 1.5*width, df["Mask_mAP50-95"], width, label="mAP@0.50:0.95", color="#e74c3c")
    axes[1].set_title("Instance Segmentation (Mask) Metrics across 5 Folds", fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(folds)
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    metrics_chart_path = os.path.join(output_dir, "kfold_metrics_comparison.png")
    fig.savefig(metrics_chart_path, dpi=200, bbox_inches="tight")
    if artifact_dir:
        fig.savefig(os.path.join(artifact_dir, "kfold_metrics_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Grafik metrik tersimpan di: {metrics_chart_path}")
    
    # 2. Plot Training Loss Curves across Folds
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22"]
    
    for idx, rec in enumerate(results_summary):
        csv_path = os.path.join(rec["run_dir"], "results.csv")
        if os.path.exists(csv_path):
            res_df = pd.read_csv(csv_path)
            res_df.columns = [c.strip() for c in res_df.columns]
            
            epoch = res_df["epoch"] if "epoch" in res_df.columns else range(len(res_df))
            if "val/seg_loss" in res_df.columns:
                axes[0].plot(epoch, res_df["val/seg_loss"], label=f"Fold {idx+1}", color=colors[idx % len(colors)], lw=2)
            elif "val/box_loss" in res_df.columns:
                axes[0].plot(epoch, res_df["val/box_loss"], label=f"Fold {idx+1}", color=colors[idx % len(colors)], lw=2)
                
            if "metrics/mAP50-95(M)" in res_df.columns:
                axes[1].plot(epoch, res_df["metrics/mAP50-95(M)"], label=f"Fold {idx+1}", color=colors[idx % len(colors)], lw=2)
            elif "metrics/mAP50(M)" in res_df.columns:
                axes[1].plot(epoch, res_df["metrics/mAP50(M)"], label=f"Fold {idx+1}", color=colors[idx % len(colors)], lw=2)
                
    axes[0].set_title("Validation Segmentation Loss across Epochs", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    axes[1].set_title("Validation Mask mAP@0.50:0.95 across Epochs", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    loss_chart_path = os.path.join(output_dir, "kfold_loss_and_map_curves.png")
    fig.savefig(loss_chart_path, dpi=200, bbox_inches="tight")
    if artifact_dir:
        fig.savefig(os.path.join(artifact_dir, "kfold_loss_and_map_curves.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Grafik loss & learning curve tersimpan di: {loss_chart_path}")
    
    return metrics_chart_path, loss_chart_path

def generate_sample_predictions_visualization(best_model_path, val_files, output_dir, artifact_dir=None, target_cm=14.0, num_samples=6):
    """Menghasilkan visualisasi prediksi mask + ellipse fitting + skala faktor pada citra validasi asli."""
    print(f"\n[5/6] Menghasilkan visualisasi inferensi segmentasi stiker lingkaran...")
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        
    model = YOLO(best_model_path)
    samples = val_files[:num_samples]
    
    n_cols = 3
    n_rows = math.ceil(len(samples) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if len(samples) > 1 else [axes]
    
    for idx, (res_img_p, lbl_p, orig_img_p) in enumerate(samples):
        # Inferensi dilakukan pada citra asli untuk menguji akurasi resolusi penuh
        img_bgr = cv2.imread(orig_img_p)
        if img_bgr is None:
            img_bgr = cv2.imread(res_img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        
        # Inference
        results = model(img_rgb, conf=0.25, verbose=False)
        
        vis_img = img_rgb.copy()
        pred_found = False
        scale_cm_per_px = 0.0
        major_axis = 0.0
        minor_axis = 0.0
        
        for r in results:
            if r.masks is not None and len(r.boxes) > 0:
                mask_np = r.masks.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                
                # Visual overlay
                vis_img[binary_mask == 1] = [255, 0, 128] # Pinkish mask
                vis_blend = cv2.addWeighted(img_rgb, 0.6, vis_img, 0.4, 0)
                
                # Ellipse Fitting
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Expand crop around sticker
                    pad = max(int(max(w, h) * 1.5), 100)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w_orig, x + w + pad)
                    y2 = min(h_orig, y + h + pad)
                    
                    if len(cnt) >= 5:
                        (xc, yc), (d1, d2), angle = cv2.fitEllipse(cnt)
                        major_axis = max(d1, d2)
                        minor_axis = min(d1, d2)
                    else:
                        xc, yc = x + w/2, y + h/2
                        major_axis, minor_axis = max(w, h), min(w, h)
                        angle = 0
                        
                    if major_axis > 0:
                        scale_cm_per_px = target_cm / major_axis
                        pred_found = True
                        
                    # Gambar crop
                    axes[idx].imshow(vis_blend[y1:y2, x1:x2])
                    
                    # Tambahkan overlay ellipse di koordinat lokal crop
                    local_xc = xc - x1
                    local_yc = yc - y1
                    ellipse_patch = patches.Ellipse(
                        (local_xc, local_yc), major_axis, minor_axis, angle=angle,
                        fill=False, edgecolor='cyan', lw=3, label=f'Fit Ellipse (D={major_axis:.1f}px)'
                    )
                    axes[idx].add_patch(ellipse_patch)
                    break
                    
        if not pred_found:
            axes[idx].imshow(vis_img)
            axes[idx].set_title(f"{os.path.basename(orig_img_p)}\n[Sticker Not Detected]", color='red')
        else:
            conf_score = float(results[0].boxes.conf[0]) if len(results[0].boxes) > 0 else 0.0
            axes[idx].set_title(
                f"Sample: {os.path.basename(orig_img_p)}\n"
                f"Conf: {conf_score:.2f} | Diam: {major_axis:.1f} px | S: {scale_cm_per_px:.5f} cm/px",
                fontsize=10, fontweight='bold'
            )
            axes[idx].legend(loc="upper right", fontsize=8)
        axes[idx].axis('off')
        
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    pred_vis_path = os.path.join(output_dir, "validation_prediction_samples.png")
    fig.savefig(pred_vis_path, dpi=200, bbox_inches="tight")
    if artifact_dir:
        fig.savefig(os.path.join(artifact_dir, "validation_prediction_samples.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Visualisasi inferensi validasi tersimpan di: {pred_vis_path}")
    return pred_vis_path

def main():
    image_dir = "datasets/sticker_circle/images"
    base_processed_dir = "datasets/sticker_circle/processed_480"
    kfold_base_dir = "runs/kfold_circle_splits"
    pretrained_model = "models/yolov8n-seg.pt"
    vis_output_dir = "runs/kfold_circle/visualizations"
    artifact_dir = r"C:\Users\bimyu\.gemini\antigravity-ide\brain\2b71bb66-34d6-403f-9556-660a61e5cdfb"
    
    # 1. Prepare YOLO labels and pre-resized images (480x480)
    valid_pairs = prepare_yolo_annotations_and_resized_images(image_dir, base_processed_dir, num_circle_points=32, target_dim=480)
    
    # 2. Setup K-Fold
    fold_configs = setup_kfold_datasets(valid_pairs, kfold_base_dir, n_splits=5, seed=42)
    
    # 3. Train K-Fold (15 epochs with batch=16, imgsz=480, and RAM cache)
    results_summary, best_fold_model_path, best_fold_idx = train_kfold(
        fold_configs, pretrained_model, epochs=15, imgsz=480, batch=16
    )
    
    # 4. Save best model to models/ directory
    final_best_model_dest = "models/best_sticker_circle.pt"
    if best_fold_model_path and os.path.exists(best_fold_model_path):
        shutil.copy2(best_fold_model_path, final_best_model_dest)
        print(f"\n[+] Model terbaik (Fold {best_fold_idx + 1}) berhasil disalin ke: {final_best_model_dest}")
        
    # 5. Generate Evaluation & Loss Plots
    metrics_plot, loss_plot = generate_evaluation_plots(results_summary, vis_output_dir, artifact_dir)
    
    # 6. Generate Validation Visualizations
    all_val_files = []
    for cfg in fold_configs:
        all_val_files.extend(cfg["val_files"])
    pred_plot = generate_sample_predictions_visualization(
        final_best_model_dest, all_val_files, vis_output_dir, artifact_dir, target_cm=14.0, num_samples=6
    )
    
    # 7. Compute & Print Aggregated Summary Table
    df = pd.DataFrame(results_summary)
    summary_df = df[["Fold", "Box_Precision", "Box_Recall", "Box_mAP50", "Box_mAP50-95",
                     "Mask_Precision", "Mask_Recall", "Mask_mAP50", "Mask_mAP50-95"]]
    
    means = summary_df.mean(numeric_only=True)
    stds = summary_df.std(numeric_only=True)
    
    print("\n" + "="*80)
    print("📈 RINGKASAN HASIL EVALUASI 5-FOLD CROSS VALIDATION (STICKER CIRCLE)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("-"*80)
    print(f"RATA-RATA MASK mAP@0.50      : {means['Mask_mAP50']:.4f} ± {stds['Mask_mAP50']:.4f}")
    print(f"RATA-RATA MASK mAP@0.50:0.95 : {means['Mask_mAP50-95']:.4f} ± {stds['Mask_mAP50-95']:.4f}")
    print(f"RATA-RATA BOX mAP@0.50       : {means['Box_mAP50']:.4f} ± {stds['Box_mAP50']:.4f}")
    print(f"RATA-RATA BOX mAP@0.50:0.95  : {means['Box_mAP50-95']:.4f} ± {stds['Box_mAP50-95']:.4f}")
    print("="*80)
    
    # Save CSV Summary
    summary_df.to_csv(os.path.join(vis_output_dir, "kfold_results_summary.csv"), index=False)
    if artifact_dir:
        summary_df.to_csv(os.path.join(artifact_dir, "kfold_results_summary.csv"), index=False)

if __name__ == "__main__":
    main()

import os
import json
import glob
import math
import shutil
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Menggunakan 8 core CPU untuk komputasi
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

def rotate_normalized_polygon(poly_norm, angle_deg):
    """
    Memutar koordinat poligon ternormalisasi [0, 1] dengan menjaga rasio aspek fisik:
    - 0°   : (nx, ny)
    - 90°  : (1 - ny, nx)  [Clockwise]
    - 180° : (1 - nx, 1 - ny)
    - 270° : (ny, 1 - nx)  [Counter-Clockwise]
    """
    rotated_pts = []
    for nx, ny in poly_norm:
        if angle_deg == 0:
            rx, ry = nx, ny
        elif angle_deg == 90:
            rx, ry = 1.0 - ny, nx
        elif angle_deg == 180:
            rx, ry = 1.0 - nx, 1.0 - ny
        elif angle_deg == 270:
            rx, ry = ny, 1.0 - nx
        else:
            rx, ry = nx, ny
        rotated_pts.append((max(0.0, min(1.0, rx)), max(0.0, min(1.0, ry))))
    return rotated_pts

def prepare_augmented_dataset_aspect_ratio_preserved(
    raw_images_dir,
    output_base_dir,
    test_size=0.15,
    random_state=42
):
    """
    Augmentasi 4 sudut rotasi TANPA peregangan (aspect ratio 100% terjaga murni).
    Stiker lingkaran tetap berbentuk lingkaran sempurna.
    """
    print("\n[1/5] Memproses Augmentasi 4 Sudut Rotasi (Aspect Ratio Terjaga 100%)...")
    train_img_dir = os.path.join(output_base_dir, "images", "train")
    train_lbl_dir = os.path.join(output_base_dir, "labels", "train")
    val_img_dir = os.path.join(output_base_dir, "images", "val")
    val_lbl_dir = os.path.join(output_base_dir, "labels", "val")
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        
    json_files = sorted(glob.glob(os.path.join(raw_images_dir, "*.json")))
    base_names = []
    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        img_p = os.path.join(raw_images_dir, f"{base}.jpg")
        if os.path.exists(img_p):
            base_names.append(base)
            
    train_bases, val_bases = train_test_split(base_names, test_size=test_size, random_state=random_state)
    print(f"   -> Citra Unik Asli: {len(base_names)} (Train: {len(train_bases)}, Val: {len(val_bases)})")
    
    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]
    
    def process_split(bases, img_out, lbl_out):
        count = 0
        for base in bases:
            jf_path = os.path.join(raw_images_dir, f"{base}.json")
            img_path = os.path.join(raw_images_dir, f"{base}.jpg")
            
            with open(jf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                continue
            orig_h, orig_w = img_orig.shape[:2]
            
            # Ekstrak poligon dasar ternormalisasi
            base_poly_norm = []
            for shape in data.get("shapes", []):
                label = shape.get("label", "").lower()
                if "sticker" not in label:
                    continue
                shape_type = shape.get("shape_type", "circle")
                pts = shape.get("points", [])
                if shape_type == "circle" and len(pts) == 2:
                    poly_pts = circle_to_polygon(pts[0], pts[1], num_points=32)
                elif shape_type == "polygon" and len(pts) >= 3:
                    poly_pts = pts
                else:
                    continue
                for pt in poly_pts:
                    nx = max(0.0, min(1.0, pt[0] / orig_w))
                    ny = max(0.0, min(1.0, pt[1] / orig_h))
                    base_poly_norm.append((nx, ny))
                break
                
            if not base_poly_norm:
                continue
                
            # Simpan 4 rotasi tanpa deformasi / tanpa stretch
            for angle, rot_flag in rotations:
                aug_base = f"{base}_rot{angle}"
                if rot_flag is not None:
                    rotated_img = cv2.rotate(img_orig, rot_flag)
                else:
                    rotated_img = img_orig.copy()
                    
                # Simpan gambar asli / rotasi murni tanpa peregangan ukuran
                out_img_p = os.path.join(img_out, f"{aug_base}.jpg")
                cv2.imwrite(out_img_p, rotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Hitung koordinat rotasi yang sesuai
                rot_poly = rotate_normalized_polygon(base_poly_norm, angle)
                coords_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in rot_poly])
                line = f"0 {coords_str}\n"
                
                out_lbl_p = os.path.join(lbl_out, f"{aug_base}.txt")
                with open(out_lbl_p, "w", encoding="utf-8") as out_f:
                    out_f.write(line)
                    
                count += 1
        return count

    train_count = process_split(train_bases, train_img_dir, train_lbl_dir)
    val_count = process_split(val_bases, val_img_dir, val_lbl_dir)
    
    print(f"-> Selesai! Train Set: {train_count} citra | Val Set: {val_count} citra (Total: {train_count + val_count} citra)")
    
    # Buat file konfigurasi data YAML
    yaml_path = os.path.join(output_base_dir, "data_nodistort.yaml")
    yaml_content = f"""path: {os.path.abspath(output_base_dir).replace('\\', '/')}
train: images/train
val: images/val

names:
  0: sticker
"""
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)
        
    return yaml_path

class RealtimeLogCallback:
    """Callback YOLOv8 untuk mencatat progres pelatihan real-time."""
    def __init__(self, log_filepath):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)
        self.best_mask_map = 0.0
        self.patience_counter = 0
        
        with open(self.log_filepath, "w", encoding="utf-8") as f:
            f.write("="*112 + "\n")
            f.write(f"🚀 REAL-TIME TRAINING LOG: YOLOv8-SEG CIRCLE STICKER (NO-DISTORTION) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*112 + "\n")
            f.write(f"{'Epoch':^7} | {'Train_Box':^9} | {'Train_Seg':^9} | {'Train_Cls':^9} | {'Val_Box_P':^9} | {'Val_Box_R':^9} | {'Box_mAP50':^9} | {'Mask_mAP50':^10} | {'Mask_50-95':^11} | {'Status':^8} | {'Timestamp':^19}\n")
            f.write("-"*112 + "\n")

    def on_fit_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        max_epochs = trainer.epochs
        
        # Loss
        box_loss = float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0.0
        seg_loss = float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0.0
        cls_loss = float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0.0
        
        # Metrics
        metrics = trainer.metrics if hasattr(trainer, "metrics") else {}
        box_p = float(metrics.get("metrics/precision(B)", 0.0))
        box_r = float(metrics.get("metrics/recall(B)", 0.0))
        box_map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
        mask_map50 = float(metrics.get("metrics/mAP50(M)", 0.0))
        mask_map = float(metrics.get("metrics/mAP50-95(M)", 0.0))
        
        status = " "
        if mask_map > self.best_mask_map:
            self.best_mask_map = mask_map
            status = "⭐ BEST"
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            status = f"p={self.patience_counter}/10"
            
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{epoch:>3}/{max_epochs:<3} | {box_loss:>9.4f} | {seg_loss:>9.4f} | {cls_loss:>9.4f} | {box_p:>9.4f} | {box_r:>9.4f} | {box_map50:>9.4f} | {mask_map50:>10.4f} | {mask_map:>11.4f} | {status:^8} | {time_str}\n"
        
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(log_line)
            f.flush()

def train_nodistort_model(
    yaml_path,
    pretrained_model="models/yolov8n-seg.pt",
    epochs=80,
    patience=10,
    imgsz=640,
    batch=8,
    log_file="runs/training_realtime_progress.log"
):
    """Melatih YOLOv8 dengan rect=True untuk menjaga aspect ratio alami."""
    print(f"\n[2/5] Memulai Pelatihan YOLOv8-Seg (Max Epochs: {epochs}, Patience: {patience}, Batch: {batch}, imgsz: {imgsz})...")
    print(f"📝 Real-time log ditulis ke: {log_file}")
    
    model = YOLO(pretrained_model)
    
    cb = RealtimeLogCallback(log_file)
    model.add_callback("on_fit_epoch_end", cb.on_fit_epoch_end)
    
    project_dir = "runs/circle_sticker_nodistort"
    run_name = "train_run"
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        rect=True,      # Rectangular training untuk menjaga rasio aspek alami citra
        batch=batch,
        lr0=0.003,
        device="cpu",
        workers=0,
        cache=True,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        verbose=False,
        seed=42
    )
    
    best_model_weights = os.path.join(project_dir, run_name, "weights", "best.pt")
    dest_best_model = "models/best_sticker_circle.pt"
    
    if os.path.exists(best_model_weights):
        shutil.copy2(best_model_weights, dest_best_model)
        print(f"\n🏆 Model Terbaik berhasil disalin ke: {dest_best_model}")
        
    return dest_best_model, os.path.join(project_dir, run_name)

def generate_evaluation_visualizations(run_dir, best_model_path, val_img_dir, output_vis_dir, artifact_dir=None):
    """Menghasilkan grafik learning curves dan visualisasi fitting elips sempurna."""
    print("\n[3/5] Menghasilkan Visualisasi Evaluasi & Grafik Learning Curves...")
    os.makedirs(output_vis_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        
    # 1. Plot Loss & mAP Curves
    csv_path = os.path.join(run_dir, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df)+1)
        
        if "train/seg_loss" in df.columns and "val/seg_loss" in df.columns:
            axes[0].plot(epochs, df["train/seg_loss"], label="Train Seg Loss", color="#3498db", lw=2)
            axes[0].plot(epochs, df["val/seg_loss"], label="Val Seg Loss", color="#e74c3c", lw=2)
        if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
            axes[0].plot(epochs, df["train/cls_loss"], label="Train Cls Loss", color="#9b59b6", lw=1.5, linestyle="--")
            axes[0].plot(epochs, df["val/cls_loss"], label="Val Cls Loss", color="#e67e22", lw=1.5, linestyle="--")
            
        axes[0].set_title("Training & Validation Loss Curves (No Distortion)", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        if "metrics/mAP50(M)" in df.columns:
            axes[1].plot(epochs, df["metrics/mAP50(M)"], label="Mask mAP@0.50", color="#2ecc71", lw=2.5)
        if "metrics/mAP50-95(M)" in df.columns:
            axes[1].plot(epochs, df["metrics/mAP50-95(M)"], label="Mask mAP@0.50:0.95", color="#1abc9c", lw=2.5)
        if "metrics/mAP50(B)" in df.columns:
            axes[1].plot(epochs, df["metrics/mAP50(B)"], label="Box mAP@0.50", color="#f1c40f", lw=1.5, linestyle=":")
            
        axes[1].set_title("Validation Mask & Box mAP Performance", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP Score")
        axes[1].set_ylim([0.0, 1.05])
        axes[1].legend(loc="lower right")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        curves_path = os.path.join(output_vis_dir, "nodistort_training_curves.png")
        fig.savefig(curves_path, dpi=200, bbox_inches="tight")
        if artifact_dir:
            fig.savefig(os.path.join(artifact_dir, "nodistort_training_curves.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"-> Kurva training tersimpan di: {curves_path}")
        
    # 2. Visualisasi Prediksi pada Sampel Validasi
    print("\n[4/5] Menguji Model Terbaik pada Sampel Data Validasi...")
    model = YOLO(best_model_path)
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))[:8]
    
    n_cols = 4
    n_rows = math.ceil(len(val_images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if len(val_images) > 1 else [axes]
    
    target_cm = 14.0
    for idx, img_p in enumerate(val_images):
        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        
        results = model(img_rgb, conf=0.25, verbose=False)
        vis_img = img_rgb.copy()
        pred_found = False
        major_axis = 0.0
        scale_cm_per_px = 0.0
        
        for r in results:
            if r.masks is not None and len(r.boxes) > 0:
                mask_np = r.masks.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                
                vis_img[binary_mask == 1] = [255, 0, 128]
                vis_blend = cv2.addWeighted(img_rgb, 0.6, vis_img, 0.4, 0)
                
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                    pad = max(int(max(w, h) * 1.5), 60)
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
                        
                    axes[idx].imshow(vis_blend[y1:y2, x1:x2])
                    local_xc = xc - x1
                    local_yc = yc - y1
                    ellipse_patch = patches.Ellipse(
                        (local_xc, local_yc), major_axis, minor_axis, angle=angle,
                        fill=False, edgecolor='cyan', lw=3, label=f'Fit (D={major_axis:.1f}px)'
                    )
                    axes[idx].add_patch(ellipse_patch)
                    break
                    
        if not pred_found:
            axes[idx].imshow(vis_img)
            axes[idx].set_title(f"{os.path.basename(img_p)}\n[No Sticker]", color='red')
        else:
            conf_val = float(results[0].boxes.conf[0])
            axes[idx].set_title(
                f"{os.path.basename(img_p)}\nConf: {conf_val:.2f} | D: {major_axis:.1f}px | S: {scale_cm_per_px:.4f} cm/px",
                fontsize=9, fontweight='bold'
            )
            axes[idx].legend(loc="upper right", fontsize=8)
        axes[idx].axis('off')
        
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    pred_vis_path = os.path.join(output_vis_dir, "nodistort_validation_predictions.png")
    fig.savefig(pred_vis_path, dpi=200, bbox_inches="tight")
    if artifact_dir:
        fig.savefig(os.path.join(artifact_dir, "nodistort_validation_predictions.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Visualisasi inferensi tersimpan di: {pred_vis_path}")

def main():
    raw_images_dir = "datasets/sticker_circle/images"
    output_base_dir = "datasets/sticker_circle/augmented_nodistort"
    pretrained_model = "models/yolov8n-seg.pt"
    vis_output_dir = "runs/circle_sticker_nodistort/visualizations"
    log_file = "runs/training_realtime_progress.log"
    artifact_dir = r"C:\Users\bimyu\.gemini\antigravity-ide\brain\2b71bb66-34d6-403f-9556-660a61e5cdfb"
    
    # 1. Augmentasi 4 sudut rotasi dengan rasio aspek murni terjaga
    yaml_path = prepare_augmented_dataset_aspect_ratio_preserved(
        raw_images_dir, output_base_dir, test_size=0.15, random_state=42
    )
    
    # 2. Pelatihan dengan rect=True dan early stopping 10 epoch
    best_model_path, run_dir = train_nodistort_model(
        yaml_path=yaml_path,
        pretrained_model=pretrained_model,
        epochs=80,
        patience=10,
        imgsz=640,
        batch=8,
        log_file=log_file
    )
    
    # 3. Visualisasi Evaluasi & Prediksi
    val_img_dir = os.path.join(output_base_dir, "images", "val")
    generate_evaluation_visualizations(run_dir, best_model_path, val_img_dir, vis_output_dir, artifact_dir)
    
    print("\n[5/5] Pelatihan & Evaluasi No-Distortion Selesai 100%!")

if __name__ == "__main__":
    main()

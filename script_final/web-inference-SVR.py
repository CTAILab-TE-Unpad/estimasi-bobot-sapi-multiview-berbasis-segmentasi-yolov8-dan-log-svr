import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import os
import math
from ultralytics import YOLO

# --- Config & Page Setup ---
st.set_page_config(page_title="Cattle Weight Inference", layout="wide", page_icon="🐮")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models") if os.path.exists(os.path.join(PROJECT_DIR, "models", "best_sticker.pt")) else os.path.join(SCRIPT_DIR, "models")

YOLO_SEG_MODEL = os.path.join(MODELS_DIR, "yolov8l-seg.pt")
YOLO_STICKER_SQUARE_MODEL = os.path.join(MODELS_DIR, "best_sticker.pt")
YOLO_STICKER_CIRCLE_MODEL = os.path.join(MODELS_DIR, "best_sticker_circle.pt")
SVR_MODEL_PATH = os.path.join(MODELS_DIR, "svr_log_pipeline.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")

@st.cache_resource
def load_models():
    model_seg = YOLO(YOLO_SEG_MODEL)
    model_sticker_square = YOLO(YOLO_STICKER_SQUARE_MODEL)
    if os.path.exists(YOLO_STICKER_CIRCLE_MODEL):
        model_sticker_circle = YOLO(YOLO_STICKER_CIRCLE_MODEL)
    else:
        model_sticker_circle = model_sticker_square
    svr_pipe = joblib.load(SVR_MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model_seg, model_sticker_square, model_sticker_circle, svr_pipe, meta

model_seg, model_sticker_square, model_sticker_circle, svr_pipe, meta = load_models()

def ramanujan_girth(a: float, b: float, correction: float = 1.15) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return correction * np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))

def get_sticker_scale(img: np.ndarray, model: YOLO, target_cm: float = 10.16, shape: str = 'square', conf: float = 0.25):
    results = model(img, conf=conf, verbose=False)
    for r in results:
        if len(r.boxes) == 0:
            continue

        box = r.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        width_px = max(1, x2 - x1)
        height_px = max(1, y2 - y1)

        mask = None
        area = 0
        if r.masks is not None and len(r.masks.data) > 0:
            mask_data = r.masks.data[0].cpu().numpy()
            mask = cv2.resize(mask_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            area = int(np.count_nonzero(mask > 0.5))

        if shape.lower().startswith('sq'):
            if area > 0:
                scale = target_cm / np.sqrt(float(area))
            else:
                side_px = (width_px + height_px) / 2.0
                scale = target_cm / side_px
        else:
            # --- Circular Sticker: High-Res Sub-Pixel Invariant Ellipse Fitting ---
            pad_x = int(width_px * 0.15)
            pad_y = int(height_px * 0.15)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img.shape[1], x2 + pad_x)
            crop_y2 = min(img.shape[0], y2 + pad_y)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            major_axis_px = float((width_px + height_px) / 2.0)  # default fallback

            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 30, 100)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                cnts, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                crop_cx = crop.shape[1] / 2.0
                crop_cy = crop.shape[0] / 2.0
                expected_r = (width_px + height_px) / 4.0
                best_score = float("inf")

                for c in cnts:
                    if len(c) >= 15 and cv2.contourArea(c) > 200:
                        (xc, yc), (d1, d2), angle = cv2.fitEllipse(c)
                        major = float(max(d1, d2))
                        minor = float(min(d1, d2))
                        dist_to_center = float(np.hypot(xc - crop_cx, yc - crop_cy))
                        size_diff = abs(major - 2 * expected_r) / (2 * expected_r)

                        if size_diff < 0.25 and dist_to_center < expected_r * 0.5:
                            score = dist_to_center + size_diff * 50
                            if score < best_score:
                                best_score = score
                                major_axis_px = major

            if major_axis_px <= 0:
                continue
            scale = target_cm / major_axis_px

        return scale, (int(x1), int(y1), int(width_px), int(height_px)), mask

    return None, None, None

def process_side(img: np.ndarray, seg_model: YOLO, sticker_model: YOLO, target_cm: float = 10.16, shape: str = 'square'):
    scale, sticker_bbox, sticker_mask = get_sticker_scale(img, sticker_model, target_cm=target_cm, shape=shape)
    results = seg_model(img, verbose=False)
    cow_mask = None
    for r in results:
        if r.masks is not None:
            mask = r.masks.data[0].cpu().numpy()
            cow_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cow_mask = (cow_mask * 255).astype(np.uint8)
            break
            
    if cow_mask is None:
        return None
        
    y_idx, x_idx = np.where(cow_mask == 255)
    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    
    BL_px = float(x_max - x_min)
    WH_px = float(y_max - y_min)  # Full vertical side height
    
    chest_x = int(x_min + 0.30 * BL_px)
    body_cutoff = int(y_min + 0.55 * (y_max - y_min))
    col_full = np.where(cow_mask[:, chest_x] == 255)[0]
    col = col_full[col_full <= body_cutoff]
    if len(col) == 0:
        col = col_full
        
    if len(col) > 0:
        b_px = float(col.max() - col.min()) / 2.0
        y_start, y_end = int(col.min()), int(col.max())
    else:
        b_px = WH_px / 4.0
        y_start, y_end = int(y_min), int(y_max)
        
    return {
        'scale': scale, 'sticker_bbox': sticker_bbox, 'sticker_mask': sticker_mask,
        'cow_mask': cow_mask, 'BL_px': BL_px, 'WH_px': WH_px, 'b_px': b_px,
        'x_min': int(x_min), 'x_max': int(x_max), 'y_min': int(y_min), 'y_max': int(y_max),
        'chest_x': chest_x, 'WH_y_start': y_start, 'WH_y_end': y_end
    }

def process_back(img: np.ndarray, seg_model: YOLO, sticker_model: YOLO, target_cm: float = 10.16, shape: str = 'square'):
    scale, sticker_bbox, sticker_mask = get_sticker_scale(img, sticker_model, target_cm=target_cm, shape=shape)
    results = seg_model(img, verbose=False)
    cow_mask = None
    for r in results:
        if r.masks is not None:
            mask = r.masks.data[0].cpu().numpy()
            cow_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cow_mask = (cow_mask * 255).astype(np.uint8)
            break
            
    if cow_mask is None:
        return None
        
    y_idx, x_idx = np.where(cow_mask == 255)
    y_min, y_max = y_idx.min(), y_idx.max()
    x_min, x_max = x_idx.min(), x_idx.max()
    
    chest_y = int(y_min + 0.25 * (y_max - y_min))
    row_idx = np.where(cow_mask[chest_y, :] == 255)[0]
    
    if len(row_idx) > 0:
        a_px = float(row_idx.max() - row_idx.min()) / 2.0
        x_center = float(row_idx.min() + row_idx.max()) / 2.0
    else:
        a_px = float(x_max - x_min) / 4.0
        x_center = float(x_min + x_max) / 2.0
        
    return {
        'scale': scale, 'sticker_bbox': sticker_bbox, 'sticker_mask': sticker_mask,
        'cow_mask': cow_mask, 'a_px': a_px, 'x_center': x_center, 'chest_y': float(chest_y),
        'WH_px': float(y_max - y_min)
    }

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Calibration Settings")
st.sidebar.markdown("Atur parameter ukuran dan bentuk stiker referensi untuk foto samping dan belakang.")

sticker_shape = st.sidebar.selectbox(
    "Bentuk Stiker / Model Ukuran",
    options=["Persegi / Luas Area (Square)", "Bulat / Diameter (Circle)"],
    index=0,
    help="Persegi: Ukuran menunjukkan panjang sisi persegi (contoh: 10.16 cm = 4 inci).\nBulat: Ukuran menunjukkan diameter lingkaran (contoh: 5.0 cm)."
)

shape_mode = 'square' if "Persegi" in sticker_shape else 'circle'

col_cal1, col_cal2 = st.sidebar.columns(2)
with col_cal1:
    side_sticker_cm = st.number_input(
        "Stiker Samping (cm)",
        min_value=0.5,
        max_value=50.0,
        value=10.16 if shape_mode == 'square' else 5.0,
        step=0.1,
        help="Ukuran nyata stiker kalibrasi pada foto tampak samping."
    )

with col_cal2:
    back_sticker_cm = st.number_input(
        "Stiker Belakang (cm)",
        min_value=0.5,
        max_value=50.0,
        value=10.16 if shape_mode == 'square' else 5.0,
        step=0.1,
        help="Ukuran nyata stiker kalibrasi pada foto tampak belakang."
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips Pengujian Lapangan:**\n- Tempelkan stiker di bahu/punggung sapi.\n- Jika stiker hanya terdeteksi di satu sisi, sistem akan menjembatani (*bridge*) skala menggunakan rasio tinggi gumba.")

# --- Main App Header ---
st.title("🐮 Cattle Weight Estimation App (Multiview SVR-Log)")
st.markdown("Unggah foto tampak samping (*Side View*) dan tampak belakang (*Back View*) sapi untuk mengestimasi bobot badan secara nirsentuh.")

col1, col2 = st.columns(2)
with col1:
    side_file = st.file_uploader("📷 Upload Side Image (Tampak Samping)", type=["jpg", "png", "jpeg"])
with col2:
    back_file = st.file_uploader("📷 Upload Back Image (Tampak Belakang)", type=["jpg", "png", "jpeg"])

if side_file and back_file:
    side_bytes = np.asarray(bytearray(side_file.read()), dtype=np.uint8)
    back_bytes = np.asarray(bytearray(back_file.read()), dtype=np.uint8)
    
    side_img = cv2.imdecode(side_bytes, 1)
    back_img = cv2.imdecode(back_bytes, 1)
    
    with st.spinner("Mengolah segmentasi YOLOv8 & deteksi stiker..."):
        active_sticker_model = model_sticker_circle if shape_mode == 'circle' else model_sticker_square
        side_res = process_side(side_img, model_seg, active_sticker_model, target_cm=side_sticker_cm, shape=shape_mode)
        back_res = process_back(back_img, model_seg, active_sticker_model, target_cm=back_sticker_cm, shape=shape_mode)
        
    if side_res and back_res:
        S_side = side_res['scale']
        S_back = back_res['scale']
        
        WH_side_px = side_res['WH_px']
        WH_back_px = back_res['WH_px']
        
        calibration_successful = True
        bridge_msg = None
        
        if S_side is None and S_back is not None:
            if WH_side_px > 0:
                S_side = S_back * (WH_back_px / WH_side_px)
                bridge_msg = "⚠️ Stiker tidak terdeteksi pada foto samping. Skala foto samping dijembatani (*bridged*) dari foto belakang berbasis rasio tinggi gumba."
            else:
                calibration_successful = False
        elif S_back is None and S_side is not None:
            if WH_back_px > 0:
                S_back = S_side * (WH_side_px / WH_back_px)
                bridge_msg = "⚠️ Stiker tidak terdeteksi pada foto belakang. Skala foto belakang dijembatani (*bridged*) dari foto samping berbasis rasio tinggi gumba."
            else:
                calibration_successful = False
        elif S_side is None and S_back is None:
            calibration_successful = False

        if calibration_successful:
            if bridge_msg:
                st.warning(bridge_msg)
            else:
                st.success("✅ Stiker terdeteksi pada kedua foto. Kalibrasi independen dua sudut pandang berhasil.")
                
            BL_cm = S_side * side_res['BL_px']
            WH_cm = S_side * side_res['WH_px']
            b_cm  = S_side * side_res['b_px']
            a_cm  = S_back * back_res['a_px']
            CG_cm = ramanujan_girth(a_cm, b_cm)
            
            # Feature Engineering
            BL = BL_cm
            WH = WH_cm
            CG = CG_cm
            BL_WH = BL * WH
            BL_sq = BL ** 2
            WH_BL_ratio = WH / BL if BL > 0 else 0
            CG_sq = CG ** 2
            vol_proxy = BL * (CG ** 2)
            log_vol = np.log(vol_proxy) if vol_proxy > 0 else 0
            
            features_dict = {
                'BL_yolov8l': BL, 'WH_yolov8l': WH, 'CG_yolov8l': CG,
                'BL_WH': BL_WH, 'BL_sq': BL_sq, 'WH_BL_ratio': WH_BL_ratio,
                'CG_sq': CG_sq, 'vol_proxy': vol_proxy, 'log_vol': log_vol
            }
            
            features_df = pd.DataFrame([features_dict])
            features_df = features_df[meta['feature_columns']]
            
            # Inference
            log_pred = svr_pipe.predict(features_df)[0]
            weight_pred = np.exp(log_pred)
            
            st.metric(
                label="⚖️ Estimasi Bobot Sapi (kg)",
                value=f"{weight_pred:.2f} kg",
                delta=f"Log-Space SVR (RBF Kernel)"
            )
            
            # Key Physical Metrics Cards
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Panjang Badan (BL)", f"{BL_cm:.1f} cm")
            mcol2.metric("Tinggi Gumba (WH)", f"{WH_cm:.1f} cm")
            mcol3.metric("Lingkar Dada (CG)", f"{CG_cm:.1f} cm")
            mcol4.metric("Lebar Dada (2a)", f"{2*a_cm:.1f} cm")
            
            # Visualizations
            st.subheader("🖼️ Visualisasi Ekstraksi Geometri & Masker YOLOv8")
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # 1. Side View Plot
            side_rgb = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
            ov_side = side_rgb.copy()
            ov_side[side_res['cow_mask'] == 255] = [0, 255, 0]
            side_blend = cv2.addWeighted(side_rgb, 0.7, ov_side, 0.3, 0)
            
            x_min, x_max = side_res['x_min'], side_res['x_max']
            y_min, y_max = side_res['y_min'], side_res['y_max']
            chest_x = side_res['chest_x']
            
            axes[0].imshow(side_blend)
            axes[0].plot([x_min + (x_max-x_min)/2]*2, [y_min, y_max], 'c-', lw=4, label=f'WH = {WH_cm:.1f} cm')
            axes[0].plot([x_min, x_max], [y_min + 100]*2, 'y-', lw=4, label=f'BL = {BL_cm:.1f} cm')
            axes[0].plot([chest_x]*2, [side_res['WH_y_start'], side_res['WH_y_end']], 'm-', lw=4, label=f'2b = {2*b_cm:.1f} cm')
            
            if side_res['sticker_bbox'] is not None:
                sbx, sby, sbw, sbh = side_res['sticker_bbox']
                rect = patches.Rectangle((sbx, sby), sbw, sbh, linewidth=2, edgecolor='r', facecolor='none', label='Sticker')
                axes[0].add_patch(rect)
            
            axes[0].set_title(f"Side View (Skala: {S_side:.5f} cm/px | Stiker: {side_sticker_cm} cm)")
            axes[0].legend(loc='upper right')
            axes[0].axis('off')
            
            # 2. Back View Plot
            back_rgb = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
            ov_back = back_rgb.copy()
            ov_back[back_res['cow_mask'] == 255] = [0, 255, 0]
            back_blend = cv2.addWeighted(back_rgb, 0.7, ov_back, 0.3, 0)
            
            xc = back_res['x_center']
            yc = back_res['chest_y']
            a_px = back_res['a_px']
            b_px = side_res['b_px']
            
            axes[1].imshow(back_blend)
            axes[1].plot([xc - a_px, xc + a_px], [yc]*2, 'm-', lw=4, label=f'2a = {2*a_cm:.1f} cm')
            
            if back_res['sticker_bbox'] is not None:
                bbx, bby, bbw, bbh = back_res['sticker_bbox']
                rect_back = patches.Rectangle((bbx, bby), bbw, bbh, linewidth=2, edgecolor='r', facecolor='none', label='Sticker')
                axes[1].add_patch(rect_back)
            
            ellipse = patches.Ellipse((xc, yc), 2*a_px, 2*b_px, fill=False, edgecolor='blue', lw=3, label=f'CG = {CG_cm:.1f} cm')
            axes[1].add_patch(ellipse)
            
            axes[1].set_title(f"Back View (Skala: {S_back:.5f} cm/px | Stiker: {back_sticker_cm} cm)")
            axes[1].legend(loc='upper right')
            axes[1].axis('off')
            
            st.pyplot(fig)
            
            # Dataframe Display
            st.subheader("📋 Fitur Morfometrik yang Dimasukkan ke Model SVR")
            st.dataframe(features_df.style.format("{:.2f}"))
        else:
            st.error("❌ Stiker tidak terdeteksi pada kedua foto. Silakan pastikan stiker penanda terlihat jelas di foto samping atau belakang.")
    else:
        st.error("❌ Gagal melakukan segmentasi sapi pada salah satu atau kedua foto. Coba foto lain dengan kontur tubuh sapi yang lebih jelas.")

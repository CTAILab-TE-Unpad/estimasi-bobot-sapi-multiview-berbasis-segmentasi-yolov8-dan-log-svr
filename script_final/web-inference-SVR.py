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

# --- Config & Paths ---
st.set_page_config(page_title="Cattle Weight Inference", layout="wide")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# We use yolov8l for extraction as per the pipeline
YOLO_SEG_MODEL = os.path.join(MODELS_DIR, "yolov8l-seg.pt")
YOLO_STICKER_MODEL = os.path.join(MODELS_DIR, "best_sticker.pt")
SVR_MODEL_PATH = os.path.join(MODELS_DIR, "svr_log_pipeline.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")

@st.cache_resource
def load_models():
    model_seg = YOLO(YOLO_SEG_MODEL)
    model_sticker = YOLO(YOLO_STICKER_MODEL)
    svr_pipe = joblib.load(SVR_MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model_seg, model_sticker, svr_pipe, meta

model_seg, model_sticker, svr_pipe, meta = load_models()

def ramanujan_girth(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

def get_sticker_scale(img, model, target_cm=2.5, conf=0.25):
    results = model(img, conf=conf, verbose=False)
    for r in results:
        if r.masks is not None:
            for mask_data in r.masks.data:
                mask = mask_data.cpu().numpy()
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                y_idx, x_idx = np.where(mask > 0.5)
                if len(y_idx) > 0:
                    x_min, x_max = x_idx.min(), x_idx.max()
                    y_min, y_max = y_idx.min(), y_idx.max()
                    width_px = x_max - x_min
                    height_px = y_max - y_min
                    diameter_px = (width_px + height_px) / 2.0
                    return target_cm / diameter_px, (x_min, y_min, width_px, height_px), mask
    return None, None, None

def process_side(img, seg_model, sticker_model):
    scale, sticker_bbox, sticker_mask = get_sticker_scale(img, sticker_model)
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
    
    BL_px = x_max - x_min
    chest_x = x_min + int(0.25 * BL_px)
    
    col_idx = np.where(cow_mask[:, chest_x] == 255)[0]
    if len(col_idx) > 0:
        y_start, y_end = col_idx.min(), col_idx.max()
    else:
        y_start, y_end = y_min, y_max
    
    WH_px = y_end - y_start
    
    top_55_pct_y = y_min + int(0.55 * (y_max - y_min))
    lat_mask = np.zeros_like(cow_mask)
    lat_mask[y_min:top_55_pct_y, :] = cow_mask[y_min:top_55_pct_y, :]
    y_idx_lat, _ = np.where(lat_mask == 255)
    
    if len(y_idx_lat) > 0:
        b_px = (y_idx_lat.max() - y_idx_lat.min()) / 2.0
    else:
        b_px = WH_px / 2.0
        
    return {
        'scale': scale, 'sticker_bbox': sticker_bbox, 'sticker_mask': sticker_mask,
        'cow_mask': cow_mask, 'BL_px': BL_px, 'WH_px': WH_px, 'b_px': b_px,
        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
        'chest_x': chest_x, 'WH_y_start': y_start, 'WH_y_end': y_end
    }

def process_back(img, seg_model, sticker_model):
    scale, sticker_bbox, sticker_mask = get_sticker_scale(img, sticker_model)
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
    
    chest_y = y_min + int(0.25 * (y_max - y_min))
    row_idx = np.where(cow_mask[chest_y, :] == 255)[0]
    
    if len(row_idx) > 0:
        x_start, x_end = row_idx.min(), row_idx.max()
    else:
        x_start, x_end = x_idx.min(), x_idx.max()
        
    a_px = (x_end - x_start) / 2.0
    x_center = (x_start + x_end) / 2.0
    
    return {
        'scale': scale, 'sticker_bbox': sticker_bbox, 'sticker_mask': sticker_mask,
        'cow_mask': cow_mask, 'a_px': a_px, 'x_center': x_center, 'chest_y': chest_y,
        'WH_px': float(y_max - y_min)
    }

st.title("🐮 Cattle Weight Inference App")
st.markdown("Upload Side and Back images of a cow to estimate its weight using the SVR Log-Space model.")

col1, col2 = st.columns(2)
with col1:
    side_file = st.file_uploader("Upload Side Image", type=["jpg", "png", "jpeg"])
with col2:
    back_file = st.file_uploader("Upload Back Image", type=["jpg", "png", "jpeg"])

if side_file and back_file:
    side_bytes = np.asarray(bytearray(side_file.read()), dtype=np.uint8)
    back_bytes = np.asarray(bytearray(back_file.read()), dtype=np.uint8)
    
    side_img = cv2.imdecode(side_bytes, 1)
    back_img = cv2.imdecode(back_bytes, 1)
    
    with st.spinner("Processing images..."):
        side_res = process_side(side_img, model_seg, model_sticker)
        back_res = process_back(back_img, model_seg, model_sticker)
        
    if side_res and back_res:
        S_side = side_res['scale']
        S_back = back_res['scale']
        
        WH_side_px = side_res['WH_px']
        WH_back_px = back_res['WH_px']
        
        # Bridge logic if one scale factor is missing
        calibration_successful = True
        bridge_msg = None
        
        if S_side is None and S_back is not None:
            if WH_side_px > 0:
                S_side = S_back * (WH_back_px / WH_side_px)
                bridge_msg = "⚠️ Sticker undetected in side image. Calibrated side scale factor using back image sticker + withers height ratio."
            else:
                calibration_successful = False
        elif S_back is None and S_side is not None:
            if WH_back_px > 0:
                S_back = S_side * (WH_side_px / WH_back_px)
                bridge_msg = "⚠️ Sticker undetected in back image. Calibrated back scale factor using side image sticker + withers height ratio."
            else:
                calibration_successful = False
        elif S_side is None and S_back is None:
            calibration_successful = False

        if calibration_successful:
            if bridge_msg:
                st.warning(bridge_msg)
            else:
                st.info("✅ Sticker detected in both images. Dual-view independent calibration successful.")
                
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
            
            features = pd.DataFrame([{
                'BL_yolov8l': BL, 'WH_yolov8l': WH, 'CG_yolov8l': CG,
                'BL_WH': BL_WH, 'BL_sq': BL_sq, 'WH_BL_ratio': WH_BL_ratio,
                'CG_sq': CG_sq, 'vol_proxy': vol_proxy, 'log_vol': log_vol
            }])
            
            # Ensure column order matches metadata
            features = features[meta['feature_columns']]
            
            # Inference
            log_pred = svr_pipe.predict(features)[0]
            weight_pred = np.exp(log_pred)
            
            st.success(f"### Estimated Weight: {weight_pred:.2f} kg")
            
            # Visualizations
            st.subheader("Pipeline Visualizations")
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Side visualization
            side_rgb = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
            ov_side = side_rgb.copy()
            ov_side[side_res['cow_mask'] == 255] = [0, 255, 0]
            side_blend = cv2.addWeighted(side_rgb, 0.7, ov_side, 0.3, 0)
            
            x_min, x_max = side_res['x_min'], side_res['x_max']
            y_min = side_res['y_min']
            chest_x = side_res['chest_x']
            y_start, y_end = side_res['WH_y_start'], side_res['WH_y_end']
            
            axes[0].imshow(side_blend)
            axes[0].plot([x_min + (x_max-x_min)/2]*2, [y_start, y_end], 'c-', lw=4, label=f'WH = {WH_cm:.1f} cm')
            axes[0].plot([x_min, x_max], [y_min + 100]*2, 'y-', lw=4, label=f'BL = {BL_cm:.1f} cm')
            axes[0].plot([chest_x]*2, [y_min, y_min + 2*side_res['b_px']], 'm-', lw=4, label=f'2b = {2*b_cm:.1f} cm')
            
            if side_res['sticker_bbox'] is not None:
                sbx, sby, sbw, sbh = side_res['sticker_bbox']
                rect = patches.Rectangle((sbx, sby), sbw, sbh, linewidth=2, edgecolor='r', facecolor='none', label='Sticker')
                axes[0].add_patch(rect)
            
            axes[0].set_title(f"Side View (S={S_side:.4f} cm/px)")
            axes[0].legend()
            axes[0].axis('off')
            
            # Back visualization
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
            
            axes[1].set_title(f"Back View (S={S_back:.4f} cm/px)")
            axes[1].legend()
            axes[1].axis('off')
            
            st.pyplot(fig)
            
            st.subheader("Feature Extraction Results")
            st.dataframe(features)
        else:
            st.error("Sticker undetected in both images. Cannot calibrate dimensions without at least one sticker reference.")
    else:
        st.error("Failed to segment the cow in one or both images. Please try other images.")

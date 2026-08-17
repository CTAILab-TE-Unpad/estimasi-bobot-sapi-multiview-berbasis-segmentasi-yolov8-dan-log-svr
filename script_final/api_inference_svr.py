import os
import io
import base64
import math
import cv2
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend to avoid GUI thread issues
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CattleWeightAPI")

# --- Configuration & Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models") if os.path.exists(os.path.join(PROJECT_DIR, "models", "best_sticker.pt")) else os.path.join(SCRIPT_DIR, "models")

YOLO_SEG_MODEL = os.path.join(MODELS_DIR, "yolov8l-seg.pt")
YOLO_STICKER_SQUARE_MODEL = os.path.join(MODELS_DIR, "best_sticker.pt")
YOLO_STICKER_CIRCLE_MODEL = os.path.join(MODELS_DIR, "best_sticker_circle.pt")
SVR_MODEL_PATH = os.path.join(MODELS_DIR, "svr_log_pipeline.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")

# --- Global Model Variables ---
model_seg = None
model_sticker_square = None
model_sticker_circle = None
svr_pipe = None
meta = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_seg, model_sticker_square, model_sticker_circle, svr_pipe, meta
    logger.info("Loading models into memory (Lifespan Startup)...")
    try:
        model_seg = YOLO(YOLO_SEG_MODEL)
        model_sticker_square = YOLO(YOLO_STICKER_SQUARE_MODEL)
        if os.path.exists(YOLO_STICKER_CIRCLE_MODEL):
            model_sticker_circle = YOLO(YOLO_STICKER_CIRCLE_MODEL)
        else:
            logger.warning(f"Circle model not found at {YOLO_STICKER_CIRCLE_MODEL}, using square model fallback.")
            model_sticker_circle = model_sticker_square
            
        svr_pipe = joblib.load(SVR_MODEL_PATH)
        meta = joblib.load(META_PATH)
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise RuntimeError(f"Startup failure: {str(e)}")
    
    yield
    
    logger.info("Cleaning up and shutting down server (Lifespan Shutdown)...")

# --- Initialize FastAPI with Lifespan ---
app = FastAPI(
    title="Cattle Weight Estimation API",
    description="Microservice to estimate cattle weight from Side and Back images using YOLOv8 and Log-SVR with calibration scale factors.",
    version="1.1.0",
    lifespan=lifespan
)

# --- Helper Functions ---
def ramanujan_girth(a: float, b: float, correction: float = 1.15) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return correction * np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))

def get_sticker_scale(
    img: np.ndarray,
    model: YOLO,
    target_cm: float = 10.16,
    shape: str = 'square',
    conf: float = 0.25
) -> Tuple[Optional[float], Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Deteksi stiker kalibrasi dan ekstraksi faktor skala (s_factor = target_cm / dimension_px).
    Untuk stiker lingkaran: mengekstrak Sumbu Mayor elips sub-pixel (invarian terhadap distorsi kemiringan kamera).
    Untuk stiker persegi: mengekstrak panjang sisi dari kotak / masker.
    """
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
                side_px = np.sqrt(float(area))
            else:
                side_px = float((width_px + height_px) / 2.0)
            scale = target_cm / side_px
            sticker_geom = {
                "shape": "square",
                "side_px": float(side_px),
                "bbox": (x1, y1, width_px, height_px),
                "crop_bbox": (max(0, x1 - 10), max(0, y1 - 10), min(img.shape[1], x2 + 10), min(img.shape[0], y2 + 10))
            }
        else:
            # --- Circular Sticker: High-Res Sub-Pixel Invariant Ellipse Fitting ---
            pad_x = int(width_px * 0.20)
            pad_y = int(height_px * 0.20)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img.shape[1], x2 + pad_x)
            crop_y2 = min(img.shape[0], y2 + pad_y)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            major_axis_px = float((width_px + height_px) / 2.0)
            minor_axis_px = float(min(width_px, height_px))
            center_x = float(x1 + width_px / 2.0)
            center_y = float(y1 + height_px / 2.0)
            ellipse_angle = 0.0

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
                                minor_axis_px = minor
                                center_x = float(crop_x1 + xc)
                                center_y = float(crop_y1 + yc)
                                ellipse_angle = float(angle)

            if major_axis_px <= 0:
                continue
            scale = target_cm / major_axis_px
            sticker_geom = {
                "shape": "circle",
                "center": (center_x, center_y),
                "major_axis": float(major_axis_px),
                "minor_axis": float(minor_axis_px),
                "angle": float(ellipse_angle),
                "bbox": (x1, y1, width_px, height_px),
                "crop_bbox": (crop_x1, crop_y1, crop_x2, crop_y2)
            }

        return scale, (int(x1), int(y1), int(width_px), int(height_px)), mask, sticker_geom

    return None, None, None, None

def process_side(img: np.ndarray, seg_model: YOLO, sticker_model: YOLO, target_cm: float = 10.16, shape: str = 'square') -> Optional[Dict[str, Any]]:
    scale, sticker_bbox, sticker_mask, sticker_geom = get_sticker_scale(img, sticker_model, target_cm=target_cm, shape=shape)
    results = seg_model(img, verbose=False)
    cow_mask = None
    for r in results:
        if r.masks is not None and len(r.masks.data) > 0:
            mask = r.masks.data[0].cpu().numpy()
            cow_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cow_mask = (cow_mask * 255).astype(np.uint8)
            break
            
    if cow_mask is None:
        return None
        
    y_idx, x_idx = np.where(cow_mask == 255)
    if len(y_idx) == 0 or len(x_idx) == 0:
        return None
        
    x_min, x_max = int(x_idx.min()), int(x_idx.max())
    y_min, y_max = int(y_idx.min()), int(y_idx.max())
    
    BL_px = float(x_max - x_min)
    WH_px = float(y_max - y_min)
    
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
        'scale': scale,
        'sticker_bbox': sticker_bbox,
        'sticker_geom': sticker_geom,
        'cow_mask': cow_mask,
        'BL_px': BL_px,
        'WH_px': WH_px,
        'b_px': b_px,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'chest_x': chest_x,
        'WH_y_start': y_start,
        'WH_y_end': y_end
    }

def process_back(img: np.ndarray, seg_model: YOLO, sticker_model: YOLO, target_cm: float = 10.16, shape: str = 'square') -> Optional[Dict[str, Any]]:
    scale, sticker_bbox, sticker_mask, sticker_geom = get_sticker_scale(img, sticker_model, target_cm=target_cm, shape=shape)
    results = seg_model(img, verbose=False)
    cow_mask = None
    for r in results:
        if r.masks is not None and len(r.masks.data) > 0:
            mask = r.masks.data[0].cpu().numpy()
            cow_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cow_mask = (cow_mask * 255).astype(np.uint8)
            break
            
    if cow_mask is None:
        return None
        
    y_idx, x_idx = np.where(cow_mask == 255)
    if len(y_idx) == 0 or len(x_idx) == 0:
        return None
        
    y_min, y_max = int(y_idx.min()), int(y_idx.max())
    x_min, x_max = int(x_idx.min()), int(x_idx.max())
    
    chest_y = int(y_min + 0.25 * (y_max - y_min))
    row_idx = np.where(cow_mask[chest_y, :] == 255)[0]
    
    if len(row_idx) > 0:
        a_px = float(row_idx.max() - row_idx.min()) / 2.0
        x_center = float(row_idx.min() + row_idx.max()) / 2.0
    else:
        a_px = float(x_max - x_min) / 4.0
        x_center = float(x_min + x_max) / 2.0
        
    return {
        'scale': scale,
        'sticker_bbox': sticker_bbox,
        'sticker_geom': sticker_geom,
        'cow_mask': cow_mask,
        'a_px': a_px,
        'x_center': x_center,
        'chest_y': float(chest_y),
        'WH_px': float(y_max - y_min)
    }

def draw_sticker_on_axes(ax, img: np.ndarray, res: dict, sticker_dim_cm: float, scale_val: float, title_prefix: str):
    """Menggambar detail stiker sesuai bentuk (persegi / elips lingkaran) pada Matplotlib axes."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.axis('off')
    
    geom = res.get('sticker_geom')
    if geom is not None:
        shape_type = geom.get('shape', 'square')
        if shape_type == 'circle':
            xc, yc = geom['center']
            major = geom['major_axis']
            minor = geom['minor_axis']
            angle = geom['angle']
            
            # Draw fitted ellipse
            ellipse = patches.Ellipse(
                (xc, yc), major, minor, angle=angle,
                fill=False, edgecolor='cyan', lw=3, label=f'Fit (D={major:.1f}px)'
            )
            ax.add_patch(ellipse)
            ax.plot(xc, yc, 'r+', ms=10, mew=2)
            ax.set_title(f"{title_prefix} Sticker Circle\nDiameter={major:.1f}px | S={scale_val:.5f} cm/px", fontsize=11, fontweight='bold')
        else:
            x, y, w, h = geom['bbox']
            side = geom.get('side_px', max(w, h))
            rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='lime', lw=3, label=f'Fit (L={side:.1f}px)')
            ax.add_patch(rect)
            ax.set_title(f"{title_prefix} Sticker Square\nSide={side:.1f}px | S={scale_val:.5f} cm/px", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
    else:
        ax.set_title(f"{title_prefix} Sticker\n[Sticker Undetected / Bridged]", fontsize=11, color='orange', fontweight='bold')

def generate_full_dashboard_webp_b64(
    side_img: np.ndarray, side_res: dict,
    back_img: np.ndarray, back_res: dict,
    S_side: float, S_back: float,
    BL_cm: float, WH_cm: float, b_cm: float, a_cm: float, CG_cm: float,
    side_dim_cm: float, back_dim_cm: float, shape_mode: str,
    webp_quality: int = 80
) -> str:
    """Menghasilkan satu visualisasi dashboard lengkap yang dikompresi ke format WebP (Base64 string)."""
    
    fig_all, axes_all = plt.subplots(2, 2, figsize=(16, 12))
    
    # Side Blend & Annotations
    side_rgb = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
    ov_side = side_rgb.copy()
    ov_side[side_res['cow_mask'] == 255] = [0, 255, 0]
    side_blend = cv2.addWeighted(side_rgb, 0.75, ov_side, 0.25, 0)
    
    x_min, x_max = side_res['x_min'], side_res['x_max']
    y_min, y_max = side_res['y_min'], side_res['y_max']
    chest_x = side_res['chest_x']
    
    axes_all[0, 0].imshow(side_blend)
    axes_all[0, 0].plot([x_min + (x_max - x_min) / 2]*2, [y_min, y_max], 'c-', lw=3, label=f'WH = {WH_cm:.1f} cm')
    axes_all[0, 0].plot([x_min, x_max], [y_min + int((y_max - y_min)*0.15)]*2, 'y-', lw=3, label=f'BL = {BL_cm:.1f} cm')
    axes_all[0, 0].plot([chest_x]*2, [side_res['WH_y_start'], side_res['WH_y_end']], 'm-', lw=3, label=f'2b = {2*b_cm:.1f} cm')
    axes_all[0, 0].set_title(f"Morfometri Samping (BL={BL_cm:.1f}cm, WH={WH_cm:.1f}cm, 2b={2*b_cm:.1f}cm)", fontsize=11, fontweight='bold')
    axes_all[0, 0].legend(loc='upper right', fontsize=8)
    axes_all[0, 0].axis('off')
    
    # Back Blend & Annotations
    back_rgb = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
    ov_back = back_rgb.copy()
    ov_back[back_res['cow_mask'] == 255] = [0, 255, 0]
    back_blend = cv2.addWeighted(back_rgb, 0.75, ov_back, 0.25, 0)
    
    xc = back_res['x_center']
    yc = back_res['chest_y']
    a_px = back_res['a_px']
    b_px = side_res['b_px']
    
    axes_all[0, 1].imshow(back_blend)
    axes_all[0, 1].plot([xc - a_px, xc + a_px], [yc]*2, 'm-', lw=3, label=f'2a = {2*a_cm:.1f} cm')
    axes_all[0, 1].add_patch(patches.Ellipse((xc, yc), 2 * a_px, 2 * b_px, fill=False, edgecolor='deepskyblue', lw=3, label=f'CG = {CG_cm:.1f} cm'))
    axes_all[0, 1].set_title(f"Morfometri Belakang (2a={2*a_cm:.1f}cm, CG={CG_cm:.1f}cm)", fontsize=11, fontweight='bold')
    axes_all[0, 1].legend(loc='upper right', fontsize=8)
    axes_all[0, 1].axis('off')
    
    # Row 2: Stiker Sesuai Bentuk
    draw_sticker_on_axes(axes_all[1, 0], side_img, side_res, side_dim_cm, S_side, "Deteksi Stiker Samping")
    draw_sticker_on_axes(axes_all[1, 1], back_img, back_res, back_dim_cm, S_back, "Deteksi Stiker Belakang")
    
    plt.tight_layout()
    
    # Render canvas to in-memory WebP image
    fig_all.canvas.draw()
    rgba_buffer = np.asarray(fig_all.canvas.buffer_rgba())
    bgr_img = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
    plt.close(fig_all)
    
    success, webp_bytes = cv2.imencode('.webp', bgr_img, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])
    if not success:
        raise RuntimeError("Gagal mengompresi gambar visualisasi ke format WebP.")
        
    return base64.b64encode(webp_bytes).decode('utf-8')

# --- API Endpoint ---
@app.post("/predict", summary="Inference endpoint to calculate physical dimensions and estimate weight.")
async def predict(
    side_image: UploadFile = File(..., description="JPEG/PNG citra sapi tampak samping"),
    back_image: UploadFile = File(..., description="JPEG/PNG citra sapi tampak belakang"),
    sticker_shape: str = Form("square", description="Bentuk stiker kalibrasi: 'square' (persegi) atau 'circle' (lingkaran)"),
    side_sticker_cm: float = Form(10.16, description="Ukuran stiker samping dalam cm (panjang sisi jika persegi, atau diameter jika lingkaran)"),
    back_sticker_cm: float = Form(10.16, description="Ukuran stiker belakang dalam cm (panjang sisi jika persegi, atau diameter jika lingkaran)")
):
    try:
        side_bytes = await side_image.read()
        back_bytes = await back_image.read()
        
        nparr_side = np.frombuffer(side_bytes, np.uint8)
        nparr_back = np.frombuffer(back_bytes, np.uint8)
        
        side_img = cv2.imdecode(nparr_side, cv2.IMREAD_COLOR)
        back_img = cv2.imdecode(nparr_back, cv2.IMREAD_COLOR)
        
        if side_img is None or back_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode one or both uploaded images.")
            
    except Exception as e:
        logger.error(f"Image load failure: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image decoding error: {str(e)}")

    logger.info("Executing pipeline on uploaded images...")
    
    # 1. Resolve sticker shape & dimension
    shape_mode = 'circle' if str(sticker_shape).lower().startswith('cir') else 'square'
    side_dim = float(side_sticker_cm)
    back_dim = float(back_sticker_cm)

    active_sticker_model = model_sticker_circle if shape_mode == 'circle' else model_sticker_square
    
    # 2. Process side and back views
    side_res = process_side(side_img, model_seg, active_sticker_model, target_cm=side_dim, shape=shape_mode)
    back_res = process_back(back_img, model_seg, active_sticker_model, target_cm=back_dim, shape=shape_mode)
    
    if not side_res or not back_res:
        raise HTTPException(status_code=422, detail="Failed to segment the cow silhouette in one or both views.")
        
    S_side = side_res['scale']
    S_back = back_res['scale']
    WH_side_px = side_res['WH_px']
    WH_back_px = back_res['WH_px']
    
    calib_source = "Dual sticker detection"
    if S_side is None and S_back is not None:
        if WH_side_px > 0:
            S_side = S_back * (WH_back_px / WH_side_px)
            calib_source = "Side scale bridged via back sticker + withers height ratio"
        else:
            raise HTTPException(status_code=422, detail="Side sticker undetected and withers height invalid for bridge.")
    elif S_back is None and S_side is not None:
        if WH_back_px > 0:
            S_back = S_side * (WH_side_px / WH_back_px)
            calib_source = "Back scale bridged via side sticker + withers height ratio"
        else:
            raise HTTPException(status_code=422, detail="Back sticker undetected and withers height invalid for bridge.")
    elif S_side is None and S_back is None:
        raise HTTPException(status_code=422, detail="Calibration sticker undetected in both views. Cannot compute real-world measurements.")

    # 3. Calculate Morphometrics in cm
    BL_cm = S_side * side_res['BL_px']
    WH_cm = S_side * side_res['WH_px']
    b_cm  = S_side * side_res['b_px']
    a_cm  = S_back * back_res['a_px']
    CG_cm = ramanujan_girth(a_cm, b_cm)
    
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
    
    # 4. Predict Weight with Log-SVR
    log_pred = svr_pipe.predict(features_df)[0]
    weight_pred = float(np.exp(log_pred))
    
    # 5. Generate Compressed WebP Full Dashboard Visualization
    full_dashboard_b64 = generate_full_dashboard_webp_b64(
        side_img, side_res, back_img, back_res,
        S_side, S_back, BL_cm, WH_cm, b_cm, a_cm, CG_cm,
        side_dim, back_dim, shape_mode, webp_quality=80
    )
    
    response_payload = {
        "success": True,
        "predicted_weight_kg": round(weight_pred, 2),
        "s_factor": {
            "side": float(S_side),
            "back": float(S_back),
            "unit": "cm/px"
        },
        "calibration": {
            "sticker_shape": shape_mode,
            "scale_source": calib_source,
            "side_sticker_size_cm": float(side_dim),
            "back_sticker_size_cm": float(back_dim),
            "s_factor_side": float(S_side),
            "s_factor_back": float(S_back)
        },
        "physical_measurements": {
            "body_length_cm": round(BL_cm, 2),
            "withers_height_cm": round(WH_cm, 2),
            "chest_girth_cm": round(CG_cm, 2),
            "chest_width_2a_cm": round(2 * a_cm, 2),
            "chest_depth_2b_cm": round(2 * b_cm, 2)
        },
        "all_model_features": {k: float(v) for k, v in features_dict.items()},
        "full_dashboard_b64": full_dashboard_b64
    }
    
    return JSONResponse(content=response_payload)

if __name__ == "__main__":
    uvicorn.run("api_inference_svr:app", host="0.0.0.0", port=8000, reload=True)

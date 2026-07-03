import os
import io
import base64
import math
import cv2
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CattleWeightAPI")

# --- Configuration & Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

YOLO_SEG_MODEL = os.path.join(MODELS_DIR, "yolov8l-seg.pt")
YOLO_STICKER_MODEL = os.path.join(MODELS_DIR, "best_sticker.pt")
SVR_MODEL_PATH = os.path.join(MODELS_DIR, "svr_log_pipeline.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")

# --- Global Model Variables ---
model_seg = None
model_sticker = None
svr_pipe = None
meta = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_seg, model_sticker, svr_pipe, meta
    logger.info("Loading models into memory (Lifespan Startup)...")
    try:
        model_seg = YOLO(YOLO_SEG_MODEL)
        model_sticker = YOLO(YOLO_STICKER_MODEL)
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
    description="Microservice to estimate cattle weight from Side and Back images using YOLOv8 and SVR.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Helper Functions ---
def ramanujan_girth(a: float, b: float) -> float:
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

def get_sticker_scale(img: np.ndarray, model: YOLO, target_cm: float = 2.5, conf: float = 0.25):
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
                    return target_cm / diameter_px, (int(x_min), int(y_min), int(width_px), int(height_px)), mask
    return None, None, None

def process_side(img: np.ndarray, seg_model: YOLO, sticker_model: YOLO) -> Optional[Dict[str, Any]]:
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
        'scale': scale, 'sticker_bbox': sticker_bbox,
        'cow_mask': cow_mask, 'BL_px': float(BL_px), 'WH_px': float(WH_px), 'b_px': float(b_px),
        'x_min': int(x_min), 'x_max': int(x_max), 'y_min': int(y_min), 'y_max': int(y_max),
        'chest_x': int(chest_x), 'WH_y_start': int(y_start), 'WH_y_end': int(y_end)
    }

def process_back(img: np.ndarray, seg_model: YOLO, sticker_model: YOLO) -> Optional[Dict[str, Any]]:
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
        'scale': scale, 'sticker_bbox': sticker_bbox,
        'cow_mask': cow_mask, 'a_px': float(a_px), 'x_center': float(x_center), 'chest_y': float(chest_y),
        'WH_px': float(y_max - y_min)
    }

def generate_base64_visualizations(side_img: np.ndarray, side_res: dict, back_img: np.ndarray, back_res: dict,
                                  S_side: float, S_back: float, BL_cm: float, WH_cm: float, b_cm: float, a_cm: float, CG_cm: float) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Side Plot
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
    
    # 2. Back Plot
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
    
    # Save figure to base64
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

# --- API Endpoint ---
@app.post("/predict", summary="Inference endpoint to calculate physical dimensions and estimate weight.")
async def predict(
    side_image: UploadFile = File(..., description="JPEG/PNG image representing side view of the cow"),
    back_image: UploadFile = File(..., description="JPEG/PNG image representing back view of the cow")
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
    
    # Process views
    side_res = process_side(side_img, model_seg, model_sticker)
    back_res = process_back(back_img, model_seg, model_sticker)
    
    if not side_res or not back_res:
        raise HTTPException(status_code=422, detail="Failed to segment the cow silhouette in one or both views.")
        
    S_side = side_res['scale']
    S_back = back_res['scale']
    WH_side_px = side_res['WH_px']
    WH_back_px = back_res['WH_px']
    
    # Scale calibration bridge
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

    # Calculate real-world metrics (cm)
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
    
    # Predict weight
    features_df = pd.DataFrame([features_dict])
    features_df = features_df[meta['feature_columns']]
    
    log_pred = svr_pipe.predict(features_df)[0]
    weight_pred = float(np.exp(log_pred))
    
    # Generate visualization
    visualizations_b64 = generate_base64_visualizations(
        side_img, side_res, back_img, back_res,
        S_side, S_back, BL_cm, WH_cm, b_cm, a_cm, CG_cm
    )
    
    response_payload = {
        "success": True,
        "predicted_weight_kg": round(weight_pred, 2),
        "calibration": {
            "scale_source": calib_source,
            "scale_side_cm_per_px": float(S_side),
            "scale_back_cm_per_px": float(S_back)
        },
        "features": {
            "body_length_cm": round(BL_cm, 2),
            "withers_height_cm": round(WH_cm, 2),
            "chest_girth_cm": round(CG_cm, 2),
            "chest_width_2a_cm": round(2 * a_cm, 2),
            "chest_depth_2b_cm": round(2 * b_cm, 2)
        },
        "all_model_features": {k: float(v) for k, v in features_dict.items()},
        "visualization_png_b64": visualizations_b64
    }
    
    return JSONResponse(content=response_payload)

if __name__ == "__main__":
    # Runs the API server locally on port 8000
    uvicorn.run("api_inference_svr:app", host="0.0.0.0", port=8000, reload=True)

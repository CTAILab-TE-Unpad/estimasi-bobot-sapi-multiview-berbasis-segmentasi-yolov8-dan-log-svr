import os
import cv2
import numpy as np
from cattle_weight.config import (
    ROI_LATERAL_X, ROI_LATERAL_Y, FLOOR_Y_LATERAL,
    ROI_TOP_X, ROI_TOP_Y
)
from cattle_weight.segmentation import segment_cattle
from cattle_weight.calibration import get_scale_lateral, get_scale_top
from cattle_weight.morphometry import extract_lateral_pixels, extract_top_pixels, estimate_chest_girth
from cattle_weight.utils import QualityGateError, log_failure, logger

def find_image_by_id(directory, idx, view_char):
    """
    Finds a file in directory starting with <idx>_s_ (for side/lateral) or <idx>_r_ (for rear/top).
    Normalizes floats and integers.
    """
    if not os.path.exists(directory):
        return None
    for filename in os.listdir(directory):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                file_id = float(parts[0])
                file_view = parts[1].lower()
                if file_id == float(idx) and file_view == view_char:
                    return os.path.join(directory, filename)
            except ValueError:
                continue
    return None

def process_single_image(idx, row, data_dir):
    """
    Processes the lateral and top view images for a single cow (index idx)
    and extracts calibrated physical dimensions (cm).
    """
    wh_actual = float(row['withers height'])
    cw_actual = float(row['chest width'])
    weight_actual = float(row['live weithg'])
    
    # 1. Load Images
    lat_path = find_image_by_id(os.path.join(data_dir, "Side", "images"), idx, 's')
    top_path = find_image_by_id(os.path.join(data_dir, "Back", "images"), idx, 'r')
    
    if not lat_path or not os.path.exists(lat_path):
        raise FileNotFoundError(f"Lateral image not found for ID: {idx}")
    if not top_path or not os.path.exists(top_path):
        raise FileNotFoundError(f"Top image not found for ID: {idx}")
        
    lat_frame = cv2.imread(lat_path)
    top_frame = cv2.imread(top_path)
    
    # 2. Segment Cattle
    try:
        _, lat_mask = segment_cattle(lat_frame, ROI_LATERAL_X, ROI_LATERAL_Y)
    except QualityGateError as e:
        raise QualityGateError(f"Lateral segmentation quality gate failed: {str(e)}")
        
    try:
        _, top_mask = segment_cattle(top_frame, ROI_TOP_X, ROI_TOP_Y)
    except QualityGateError as e:
        raise QualityGateError(f"Top segmentation quality gate failed: {str(e)}")
        
    # 3. Extract Pixel Landmarks using dense mask
    y_min_lat, lat_w_px, b_px = extract_lateral_pixels(lat_mask)
    top_w_px, a_px = extract_top_pixels(top_mask)
    
    # 4. Calibration (Dynamic Scale Factors)
    S_lateral = get_scale_lateral(wh_actual, y_min_lat, FLOOR_Y_LATERAL)
    S_top = get_scale_top(cw_actual, a_px)
    
    # 5. Convert Pixel Measurements to Physical Units (cm)
    BL_lateral_cm = lat_w_px * S_lateral
    BL_top_cm = top_w_px * S_top
    b_lateral_cm = b_px * S_lateral
    a_top_cm = a_px * S_top
    
    # 6. Girth Calculation & Body Length Cross-Validation
    # k_corr correction factor and validation tolerance are loaded from config automatically
    CG_cm, a_cm, b_cm = estimate_chest_girth(
        BL_lateral_cm, BL_top_cm, a_top_cm, b_lateral_cm
    )
    
    return {
        'N': idx,
        'WH_cm': wh_actual,
        'CW_cm': cw_actual,
        'BL_lateral_cm': BL_lateral_cm,
        'BL_top_cm': BL_top_cm,
        'b_cm': b_cm,
        'a_cm': a_cm,
        'CG_cm': CG_cm,
        'weight_kg': weight_actual
    }

def run_batch_pipeline(df, data_dir):
    """
    Runs the pipeline over the entire dataframe of measurements.
    Collects features of successfully processed cows and logs quality gate failures.
    """
    features_list = []
    success_count = 0
    failure_count = 0
    
    logger.info(f"Starting batch processing of {len(df)} samples...")
    
    for _, row in df.iterrows():
        idx = int(row['N'])
        try:
            res = process_single_image(idx, row, data_dir)
            features_list.append(res)
            success_count += 1
        except Exception as e:
            failure_count += 1
            log_failure("Batch Processing", str(e), idx)
            
    logger.info(f"Batch processing completed. Successful: {success_count}, Failed: {failure_count}")
    
    if len(features_list) == 0:
        raise RuntimeError("No samples successfully passed the quality gates!")
        
    return features_list

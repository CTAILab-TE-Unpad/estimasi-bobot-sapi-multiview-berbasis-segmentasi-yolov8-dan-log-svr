import cv2
import numpy as np
from cattle_weight.config import AREA_RATIO_MIN, AREA_RATIO_MAX, YOLO_MODEL_NAME, YOLO_CONF_THRESHOLD
from cattle_weight.utils import QualityGateError

_yolo_model = None

def get_yolo_model():
    """
    Lazy-loads the YOLOv8-seg model to optimize memory usage.
    """
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_NAME)
    return _yolo_model

def segment_cattle(frame, roi_x, roi_y=None, min_ratio=None, max_ratio=None):
    """
    Performs instance segmentation on the cow image using YOLOv8-seg.
    Selects the target object (cow or class-agnostic match) based on ROI overlap
    and returns the largest contour and binary mask.
    """
    if frame is None:
        raise ValueError("Input frame is None")
        
    min_r = AREA_RATIO_MIN if min_ratio is None else min_ratio
    max_r = AREA_RATIO_MAX if max_ratio is None else max_ratio
    
    # 1. Run YOLOv8-seg inference
    model = get_yolo_model()
    results = model(frame, verbose=False)[0]
    
    if results.masks is None or len(results.boxes) == 0:
        raise QualityGateError("YOLO: No objects detected in the image.")
        
    # Define ROI boundaries
    x_min_roi, x_max_roi = roi_x
    y_min_roi, y_max_roi = (0, frame.shape[0]) if roi_y is None else roi_y
    
    # 2. Select the object with maximum overlap with the ROI (class-agnostic to handle OOD top views)
    target_mask = None
    max_overlap = -1
    
    for box, mask in zip(results.boxes, results.masks):
        conf = float(box.conf[0])
        
        # Select any object with confidence above threshold that overlaps with the ROI
        if conf >= YOLO_CONF_THRESHOLD:
            # Bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            
            # Intersection with ROI
            ix1 = max(x1, x_min_roi)
            iy1 = max(y1, y_min_roi)
            ix2 = min(x2, x_max_roi)
            iy2 = min(y2, y_max_roi)
            
            overlap_w = max(0.0, ix2 - ix1)
            overlap_h = max(0.0, iy2 - iy1)
            overlap_area = overlap_w * overlap_h
            
            if overlap_area > max_overlap:
                max_overlap = overlap_area
                target_mask = mask
                
    if target_mask is None or max_overlap <= 0:
        raise QualityGateError("YOLO: No object detected overlapping with the ROI.")
        
    # 3. Extract and resize binary mask
    m_data = target_mask.data[0].cpu().numpy()
    binary_mask = (m_data > 0.5).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 4. Crop binary mask to ROI boundary to align with physical dimension bounds
    roi_mask = np.zeros_like(binary_mask_resized)
    roi_mask[y_min_roi:y_max_roi, x_min_roi:x_max_roi] = 255
    opened = cv2.bitwise_and(binary_mask_resized, roi_mask)
    
    # 5. Extract contours and validate quality gate
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise QualityGateError("No contours found inside the ROI mask.")
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    total_area = float(frame.shape[0] * frame.shape[1])
    area_ratio = area / total_area
    if not (min_r <= area_ratio <= max_r):
        raise QualityGateError(
            f"Segmented cow contour area ratio {area_ratio:.2f} is outside the allowed range [{min_r:.2f}, {max_r:.2f}]."
        )
        
    return largest_contour, opened

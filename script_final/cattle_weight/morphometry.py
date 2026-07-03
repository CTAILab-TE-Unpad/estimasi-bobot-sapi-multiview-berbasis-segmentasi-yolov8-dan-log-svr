import numpy as np
from cattle_weight.config import CHEST_X_RATIO, CG_CORRECTION_FACTOR, CROSS_VALIDATE_TOLERANCE
from cattle_weight.utils import QualityGateError

def extract_lateral_pixels(mask_lateral, chest_x_ratio=CHEST_X_RATIO):
    """
    Extracts geometric parameters in pixels from the lateral view dense binary mask:
    - y_min: topmost y-coordinate of the back (withers)
    - lat_w: width of the cow's body bounding box (Body Length in pixels)
    - b_px: chest depth semi-axis in pixels (at 30% along BL from shoulder)
    """
    white_y, white_x = np.where(mask_lateral == 255)
    if len(white_x) == 0:
        raise QualityGateError("Lateral segmentation mask is empty.")
        
    x_min, x_max = white_x.min(), white_x.max()
    lat_w = float(x_max - x_min)
    y_min = float(white_y.min())
    
    # Robust chest depth b_px using the dense mask over a window
    chest_x = x_min + chest_x_ratio * lat_w
    thicknesses = []
    for cx in range(int(chest_x - 15), int(chest_x + 16)):
        if 0 <= cx < mask_lateral.shape[1]:
            col_indices = np.where(mask_lateral[:, cx] == 255)[0]
            if len(col_indices) >= 2:
                thicknesses.append(col_indices.max() - col_indices.min())
                
    if len(thicknesses) == 0:
        raise QualityGateError(
            f"Not enough points in lateral mask near chest position {chest_x:.1f} to measure depth."
        )
    b_px = float(np.median(thicknesses) / 2.0)
    
    return y_min, lat_w, b_px

def extract_top_pixels(mask_top, chest_x_ratio=CHEST_X_RATIO):
    """
    Extracts geometric parameters in pixels from the top view dense binary mask:
    - top_w: width of the cow's body bounding box (Body Length in pixels)
    - a_px: chest width semi-axis in pixels (at 30% along BL from shoulder)
    """
    white_y, white_x = np.where(mask_top == 255)
    if len(white_x) == 0:
        raise QualityGateError("Top segmentation mask is empty.")
        
    x_min, x_max = white_x.min(), white_x.max()
    top_w = float(x_max - x_min)
    
    # Robust chest width a_px using the dense mask over a window
    chest_x = x_min + chest_x_ratio * top_w
    thicknesses = []
    for cx in range(int(chest_x - 15), int(chest_x + 16)):
        if 0 <= cx < mask_top.shape[1]:
            col_indices = np.where(mask_top[:, cx] == 255)[0]
            if len(col_indices) >= 2:
                thicknesses.append(col_indices.max() - col_indices.min())
                
    if len(thicknesses) == 0:
        raise QualityGateError(
            f"Not enough points in top mask near chest position {chest_x:.1f} to measure width."
        )
    a_px = float(np.median(thicknesses) / 2.0)
    
    return top_w, a_px

def estimate_chest_girth(BL_lateral, BL_top, a_top, b_lateral, k_corr=CG_CORRECTION_FACTOR, tolerance=CROSS_VALIDATE_TOLERANCE):
    """
    Performs cross-validation of Body Length (BL) from lateral and top views,
    then estimates Chest Girth (CG) using the Ramanujan ellipse formula.
    """
    # Cross-validation on Body Length to ensure postural consistency
    rel_diff = abs(BL_lateral - BL_top) / max(BL_lateral, BL_top)
    if rel_diff > tolerance:
        raise QualityGateError(
            f"Body Length mismatch between Lateral ({BL_lateral:.2f} cm) and Top ({BL_top:.2f} cm) "
            f"is {rel_diff:.1%}, exceeding the tolerance of {tolerance:.1%}. Sapi likely bending/curved."
        )
        
    a = float(a_top)
    b = float(b_lateral)
    
    # Ramanujan ellipse approximation for girth:
    CG = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))) * k_corr
    return float(CG), a, b

def compute_error_propagation(delta_S, delta_a, delta_b, a, b):
    """
    Calculates the first-order error propagation on Chest Girth (CG).
    Assumes S, a, b have independent errors.
    """
    rel_err = np.sqrt((delta_S)**2 + (delta_a)**2 + (delta_b)**2)
    return float(rel_err)

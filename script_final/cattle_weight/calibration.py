from cattle_weight.config import FLOOR_Y_LATERAL
from cattle_weight.utils import QualityGateError

def get_scale_lateral(wh_actual, y_min, floor_y_lateral=FLOOR_Y_LATERAL):
    """
    Computes the lateral camera pixel-to-cm conversion factor S_lateral.
    S_lateral = wh_actual_cm / (floor_y_lateral - y_min_px)
    """
    WH_px = float(floor_y_lateral - y_min)
    if WH_px <= 0:
        raise QualityGateError(
            f"Invalid vertical pixel height measured for Withers Height: WH_px = {WH_px}. "
            f"Topmost point of back y_min ({y_min}) must be strictly less than floor baseline y ({floor_y_lateral})."
        )
    return float(wh_actual / WH_px)

def get_scale_top(cw_actual, a_px):
    """
    Computes the top camera pixel-to-cm conversion factor S_top.
    S_top = cw_actual_cm / (2.0 * a_px)
    """
    cw_px = 2.0 * float(a_px)
    if cw_px <= 0:
        raise QualityGateError(
            f"Invalid chest width in pixels measured: cw_px = {cw_px}. "
            f"Top chest width semi-axis a_px ({a_px}) must be strictly positive."
        )
    return float(cw_actual / cw_px)

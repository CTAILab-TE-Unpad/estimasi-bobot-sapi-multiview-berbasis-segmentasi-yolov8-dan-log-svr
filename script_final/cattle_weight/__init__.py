# Cattle Weight Prediction Modular System
from cattle_weight.config import *
from cattle_weight.utils import QualityGateError, load_measurements
from cattle_weight.calibration import get_scale_lateral, get_scale_top
from cattle_weight.segmentation import segment_cattle
from cattle_weight.morphometry import extract_lateral_pixels, extract_top_pixels, estimate_chest_girth
from cattle_weight.modeling import build_and_validate
from cattle_weight.pipeline import process_single_image, run_batch_pipeline

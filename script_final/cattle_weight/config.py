import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "acme_ai", "Pixel_B2", "B2")
CSV_PATH = os.path.join(BASE_DIR, "datasets", "cow_datasets", "Measurements.csv")

# Camera ROIs
# Lateral camera (Side)
ROI_LATERAL_Y = (50, 1420)
ROI_LATERAL_X = (10, 1890)
FLOOR_Y_LATERAL = 1420  # Floor line set to near bottom for acme_ai

# Top/Rear camera (Back)
ROI_TOP_Y = (0, 1425)
ROI_TOP_X = (300, 1600)

# Morphometry Constants
CHEST_X_RATIO = 0.30  # Position of the chest relative to Oblique Body Length (from front)
CG_CORRECTION_FACTOR = 1.15  # Girth correction factor
CROSS_VALIDATE_TOLERANCE = 0.20  # Max body length difference (20%)

# Segmentation Constants
AREA_RATIO_MIN = 0.02  # Minimum cow contour area ratio (2% of image)
AREA_RATIO_MAX = 0.90  # Maximum cow contour area ratio (90% of image)

# Modeling Constants
N_FOLDS = 10
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 6
RF_MIN_SAMPLES_LEAF = 3
RANDOM_STATE = 42

# YOLO Configuration
YOLO_MODEL_NAME = "yolov8n-seg.pt"
YOLO_CONF_THRESHOLD = 0.25


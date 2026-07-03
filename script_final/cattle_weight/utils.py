import pandas as pd
import logging
import os

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cattle_weight")

class QualityGateError(Exception):
    """Custom exception raised when a processing step fails a quality gate check."""
    pass

def load_measurements(csv_path):
    """
    Loads and parses the Measurements.csv file.
    Strips whitespaces from column names and values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Measurements CSV not found at: {csv_path}")
    
    df = pd.read_csv(csv_path, sep=';')
    df.columns = [col.strip() for col in df.columns]
    
    # Strip any string values and convert appropriate columns to float/int
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            
    return df

def log_failure(stage, reason, cow_id):
    """Logs a failure message for a specific processing stage."""
    msg = f"FAILED Quality Gate - Stage: {stage}, Cow ID: {cow_id}, Reason: {reason}"
    logger.warning(msg)

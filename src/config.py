"""
Central configuration: paths, column names, model params.
Import from here everywhere else instead of hardcoding values in each script.
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "vehicles.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "car_price_pipeline.pkl"

# Columns
TARGET = "price"

# Drop useless and high-missing columns
DROP_COLS = [
    'Unnamed: 0',
    'id',
    'url',
    'region_url',
    'VIN',
    'county',
    'image_url',
    'description',
    'size'
]

# Numerical columns
RAW_NUMERIC = ['year', 'odometer', 'lat', 'long']

# Categorical columns
CATEGORICAL = [
    'region', 'manufacturer', 'model', 'condition', 'cylinders', 
    'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color', 
    'state', 'posting_date'
    ]

# Domain sanity bounds
PRICE_MIN, PRICE_MAX = 500, 150_000
YEAR_MIN, YEAR_MAX_OFFSET = 1950, 1     # max = current_year + offset
ODOMETER_MIN, ODOMETER_MAX = 0, 500_000

# Model/ Pipeline parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
RARE_CATEGORY_TOP_N = 30
FEATURE_SELECTION_K = 50
N_ESTIMATORS = 300
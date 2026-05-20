from pathlib import Path

SEED = 2026

R = 0.04
Q_BASELINE = 0.0117

TICKER_RE = r'SPX US (\d{2}/\d{2}/\d{2}) ([CP])(\d+) Index'
COL_SUFFIX_RE = r'\s*\([RL]\d+\)$'

QUALITY_THRESHOLDS = {
    'iv_min': 0.95,
    'delta_min': 0.95,
    'gamma_min': 0.95,
    'mid_full_threshold': 0.80,
}

WINDOW_A_EXPIRIES = ['06/18/26', '12/18/26']

WINDOW_C_MIN_DAYS = 900

Q_MIN_DTAU = 0.5

TRAIN_VAL_TEST = (0.70, 0.15, 0.15)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLOOMBERG_DIR = PROJECT_ROOT / 'data_raw' / 'Bloomberg'
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
FIGURES_DIR = DATA_DIR / 'figures'
SYNTHETIC_DIR = DATA_DIR / 'synthetic'
RAW_XLSX = BLOOMBERG_DIR / 'spx_2.xlsx'
RAW_XLSX_BACKUP = BLOOMBERG_DIR / 'spx_1.xlsx'

from .utils import load_dataset as _load_dataset

# Load datasets
TD = _load_dataset("TD.TO", "Date")
COCA_COLA = _load_dataset("KO", "Date")
NASDAQ_TRAIN = _load_dataset("NASDAQ_TRAIN", "Date")
NASDAQ_TEST = _load_dataset("NASDAQ_TEST", "Date")

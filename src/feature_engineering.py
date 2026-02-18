import pandas as pd
import os
import logging
import yaml

# ================= LOGGING =================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ================= UTILS =================
def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logger.debug("Parameters loaded from %s", params_path)
    return params


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug("Data loaded from %s | Shape: %s", file_path, df.shape)
    return df


# ================= FEATURE ENGINEERING =================
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep raw text and labels only.
    TF-IDF will be applied inside model pipeline.
    """
    if 'text' not in df.columns or 'target' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'target' columns")

    df = df[['text', 'target']]
    logger.debug("Feature engineering completed (raw text retained)")
    return df


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.debug("Processed data saved to %s", path)


# ================= MAIN =================
def main():
    load_params('params.yaml')  # optional, kept for pipeline consistency

    train_data = load_data('./data/interim/train_processed.csv')
    test_data = load_data('./data/interim/test_processed_data.csv')

    train_df = prepare_features(train_data)
    test_df = prepare_features(test_data)

    save_data(train_df, './data/processed/train.csv')
    save_data(test_df, './data/processed/test.csv')


if __name__ == '__main__':
    main()

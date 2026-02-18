import os
import pandas as pd
import pickle
import logging
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ================= LOGGING =================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))
file_handler.setFormatter(formatter)

# Avoid duplicate logs
if not logger.handlers:
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
    logger.debug("Data loaded from %s | Shape: %s", file_path, df.shape)
    return df


# ================= MODEL TRAINING =================
def train_pipeline(X_train, y_train, params):
    """
    Train a full TEXT → TF-IDF → RandomForest pipeline
    """

    logger.debug("Creating ML pipeline")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=params["max_features"],
                ngram_range=tuple(params["ngram_range"]),
                stop_words="english"
            )
        ),
        (
            "clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
            ))
        
    ])

    logger.debug("Training pipeline started")
    pipeline.fit(X_train, y_train)
    logger.debug("Training pipeline completed")

    return pipeline


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.debug("Model pipeline saved at %s", path)


# ================= MAIN =================
def main():
    logger.debug("Model building stage started")

    params = load_params('params.yaml')['model_building']

    train_data = load_data('./data/processed/train.csv')

    # ✅ CRITICAL FIX: handle NaN + enforce string type
    X_train = (
        train_data['text']
        .fillna("")
        .astype(str)
    )

    y_train = train_data['target']

    logger.debug("Training data prepared | Samples: %d", len(X_train))

    model = train_pipeline(X_train, y_train, params)

    save_model(model, 'models/model.pkl')

    logger.debug("Model building stage completed successfully")


if __name__ == '__main__':
    main()

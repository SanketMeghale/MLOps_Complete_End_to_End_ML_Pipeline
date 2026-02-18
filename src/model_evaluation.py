import os
import json
import pickle
import logging
import pandas as pd
import yaml

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live

# ================= LOGGING =================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ================= UTILS =================
def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    logger.debug('Parameters retrieved from %s', params_path)
    return params


def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    logger.debug('Model loaded from %s', file_path)
    return model


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logger.debug('Data loaded from %s | Shape: %s', file_path, df.shape)
    return df


# ================= EVALUATION =================
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate pipeline model on raw text with string labels
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label='spam'),
        'recall': recall_score(y_test, y_pred, pos_label='spam')
    }

    # AUC only if model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

        # Find index of 'spam' class
        spam_index = list(model.classes_).index('spam')
        metrics['auc'] = roc_auc_score(y_test, y_proba[:, spam_index])

    return metrics



def save_metrics(metrics: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    logger.debug('Metrics saved to %s', file_path)


# ================= MAIN =================
def main():
    try:
        params = load_params('params.yaml')

        model = load_model('./models/model.pkl')

        # ✅ LOAD RAW TEXT DATA (NOT TF-IDF)
        test_data = load_data('./data/processed/test.csv')

        # ✅ CRITICAL SAFETY
        X_test = test_data['text'].fillna("").astype(str)
        y_test = test_data['target']

        metrics = evaluate_model(model, X_test, y_test)

        # ✅ DVCLive logging
        with Live(save_dvc_exp=True) as live:
            for key, value in metrics.items():
                live.log_metric(key, value)

            live.log_params(params)

        # ✅ JSON metrics for DVC
        save_metrics(metrics, 'reports/metrics.json')

        logger.debug("Model evaluation completed successfully")

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

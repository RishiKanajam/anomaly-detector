import os
import joblib
import numpy as np

def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models = {
        'iso_forest': joblib.load(os.path.join(model_dir, 'iso_forest.pkl')),
        'svm': joblib.load(os.path.join(model_dir, 'svm.pkl')),
        'lof': joblib.load(os.path.join(model_dir, 'lof.pkl')),
        'scaler': joblib.load(os.path.join(model_dir, 'scaler.pkl')),
    }
    return models

def predict(model_name, X_scaled, models):
    if model_name == 'iso_forest':
        scores = -models['iso_forest'].decision_function(X_scaled)
        preds = models['iso_forest'].predict(X_scaled)

    elif model_name == 'svm':
        scores = -models['svm'].decision_function(X_scaled)
        preds = models['svm'].predict(X_scaled)

    elif model_name == 'ensemble':
        iso_pred = models['iso_forest'].predict(X_scaled)
        svm_pred = models['svm'].predict(X_scaled)
        lof_pred = models['lof'].fit_predict(X_scaled)
        votes = np.vstack([iso_pred, svm_pred, lof_pred]).T
        preds = np.where(np.sum(votes == -1, axis=1) >= 2, -1, 1)
        scores = np.mean(votes == -1, axis=1)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return preds, scores

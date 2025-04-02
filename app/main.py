# app/main.py

from flask import Flask, render_template, request
from utils import load_models, predict
import pandas as pd

app = Flask(__name__)
models = load_models()  # ✅ load once at startup

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        model_name = request.form['model']
        df = pd.read_csv(file)

        # Optional: hide noisy/unwanted columns in display
        HIDE_COLUMNS = ['userAgent', 'timestamp', 'userIdentity.userName']
        df_display = df.drop(columns=[col for col in HIDE_COLUMNS if col in df.columns])

        # Get only features used in training
        trained_columns = models['scaler'].feature_names_in_
        X = df_display[trained_columns]
        X_scaled = models['scaler'].transform(X)

        # Predict using selected model
        preds, scores = predict(model_name, X_scaled, models)
        df_display['Prediction'] = preds
        df_display['Anomaly Score'] = scores

        anomaly_count = (preds == -1).sum()

        return render_template("index.html",
            model=model_name,
            data=df_display.to_dict(orient='records'),
            columns=df_display.columns,
            scores=scores.tolist(),
            labels=preds.tolist(),
            anomaly_count=anomaly_count
        )

    # GET request — render empty UI
    return render_template("index.html",
        model=None,
        data=[],
        columns=[],
        scores=[],
        labels=[],
        anomaly_count=0
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

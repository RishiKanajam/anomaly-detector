# 🚨 AI-Powered Log Anomaly Detection System (AIOps)

This is a Flask-based web application that allows users to upload cloud log files (CSV format) and detect anomalies using multiple machine learning models. It supports **Isolation Forest**, **One-Class SVM**, and **an ensemble approach**.

Hosted with Docker and easily deployable to [Render](https://render.com).

---

## 🔍 Features

- Upload CSV log files and analyze them for anomalies
- Choose from:
  - Isolation Forest
  - One-Class SVM
  - Ensemble (Voting)
- Visualize anomaly scores (color-coded)
- Filter results to show only anomalies
- Download results as CSV
- Built-in UI with Tailwind CSS and Plotly
- Containerized with Docker for production use

---

## 📈 Models and Ensemble

| Model            | Method                                        | Score Range        |
|------------------|-----------------------------------------------|--------------------|
| Isolation Forest | Distance from learned normal clusters         | Typically ~[0, 10] |
| One-Class SVM    | Distance from decision boundary (hyperplane)  | Can be large (e.g., 3400+) |
| Ensemble         | Voting score from 3 models                    | 0.0 = normal, 1.0 = anomaly |

> Scores differ because each model uses a different math technique to calculate "distance from normal."

## ⚠️ Notes

- models/ is not committed to GitHub due to large file limits
- Models are included in the Docker image at build time
- LOF and Autoencoder models are optional and excluded in deployment
- Only iso_forest.pkl, svm.pkl, and scaler.pkl are needed

## 📊 Understanding Score Differences
Why are scores different across models?

Each model uses a different method:

- Isolation Forest: based on how easily a point is isolated (lower is more normal)
- One-Class SVM: distance from learned boundary (may have large numeric values)
- Ensemble: confidence from majority vote (0.0 = normal, 1.0 = anomaly)
- These scores are not directly comparable — treat each one as a relative indicator within its own model.

📧 Contact
Made by @RishiKanajam with all the debugging help I can get 😉

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



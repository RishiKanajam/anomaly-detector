# train_models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("/Users/rishikanajam/Downloads/archive/nineteenFeaturesDf.csv")
X = df.select_dtypes(include='number')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler.feature_names_in_ = X.columns  # attach column names to scaler
joblib.dump(scaler, "models/scaler.pkl")

# 1. Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_scaled)
joblib.dump(iso, 'models/iso_forest.pkl')

# 2. One-Class SVM
svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm.fit(X_scaled)
joblib.dump(svm, 'models/svm.pkl')

# 3. Local Outlier Factor (fit_predict only, can't be used alone for API)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_pred = lof.fit_predict(X_scaled)
joblib.dump(lof, 'models/lof.pkl')  # Note: LOF has limited reuse.

# 4. Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
X_train, X_val = train_test_split(X_tensor, test_size=0.2, random_state=42)

model = Autoencoder(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    output = model(X_train)
    loss = criterion(output, X_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/autoencoder.pth")

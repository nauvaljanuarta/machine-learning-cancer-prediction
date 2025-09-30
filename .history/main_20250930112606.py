import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing & ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("risk_factors_cervical_cancer.csv")

# 2. Cek info awal
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df.head())

# 3. Tangani missing value → ganti dengan median
df = df.replace("?", np.nan).astype(float)
df = df.fillna(df.median())

# 4. Tentukan fitur (X) dan target (y)
X = df.drop(columns=["Biopsy"])   # semua kolom kecuali target
y = df["Biopsy"]                  # target → hasil biopsi kanker serviks (0/1)

# 5. Normalisasi fitur numerik
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split data dulu sebelum SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Terapkan SMOTE hanya pada data latih
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE :", np.bincount(y_train_res))

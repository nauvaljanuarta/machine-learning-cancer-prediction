import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML Tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("risk_factors_cervical_cancer.csv")

# Ganti string "?" jadi NaN lalu ubah ke numerik
df.replace("?", np.nan, inplace=True)
df = df.astype(float)

# Isi NaN dengan median
df.fillna(df.median(), inplace=True)

print("Shape dataset:", df.shape)
print("5 data teratas:\n", df.head())

# ===============================
# 2. Tentukan Fitur (X) dan Target (y)
# ===============================
X = df.drop(columns=["Biopsy"])
y = df["Biopsy"]

print("\nDistribusi awal target Biopsy (0 = sehat, 1 = terkena kanker):")
print(y.value_counts())
print("\nPersentase distribusi target:")
print(y.value_counts(normalize=True) * 100)

# ===============================
# 3. Normalisasi Min-Max
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 4. Seleksi Fitur (ANOVA, ambil 10 terbaik)
# ===============================
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]

print("\nFitur terpilih dari ANOVA:")
print(selected_features)

# ===============================
# 5. Split Data (Train/Test)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print("\nJumlah data train:", len(X_train))
print("Jumlah data test :", len(X_test))

# ===============================
# 6. Distribusi Target Sebelum SMOTE
# ===============================
plt.figure(figsize=(6,4))
y_train.value_counts().plot(
    kind="bar", color=["skyblue","salmon"], edgecolor="black"
)
plt.title("Distribusi Target Biopsy (Data Train) Sebelum SMOTE")
plt.xlabel("Kelas (0=Sehat, 1=Kanker)")
plt.ylabel("Jumlah Pasien")
plt.xticks(rotation=0)
plt.show()

# ===============================
# 7. Balancing Data dengan SMOTE
# ===============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nDistribusi setelah SMOTE:")
print(pd.Series(y_train_res).value_counts())

# Visualisasi setelah SMOTE
plt.figure(figsize=(6,4))
pd.Series(y_train_res).value_counts().plot(
    kind="bar", color=["skyblue","salmon"], edgecolor="black"
)
plt.title("Distribusi Target Biopsy (Data Train) Sesudah SMOTE")
plt.xlabel("Kelas (0=Sehat, 1=Kanker)")
plt.ylabel("Jumlah Pasien")
plt.xticks(rotation=0)
plt.show()

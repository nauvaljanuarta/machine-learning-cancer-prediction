import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# 1. Membaca Data
df = pd.read_csv("risk_factors_cervical_cancer.csv")

# Ganti string anomali dengan NaN
df.replace(['?', ' ', ''], np.nan, inplace=True)

# Konversi ke numerik
df = df.apply(pd.to_numeric, errors='coerce')

print("5 data teratas:")
print(df.head())

# 2. Cek Missing Values
print("\nCek Missing Values:")
print(df.isnull().sum())


# 3. Handling Missing Values

# Isi NaN dengan median per kolom
df.fillna(df.median(), inplace=True)

print("\nSetelah handling missing values:")
print(df.isnull().sum())

# 4. Normalisasi Min-Max
numeric_cols = df.drop("Biopsy", axis=1).columns

# Dengan Library
scaler = MinMaxScaler()
df_minmax_lib = df.copy()
df_minmax_lib[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n=== Normalisasi Min-Max (Library), 5 baris pertama ===")
print(df_minmax_lib[numeric_cols].head())

# Dengan Rumus Manual
df_minmax_manual = df.copy()
for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    df_minmax_manual[col] = (df[col] - min_val) / (max_val - min_val)

print("\n=== Normalisasi Min-Max (Manual), 5 baris pertama ===")
print(df_minmax_manual[numeric_cols].head())

# 5. Seleksi Fitur dengan ANOVA
X = df_minmax_lib.drop("Biopsy", axis=1)
y = df_minmax_lib["Biopsy"]

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
mask = selector.get_support()
selected_features = X.columns[mask]

print("\nFitur terpilih dari seleksi ANOVA:")
print(selected_features)

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.2, random_state=1, stratify=y
)

print("\nJumlah data train:", X_train.shape[0])
print("Jumlah data test:", X_test.shape[0])

# ===============================
# 7. SMOTE Balancing (hanya train)
# ===============================
print("\nDistribusi kelas sebelum SMOTE (data train):")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nDistribusi kelas sesudah SMOTE (data train):")
print(pd.Series(y_train_res).value_counts())

# Visualisasi distribusi kelas sebelum & sesudah SMOTE
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

y_train.value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["skyblue", "salmon"])
axes[0].set_title("Distribusi Kelas Train Sebelum SMOTE")
axes[0].set_xlabel("Kelas")
axes[0].set_ylabel("Jumlah Sampel")

pd.Series(y_train_res).value_counts().sort_index().plot(kind="bar", ax=axes[1], color=["skyblue", "salmon"])
axes[1].set_title("Distribusi Kelas Train Sesudah SMOTE")
axes[1].set_xlabel("Kelas")
axes[1].set_ylabel("Jumlah Sampel")

plt.tight_layout()
plt.show()

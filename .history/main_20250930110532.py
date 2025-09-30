import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE


df = pd.read_csv("risk_factors_cervical_cancer.csv")

# mengganti string anomali jadi NaN
df.replace(['?', ' ', ''], np.nan, inplace=True)
# konversi ke numerik
df = df.apply(pd.to_numeric, errors='coerce')
print("Jumlah NaN tiap kolom:\n", df.isna().sum().sum())

# ganti NaN dengan median
df.fillna(df.median(), inplace=True)
print("\npenggantian nan dengan median")
print("data nan menjadi : \n", df.isna().sum().sum())

target_col = 'Biopsy'
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
print("\nmapping ke kelas target")
print("distribusi kelas:\n", df[target_col].value_counts())

# pembagian targrt dan fitur 
X = df.drop(target_col, axis=1)
y = df[target_col]
print("data feature")
print(X.columns.to_list)

# normalisasi minmax
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("\ndata setelah normalisasi minmax")
print (X_scaled)

# menghilangkan fitur konstan
var_thresh = VarianceThreshold(threshold=0.0)
X_var = var_thresh.fit_transform(X_scaled)
mask_var = var_thresh.get_support()
selected_features_var = X.columns[mask_var]

#= ANOVA
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_var, y)
mask_anova = selector.get_support()
final_features = selected_features_var[mask_anova]
print("\nfitur terpilih dari seleksi ANOVA:\n", final_features)
print("\ndata fitur hasil seleksi ANOVA:\n", X_selected)


# balancing data dengan SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_selected, y)
print("\ndistribusi kelas setelah SMOTE:")
print(pd.Series(y_res).value_counts())

# split data training dan test
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=1, stratify=y_res
)
print("\nJumlah data train:", X_train.shape[0])
print("Jumlah data test:", X_test.shape[0])

# visualisasi
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# sebelum smote 
pd.Series(y).value_counts().sort_index().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
axes[0].set_title('Distribusi Kelas Sebelum SMOTE')
axes[0].set_xlabel('Kelas')
axes[0].set_ylabel('Jumlah Sampel')

# setelah smote
pd.Series(y_res).value_counts().sort_index().plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])
axes[1].set_title('Distribusi Kelas Setelah SMOTE')
axes[1].set_xlabel('Kelas')
axes[1].set_ylabel('Jumlah Sampel')

plt.tight_layout()
plt.show()

# visualisasi fitur dari seleksi anova
X_train_anova = X_train[:, :2]  # ambil 2 kolom pertama dari fitur hasil seleksi ANOVA

plt.figure(figsize=(6, 5))
plt.scatter(X_train_anova[y_train == 0, 0], X_train_anova[y_train == 0, 1],
            alpha=0.6, label='Kelas 0', c='skyblue', edgecolor='k')
plt.scatter(X_train_anova[y_train == 1, 0], X_train_anova[y_train == 1, 1],
            alpha=0.6, label='Kelas 1', c='salmon', edgecolor='k')

plt.title("Visualisasi Data Train (2 Fitur Terbaik ANOVA)")
plt.xlabel("Fitur 1 (ANOVA)")
plt.ylabel("Fitur 2 (ANOVA)")
plt.legend()
plt.tight_layout()
plt.show()

# automate_FuadHasanWirayudha.py
# Automated Data Loading, EDA, and Preprocessing
# Author: Fuad Hasan Wirayudha

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("=== AUTOMATIC SCRIPT STARTED ===")

# ============================================================
# 1. LOAD DATASET
# ============================================================
FILE_NAME = "Pima Indians Diabetes Database.csv"

try:
    df = pd.read_csv(FILE_NAME)
    print(f"[OK] Dataset berhasil dimuat: {FILE_NAME}")
except FileNotFoundError:
    print(f"[ERROR] File '{FILE_NAME}' tidak ditemukan.")
    exit()

# ============================================================
# 2. EDA DASAR
# ============================================================
print("\n=== EDA ===")
print("\n📌 5 data teratas:")
print(df.head())

print("\n📌 Info data:")
print(df.info())

print("\n📌 Statistik deskriptif:")
print(df.describe())

print("\n📌 Cek missing values:")
print(df.isnull().sum())

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
print("\n=== DATA PREPROCESSING ===")

# --- 3.1 Tangani Missing Values ---
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print("[OK] Missing values telah ditangani (mean imputation).")

# --- 3.2 Tangani Duplikasi ---
dupe_count = df_imputed.duplicated().sum()
df_imputed = df_imputed.drop_duplicates()
print(f"[OK] {dupe_count} data duplikat telah dihapus.")

# --- 3.3 Scaling (Standardization) ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_imputed)

df_scaled = pd.DataFrame(scaled_data, columns=df_imputed.columns)
print("[OK] Fitur numerik telah dinormalisasi (StandardScaler).")

# ============================================================
# 4. SAVE CLEANED DATA
# ============================================================
CLEAN_FILE = "cleaned_pima_diabetes_fuad.csv"
df_scaled.to_csv(CLEAN_FILE, index=False)

print(f"\n[OK] Data hasil preprocessing telah disimpan sebagai: {CLEAN_FILE}")
print("\n=== SCRIPT COMPLETED ===")


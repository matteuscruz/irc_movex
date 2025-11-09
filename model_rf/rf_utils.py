import os
import pandas as pd
import numpy as np
from glob import glob
from typing import Optional, List, Dict
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

# Funções auxiliares para carregamento de dados
def find_csv_files(quarter_path: str) -> List[str]:
    return glob(os.path.join(quarter_path, "*.csv"))

def load_csv_file(file_path: str, experiment: str, quarter: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['experiment'] = experiment
    df['quarter'] = quarter
    return df

def load_and_concat_csv_season(experiment_folder: str) -> Optional[pd.DataFrame]:
    dfs = []
    quarter_paths = glob(os.path.join(experiment_folder, "Trimestre_*"))
    for quarter_path in quarter_paths:
        quarter = os.path.basename(quarter_path).split("_")[-1]
        csv_files = find_csv_files(quarter_path)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda file: load_csv_file(file, os.path.basename(experiment_folder), quarter),
                csv_files
            ))
        dfs.extend(results)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def load_all_experiments(experiment_glob_pattern: str) -> Optional[pd.DataFrame]:
    all_dfs = []
    experiment_folders = glob(experiment_glob_pattern)
    print(f"🔍 Experimentos encontrados: {len(experiment_folders)}")
    for exp_folder in tqdm(experiment_folders, desc="⚙️ Processando experimentos"):
        result_df = load_and_concat_csv_season(exp_folder)
        if result_df is not None:
            all_dfs.append(result_df)
        else:
            print(f"⚠️ Nenhum dado carregado para: {exp_folder}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

def filter_experiment_data(df: pd.DataFrame, experiment: str, lat: float, lon: float, season: int) -> pd.DataFrame:
    return df[
        (df['experiment'] == experiment) &
        (df['latitude'] == lat) &
        (df['longitude'] == lon) &
        (df['season'] == season)
    ]

def load_and_prepare_gmt(gmt_path: str) -> pd.DataFrame:
    df_gmt = pd.read_csv(gmt_path)
    df_gmt['Date'] = pd.to_datetime(df_gmt['Date'])
    df_gmt['meteorological_year'] = df_gmt['Date'].dt.year
    df_gmt['season'] = df_gmt['Date'].dt.quarter
    return df_gmt

def merge_with_gmt(df_target: pd.DataFrame, df_gmt: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(df_target, df_gmt, how='left', on=['meteorological_year', 'season'])
    merged = merged.dropna(axis=1)
    merged = merged.rename(columns={'Monthly_Anomaly_x': 'GMT'})
    return merged.sort_values(by=['meteorological_year', 'season'])

# Função de Regressão Linear (Out-of-sample)
def out_of_sample_regression(df: pd.DataFrame, target_column: str, feature_column: str) -> np.ndarray:
    df = df.sort_values(by='meteorological_year')
    years = df['meteorological_year'].unique()
    y_pred = []
    for year in years:
        train_data = df[df['meteorological_year'] != year]
        test_data = df[df['meteorological_year'] == year]
        X_train = train_data[[feature_column]]
        y_train = train_data[target_column]
        X_test = test_data[[feature_column]]
        y_test = test_data[target_column]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_year = model.predict(X_test)
        y_pred.append(y_pred_year[0])
    return np.array(y_pred)

# Função para calcular o R²
def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("As dimensões de 'y_true' e 'y_pred' devem ser iguais.")
    mean_y_true = y_true.mean()
    SStot = np.sum((y_true - mean_y_true) ** 2)
    SSres = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (SSres / SStot) if SStot != 0 else np.nan
    if np.isinf(r2):
        r2 = np.nan
    return r2

# Função para out-of-sample com Random Forest
def out_of_sample_random_forest(df: pd.DataFrame, target_column: str, feature_columns: List[str], rf_params: Optional[Dict] = None) -> np.ndarray:
    if rf_params is None:
        rf_params = {}
    df = df.sort_values(by='meteorological_year')
    years = df['meteorological_year'].unique()
    y_pred = []
    for year in years:
        train_data = df[df['meteorological_year'] != year]
        test_data = df[df['meteorological_year'] == year]
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        y_pred_year = model.predict(X_test)
        y_pred.append(y_pred_year[0])
    return np.array(y_pred)

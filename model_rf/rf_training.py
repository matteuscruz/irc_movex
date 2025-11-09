import os
import gc
import time
from tqdm import tqdm
from itertools import combinations

import cudf
import cupy as cp
import numpy as np
from cuml.ensemble import RandomForestRegressor as cuRF

def process_location_gpu(group_name, group_data, ano_alvo, trimestre_alvo, feature_cols):
    lat, lon = group_name

    df_treinamento = group_data[
        (group_data['meteorological_year'] != ano_alvo) &
        (group_data['season'] == trimestre_alvo)
    ]
    df_teste = group_data[
        (group_data['meteorological_year'] == ano_alvo) &
        (group_data['season'] == trimestre_alvo)
    ]

    if len(df_treinamento) == 0 or len(df_teste) == 0:
        return None

    X_train = df_treinamento[feature_cols]
    y_train = df_treinamento['pr']
    X_test = df_teste[feature_cols]
    y_test = df_teste['pr']

    min_samples_split = max(2, int(len(X_train) * 0.5))
    min_samples_leaf = max(1, int(len(X_train) * 0.1))

    model = cuRF(
        n_estimators=1,
        max_depth=1,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_bins=16
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = []
    test_data_dict = X_test.to_pandas().to_dict('records')
    y_test_cpu = y_test.to_pandas()
    y_pred_cpu = cp.asnumpy(y_pred)

    for i, (true_val, pred_val) in enumerate(zip(y_test_cpu, y_pred_cpu)):
        result_item = {
            'latitude': lat,
            'longitude': lon,
            'y_true': true_val,
            'y_pred': pred_val,
            'meteorological_year': int(ano_alvo),
            'season': int(trimestre_alvo),
        }
        result_item.update(test_data_dict[i])
        results.append(result_item)

    return results

def main():
    fixed_features = ['Nino4_Indice', 'Monthly_Anomaly']
    variable_features_to_test = [
        'SOI_ANOMALY_Indice', 'SAM_Indice', 'NAO_Indice', 'ATL3',
        'IOD_Indice', 'TNA_Indice', 'MJO_RMM1', 'MJO_RMM2'
    ]

    all_feature_combinations = []
    for r in range(1, 3):
        combos_r = combinations(variable_features_to_test, r)
        all_feature_combinations.extend(list(combos_r))

    all_features = fixed_features + variable_features_to_test
    required_cols = ['date', 'latitude', 'longitude', 'pr'] + all_features

    df = cudf.read_parquet('data/merged_df_v2.parquet', columns=required_cols)

    lat_min, lat_max = -28, -23
    lon_min, lon_max = -55, -48
    df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
            (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)].copy()

    if len(df) == 0:
        raise ValueError("No data found in the specified coordinate range.")

    df['date'] = cudf.to_datetime(df['date'])
    df = df.sort_values(by='date')

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    season_map_numeric = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    season_map_series = cudf.Series(season_map_numeric)
    df['season'] = df['month'].map(season_map_series)
    df['meteorological_year'] = cp.where(df['month'] == 12, df['year'] + 1, df['year'])

    df.drop(columns=['date', 'year', 'month'], inplace=True)
    gc.collect()

    anos_disponiveis = df['meteorological_year'].unique().to_pandas().sort_values().tolist()
    trimestres_disponiveis = df['season'].unique().to_pandas().sort_values().tolist()
    season_map_names = {1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'}

    grouped_data = dict(tuple(df.groupby(['latitude', 'longitude'])))

    for combo_tuple in tqdm(all_feature_combinations, desc="Overall Experiments"):
        combo_list = list(combo_tuple)
        feature_cols = fixed_features + combo_list
        experiment_name = '-'.join(combo_list) if combo_list else 'no_variable'

        print(f"\n----- Running Experiment for: {experiment_name} -----")
        print(f"Features for this run: {feature_cols}")

        base_output_dir = f'resultados_{experiment_name}'

        for ano_alvo in tqdm(anos_disponiveis, desc=f"Years ({experiment_name})", leave=False):
            for trimestre_alvo in tqdm(trimestres_disponiveis, desc=f"Seasons ({ano_alvo})", leave=False):
                resultados_trimestre = []

                for name, group in grouped_data.items():
                    res = process_location_gpu(name, group, ano_alvo, trimestre_alvo, feature_cols)
                    if res:
                        resultados_trimestre.extend(res)

                if resultados_trimestre:
                    season_name = season_map_names.get(trimestre_alvo, f"Trimestre_{trimestre_alvo}")
                    season_output_dir = os.path.join(base_output_dir, f"Trimestre_{season_name}")
                    os.makedirs(season_output_dir, exist_ok=True)

                    resultados_df = cudf.DataFrame(resultados_trimestre)
                    filename = os.path.join(season_output_dir, f'predicoes_{ano_alvo}.csv')
                    resultados_df.to_pandas().to_csv(filename, index=False)

if __name__ == '__main__':
    main()
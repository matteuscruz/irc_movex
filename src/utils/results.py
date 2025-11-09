import pandas as pd
import os
import logging

def format_prediction_result(lat, lon, true, pred, ano_alvo, quadrimestre_alvo,
                             feature_set, X_test, i, cluster):
    """Formata um resultado individual de predição"""
    result = {
        'latitude': lat,
        'longitude': lon,
        'y_true': true,
        'y_pred': pred,
        'meteorological_year': ano_alvo,
        'quarter': quadrimestre_alvo,
        'movs_used': '+'.join(feature_set),
        'k_means_cluster': cluster
    }
    for feature in feature_set:
        result[feature] = X_test[feature].iloc[i]
    return result

def save_results(resultados_quadrimestre, season, ano_alvo):
    """Salva resultados em arquivo CSV"""
    if resultados_quadrimestre:
        os.makedirs(f'resultados_predicoes_{season}', exist_ok=True)
        resultados_df = pd.DataFrame(resultados_quadrimestre)
        filename = f'resultados_predicoes_{season}/predicoes_{ano_alvo}_{season}.csv'
        resultados_df.to_csv(filename, index=False)
        logging.info(f"Salvo: {filename} ({len(resultados_df)} linhas)")
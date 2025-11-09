import pandas as pd
import logging
from src.config.settings import DTYPES, QUARTER_MAP

class DataProcessor:
    """Classe responsável pelo processamento e preparação dos dados"""

    def __init__(self):
        self.season_list = ['DJF', 'MAM', 'JJA', 'SON']
        self.quarter_map = QUARTER_MAP

    def filter_data_by_season_year(self, df, season, ano_alvo, selected_pairs=None):
        df_filtered = df[df['quarter'] == season]

        if selected_pairs:
            kmeans_col = f"{season}_k_means"
            movs_col = f"{season}_movs"

            mask = pd.Series(False, index=df_filtered.index)
            for movs_val, kmeans_val in selected_pairs:
                mask |= (
                        (df_filtered[kmeans_col] == kmeans_val) &
                        (df_filtered[movs_col] == movs_val)
                )

            df_filtered = df_filtered[mask]

        return df_filtered

    def group_data_by_coordinates(self, df_filtered):
        """Agrupa dados por coordenadas"""
        grouped_data = list(df_filtered.groupby(['latitude', 'longitude']))
        return grouped_data

    def get_available_years(self, df):
        """Retorna anos disponíveis no dataset"""
        anos_disponiveis = sorted(df['meteorological_year'].unique())
        return anos_disponiveis
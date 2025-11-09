"""
Módulo para transformações de séries temporais.
"""

import pandas as pd
from typing import List, Optional

from ..config.settings import config


class TimeSeriesTransformer:
    """Classe para transformações de séries temporais."""

    def __init__(self, n_lags: int = None, n_forecast_steps: int = None):
        """
        Inicializa o transformador de séries temporais.

        Args:
            n_lags: Número de lags (passos passados)
            n_forecast_steps: Número de passos futuros para previsão
        """
        self.n_lags = n_lags or config.model.n_lags
        self.n_forecast_steps = n_forecast_steps or config.model.n_forecast_steps

    def series_to_supervised(self,
                             df: pd.DataFrame,
                             target_col: str,
                             feature_cols: Optional[List[str]] = None,
                             n_in: Optional[int] = None,
                             n_out: Optional[int] = None,
                             dropnan: bool = True) -> pd.DataFrame:
        """
        Converte um DataFrame de série temporal em formato supervisionado.

        Args:
            df: DataFrame contendo features + coluna alvo
            target_col: Nome da coluna alvo (ex: 'pr')
            feature_cols: Lista de colunas de features (se None, usa todas exceto target)
            n_in: Número de lags (passos passados)
            n_out: Número de passos futuros
            dropnan: Remove linhas com NaN

        Returns:
            DataFrame em formato supervisionado
        """
        n_in = n_in or self.n_lags
        n_out = n_out or self.n_forecast_steps

        # Se não especificadas, usa todas as colunas exceto o target
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        # Verifica se as colunas existem
        missing_cols = set(feature_cols + [target_col]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas não encontradas no DataFrame: {missing_cols}")

        # Seleciona apenas as colunas necessárias
        df_work = df[feature_cols + [target_col]].copy()

        cols = []

        # Lags das features e target
        for i in range(n_in, 0, -1):
            lag_df = df_work.shift(i)
            # Renomeia colunas para incluir o lag
            lag_df.columns = [f"{col}_t-{i}" for col in lag_df.columns]
            cols.append(lag_df)

        # Valores atuais das features (t=0)
        current_features = df_work[feature_cols].copy()
        current_features.columns = [f"{col}_t0" for col in current_features.columns]
        cols.append(current_features)

        # Passos futuros do target
        for i in range(0, n_out):
            future_target = df_work[[target_col]].shift(-i)
            future_target.columns = [f"{target_col}_t+{i}"]
            cols.append(future_target)

        # Concatena todas as colunas
        agg = pd.concat(cols, axis=1)

        if dropnan:
            agg.dropna(inplace=True)

        return agg

    def create_lagged_features(self,
                               df: pd.DataFrame,
                               columns: List[str],
                               lags: List[int]) -> pd.DataFrame:
        """
        Cria features com lags específicos.

        Args:
            df: DataFrame original
            columns: Lista de colunas para criar lags
            lags: Lista de lags a serem criados

        Returns:
            DataFrame com features com lag
        """
        lagged_dfs = []

        for lag in lags:
            if lag == 0:
                # Valores atuais
                current_df = df[columns].copy()
                current_df.columns = [f"{col}_t0" for col in columns]
                lagged_dfs.append(current_df)
            else:
                # Valores com lag
                lag_df = df[columns].shift(lag)
                lag_df.columns = [f"{col}_t-{lag}" for col in columns]
                lagged_dfs.append(lag_df)

        # Concatena todos os DataFrames
        result = pd.concat(lagged_dfs, axis=1)
        return result

    def split_supervised_data(self,
                              supervised_df: pd.DataFrame,
                              target_col: str,
                              n_out: Optional[int] = None) -> tuple:
        """
        Separa dados supervisionados em X (features) e y (target).

        Args:
            supervised_df: DataFrame em formato supervisionado
            target_col: Nome base da coluna alvo
            n_out: Número de passos futuros

        Returns:
            Tupla (X, y) com features e targets
        """
        n_out = n_out or self.n_forecast_steps

        # Identifica colunas de target futuro
        target_cols = [f"{target_col}_t+{i}" for i in range(n_out)]

        # Verifica se as colunas existem
        missing_targets = set(target_cols) - set(supervised_df.columns)
        if missing_targets:
            raise ValueError(f"Colunas de target não encontradas: {missing_targets}")

        # Separa features e targets
        X = supervised_df.drop(columns=target_cols)
        y = supervised_df[target_cols]

        # Se apenas um passo futuro, retorna série ao invés de DataFrame
        if n_out == 1:
            y = y.iloc[:, 0]

        return X, y

    def add_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Adiciona features temporais baseadas em uma coluna de data.

        Args:
            df: DataFrame com dados
            date_col: Nome da coluna de data

        Returns:
            DataFrame com features temporais adicionadas
        """
        df_result = df.copy()

        # Converte para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(df_result[date_col]):
            df_result[date_col] = pd.to_datetime(df_result[date_col])

        date_series = df_result[date_col]

        # Features temporais básicas
        df_result['year'] = date_series.dt.year
        df_result['month'] = date_series.dt.month
        df_result['day'] = date_series.dt.day
        df_result['dayofweek'] = date_series.dt.dayofweek
        df_result['dayofyear'] = date_series.dt.dayofyear
        df_result['week'] = date_series.dt.isocalendar().week
        df_result['quarter'] = date_series.dt.quarter

        # Features cíclicas (sin/cos para capturar ciclicidade)
        df_result['month_sin'] = np.sin(2 * np.pi * df_result['month'] / 12)
        df_result['month_cos'] = np.cos(2 * np.pi * df_result['month'] / 12)
        df_result['day_sin'] = np.sin(2 * np.pi * df_result['day'] / 31)
        df_result['day_cos'] = np.cos(2 * np.pi * df_result['day'] / 31)

        return df_result

    def rolling_statistics(self,
                           df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int],
                           statistics: List[str] = None) -> pd.DataFrame:
        """
        Calcula estatísticas móveis para colunas especificadas.

        Args:
            df: DataFrame com dados
            columns: Lista de colunas para calcular estatísticas
            windows: Lista de janelas temporais
            statistics: Lista de estatísticas ('mean', 'std', 'min', 'max')

        Returns:
            DataFrame com estatísticas móveis adicionadas
        """
        if statistics is None:
            statistics = ['mean', 'std']

        df_result = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for window in windows:
                for stat in statistics:
                    if hasattr(df[col].rolling(window), stat):
                        new_col_name = f"{col}_rolling_{window}_{stat}"
                        df_result[new_col_name] = getattr(df[col].rolling(window), stat)()

        return df_result

try:
    import numpy as np
except ImportError:
    pass
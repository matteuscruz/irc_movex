from shapely import Point
from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import re
import logging
from tqdm import tqdm
from joblib import parallel_backend
from itertools import combinations
import optuna
from sklearn.model_selection import cross_val_score
import geopandas as gpd


def filter_sc_pr(grouped_data):
    """Filtra grupos de coordenadas que estão dentro de SC ou PR mantendo o formato original."""
    # Carrega o GeoJSON com todos os estados do Brasil
    brazil_states = gpd.read_file("gadm41_BRA_1.json")

    # Mantém só SC e PR (confirme os nomes no seu arquivo)
    sc_pr = brazil_states[
        brazil_states["NAME_1"].isin(["SantaCatarina", "Paraná"])
    ]

    filtered_groups = []

    for coords, df_group in grouped_data:
        lat, lon = coords  # confirme se é (lon, lat) mesmo
        point = Point(lon, lat)

        if sc_pr.contains(point).any():
            filtered_groups.append((coords, df_group))

    return filtered_groups


def process_location_wrapper(args):
    location_processor, group, ano_alvo, season = args
    return location_processor.process_location(group, ano_alvo, season)


class DataProcessor:
    """Classe responsável pelo processamento e preparação dos dados"""

    def __init__(self):
        self.season_list = ['DJF', 'MAM', 'JJA', 'SON']
        self.quarter_map = {
            12: 'DJF', 1: 'DJF', 2: 'DJF',
            3: 'MAM', 4: 'MAM', 5: 'MAM',
            6: 'JJA', 7: 'JJA', 8: 'JJA',
            9: 'SON', 10: 'SON', 11: 'SON'
        }

    def load_and_prepare_data(self, processed_path='data/MWC_v3_processed.parquet'):
        """Carrega e prepara os dados básicos, com cache no arquivo processado"""
        dtype = {
            # Timestamps (nanosegundos) - mantemos int64 pois são valores muito grandes
            'date': 'int64',
            'time': 'int64',
            'Date': 'int64',

            # Valores decimais com ~4 casas (float32 suficiente)
            'Nino4_Indice': 'float64',
            'SOI_ANOMALY_Indice': 'float64',
            'SAM_Indice': 'float64',
            'NAO_Indice': 'float64',
            'ATL3': 'float64',
            'IOD_Indice': 'float64',
            'TNA_Indice': 'float64',
            'MJO_RMM1': 'float64',
            'MJO_RMM2': 'float64',
            'latitude': 'float32',
            'longitude': 'float32',
            'pr': 'float64',
            'Monthly_Anomaly': 'float64',

            # Valores inteiros pequenos
            'year': 'int16',  # anos típicos: 1982-2023
            'month': 'int8',  # 1-12
            'meteorological_year': 'int16',

            # Cluster labels (valores observados até 21)
            'DJF_k_means': 'int8',  # valores até 21
            'JJA_k_means': 'int8',  # valores até 11
            'MAM_k_means': 'int8',  # valores até 11
            'SON_k_means': 'int8',  # valores até 5

            # Strings/categorias
            'DJF_movs': 'category',
            'JJA_movs': 'category',
            'MAM_movs': 'category',
            'SON_movs': 'category',
            'quarter': 'category'
        }

        if os.path.exists(processed_path):
            df = pd.read_parquet("data/MWC_v3_processed.parquet").astype(dtype)
            return df
        raw_path = input("Insira o dataset path :")
        # Caso não exista o processado, faz o pipeline completo
        logging.info("Arquivo processado não encontrado. Carregando bruto e processando...")

        df = pd.read_parquet(raw_path)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['month'].map(self.quarter_map)
        df['meteorological_year'] = df['year']
        df = df.astype(dtype)
        # Salva o resultado processado para uso futuro
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logging.info(f"Arquivo processado salvo em: {processed_path}")

        return df

    def get_available_years(self, df):
        """Retorna anos disponíveis no dataset"""
        anos_disponiveis = sorted(df['meteorological_year'].unique())
        return anos_disponiveis

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


class FeatureParser:
    """Classe responsável pelo parsing e processamento de features"""

    @staticmethod
    def parse_movs(movs_str):
        """Parse the MoVs string into individual feature names"""
        features = [f.strip() for f in re.split(r',|\+', movs_str)]
        return list(set([f for f in features if f]))

    @staticmethod
    def generate_feature_combinations(feature_cols):
        """Gera todas as combinações possíveis de features"""
        feature_cols_combs = []
        for r in range(1, len(feature_cols) + 1):
            feature_cols_combs.extend(combinations(feature_cols, r))
        return feature_cols_combs


class ModelTrainer:
    """Classe responsável pelo treinamento e predição dos modelos"""

    def __init__(self):
        # valores default, caso não rode a otimização
        self.model_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': 1
        }
        self.best_model = None

    def create_model(self, params=None):
        """Cria um novo modelo RandomForest"""
        if params is None:
            params = self.model_params
        return RandomForestRegressor(**params)

    def objective(self, trial, X_train, y_train):
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42,
            'n_jobs': 1
        }
        model = RandomForestRegressor(**params)

        # Garantir que y seja 1D
        y_array = y_train.values.ravel() if hasattr(y_train, "values") else y_train

        # Cross-validation com RMSE (negativo)
        scores = cross_val_score(model, X_train, y_array, cv=3, scoring='neg_root_mean_squared_error')

        # Inverter o sinal para ter RMSE positivo
        rmse = -np.mean(scores)
        return rmse

    def optimize_hyperparameters(self, X_train, y_train, n_trials=30):
        study = optuna.create_study(
            direction="minimize",
            study_name="random_forest_study",
            # storage="sqlite:///optuna_study.db",  # ou postgresql://user:pass@localhost/dbname
            # load_if_exists=True
        )
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)
        self.model_params.update(study.best_params)
        return study.best_params

    def train_and_predict(self, X_train, y_train, X_test):
        """Treina modelo e faz predições"""
        model = self.create_model()
        model.fit(X_train, y_train)
        self.best_model = model
        y_pred_test = model.predict(X_test)
        return y_pred_test


class ResultProcessor:
    """Classe responsável pelo processamento e formatação dos resultados"""

    @staticmethod
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

    @staticmethod
    def save_results(resultados_quadrimestre, season, ano_alvo):
        """Salva resultados em arquivo CSV"""
        if resultados_quadrimestre:
            os.makedirs(f'resultados_predicoes_{season}', exist_ok=True)
            resultados_df = pd.DataFrame(resultados_quadrimestre)
            filename = f'resultados_predicoes_{season}/predicoes_{ano_alvo}_{season}.csv'
            resultados_df.to_csv(filename, index=False)
            logging.info(f"Salvo: {filename} ({len(resultados_df)} linhas)")

class LocationProcessor:
    """Classe responsável pelo processamento de uma localização específica"""

    def __init__(self):
        self.feature_parser = FeatureParser()
        self.model_trainer = ModelTrainer()
        self.result_processor = ResultProcessor()

    def series_to_supervised(self, df, target_col, n_in=1, n_out=1, dropnan=True):
        """
        Converte um DataFrame de série temporal em formato supervisionado.

        df: DataFrame contendo features + coluna alvo
        target_col: nome da coluna alvo (ex: 'pr')
        n_in: número de lags (passos passados)
        n_out: número de passos futuros
        dropnan: remove linhas com NaN
        """
        cols = []

        # Lags das features
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i).add_suffix(f"_t-{i}"))

        # Passos futuros do target
        for i in range(0, n_out):
            cols.append(
                df[[target_col]].shift(-i).rename(columns={target_col: f"{target_col}_t+{i}"})
            )

        agg = pd.concat(cols, axis=1)

        if dropnan:
            agg.dropna(inplace=True)

        return agg

    def process_location(self, group, ano_alvo, quadrimestre_alvo):
        data = self.prepare_location_data(group, ano_alvo, quadrimestre_alvo)
        if data is None:
            return None
        return self.run_location_predictions(data, ano_alvo, quadrimestre_alvo)

    def prepare_location_data(self, group, ano_alvo, quadrimestre_alvo):
        lat = group['latitude'].iloc[0]
        lon = group['longitude'].iloc[0]

        quarter_group = group[group['quarter'] == quadrimestre_alvo]
        if quarter_group.empty:
            return None

        df_teste = quarter_group[quarter_group['meteorological_year'] == ano_alvo]
        df_treinamento = quarter_group[quarter_group['meteorological_year'] != ano_alvo]
        if df_teste.empty or df_treinamento.empty:
            return None

        movs_column = f"{quadrimestre_alvo}_movs"
        cluster = f"{quadrimestre_alvo}_k_means"
        if movs_column not in group.columns or group[movs_column].isnull().all():
            return None

        movs_str = group[movs_column].dropna().iloc[0]
        cluster = group[cluster].dropna().iloc[0]

        feature_cols = self.feature_parser.parse_movs(movs_str)
        feature_cols = [f for f in feature_cols if f in group.columns]
        if not feature_cols:
            return None

        feature_combinations = self.feature_parser.generate_feature_combinations(feature_cols)

        return {
            "lat": lat,
            "lon": lon,
            "df_treinamento": df_treinamento,
            "df_teste": df_teste,
            "feature_combinations": feature_combinations,
            "cluster": cluster
        }

    def run_location_predictions(self, data, ano_alvo, quadrimestre_alvo):
        results = []

        for feature_set in data["feature_combinations"]:
            feature_set = list(feature_set)
            try:
                X_train = data["df_treinamento"][feature_set]
                y_train = data["df_treinamento"]['pr']
                X_test = data["df_teste"][feature_set]
                y_test = data["df_teste"]['pr']

                self.model_trainer.optimize_hyperparameters(X_train, y_train, n_trials=50)

                y_pred_test = self.model_trainer.train_and_predict(X_train, y_train, X_test)
                for i, (true, pred) in enumerate(zip(y_test, y_pred_test)):
                    result = self.result_processor.format_prediction_result(
                        data["lat"], data["lon"], true, pred, ano_alvo, quadrimestre_alvo,
                        feature_set, X_test, i, data["cluster"]
                    )
                    results.append(result)
            except Exception as e:
                print(f"Ocorreu um erro no treinamento: {e}")
                continue

        return results

class PrecipitationPredictor:
    """Classe principal que coordena todo o processo de predição"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.location_processor = LocationProcessor()
        self.result_processor = ResultProcessor()
        self._setup_logging()

    def _setup_logging(self):
        """Configura o sistema de logging"""
        logging.basicConfig(
            format='[%(asctime)s] %(levelname)s: %(message)s',
            level=logging.INFO,
            datefmt='%H:%M:%S'
        )

    def run_predictions(self, file_path='archive/merged_with_clusters_v3.parquet'):
        df = self.data_processor.load_and_prepare_data()
        anos_disponiveis = self.data_processor.get_available_years(df)

        # Exibe as estações disponíveis
        print("Estações disponíveis:")
        for i, season in enumerate(self.data_processor.season_list, start=1):
            print(f"{i}. {season}")

        # Solicita a seleção do usuário
        while True:
            try:
                selected_index = int(input("Selecione um trimestre (número): "))
                if 1 <= selected_index <= len(self.data_processor.season_list):
                    selected_season = self.data_processor.season_list[selected_index - 1]
                    break
                else:
                    print("Número inválido. Tente novamente.")
            except ValueError:
                print("Entrada inválida. Digite um número.")

        print(f"Estação selecionada: {selected_season}")

        # Exibe combinações únicas de movs e kmeans disponíveis
        movs_col = f"{selected_season}_movs"
        kmeans_col = f"{selected_season}_k_means"
        movs_kmeans_df = (
            df[[movs_col, kmeans_col]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if movs_kmeans_df.empty:
            print(f"\nNenhuma combinação encontrada para {selected_season}. Abortando.")
            return

        print(f"\nCombinações disponíveis de {movs_col} e {kmeans_col}:")
        for _, row in movs_kmeans_df.iterrows():
            print(f"Cluster {row[kmeans_col]}: {row[movs_col]}")

        # Entrada flexível: 1 ou mais clusters separados por vírgula
        selected_input = input("\nDigite o(s) número(s) do(s) cluster(s) que deseja usar (ex: 0 ou 0,7,14): ")
        try:
            selected_clusters = [int(x.strip()) for x in selected_input.split(',') if x.strip().isdigit()]
            if not selected_clusters:
                print("Nenhum cluster válido informado.")
                return

            # Filtra as combinações válidas
            selected_combinations = movs_kmeans_df[movs_kmeans_df[kmeans_col].isin(selected_clusters)]
            if selected_combinations.empty:
                print("Nenhuma combinação encontrada com os clusters informados.")
                return

        except ValueError:
            print("Entrada inválida. Certifique-se de digitar apenas números separados por vírgula.")
            return

        # Cria lista de pares únicos (movs, kmeans) selecionados
        selected_pairs = list(
            selected_combinations[[movs_col, kmeans_col]].itertuples(index=False, name=None)
        )

        print("\n✅ Processando os seguintes pares:")
        for m, k in selected_pairs:
            print(f"- Cluster {k} / MoVs: {m}")

        for ano_alvo in tqdm(anos_disponiveis, desc=f"{selected_season} - múltiplos pares"):
            self._process_year_season(df, ano_alvo, selected_season, selected_pairs)

    def _process_year_season(self, df, ano_alvo, season, selected_pairs):
        logging.info(f"Iniciando {season}/{ano_alvo}")
        df_filtered = self.data_processor.filter_data_by_season_year(
            df, season, ano_alvo, selected_pairs
        )

        if df_filtered is None or df_filtered.empty:
            return

        grouped_data = self.data_processor.group_data_by_coordinates(df_filtered)
        grouped_data = filter_sc_pr(grouped_data)  # Filtrando dados SANTACATARINA PARANA
        logging.info(f"Grupos com dados para {season}/{ano_alvo}: {len(grouped_data)}")

        resultados_quadrimestre = self._process_locations_parallel(
            grouped_data, ano_alvo, season
        )

        self.result_processor.save_results(resultados_quadrimestre, season, ano_alvo)

    def _process_locations_parallel(self, grouped_data, ano_alvo, season):
        """Processa localizaçFiões em paralelo usando multiprocessing"""
        args_list = [(self.location_processor, group, ano_alvo, season) for _, group in grouped_data]
        results = process_map(
            process_location_wrapper,
            args_list,
            max_workers=os.cpu_count(),
            desc=f"{season}/{ano_alvo}",
            leave=True,
        )

        resultados_quadrimestre = []
        for res in results:
            if res:
                resultados_quadrimestre.extend(res)

        return resultados_quadrimestre


def main():
    """Função principal"""
    predictor = PrecipitationPredictor()
    predictor.run_predictions()


if __name__ == '__main__':
    main()

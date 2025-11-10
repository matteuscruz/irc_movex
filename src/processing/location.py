import pandas as pd
from src.models import FeatureParser, Trainer
from src.processing.features import series_to_supervised


class LocationProcessor:
    """Classe responsável pelo processamento de uma localização específica"""

    def __init__(self):
        self.feature_parser = FeatureParser()

    def process_location(self, group, ano_alvo, quadrimestre_alvo, model_type='lstm'):
        data = self.prepare_location_data(group, ano_alvo, quadrimestre_alvo)
        if data is None:
            return None
        return self.run_location_predictions(data, ano_alvo, quadrimestre_alvo, model_type=model_type)

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

    def run_location_predictions(self, data, ano_alvo, quadrimestre_alvo,
                                 epochs=200, batch_size=64, lr=0.001, model_type="lstm"):
        from src.utils.results import format_prediction_result
        import numpy as np

        results = []

        for feature_set in data["feature_combinations"]:
            feature_set = list(feature_set)
            try:
                # 🔹 Extrai dados
                X_train = data["df_treinamento"][feature_set].values.astype(np.float32)
                y_train = data["df_treinamento"]['pr'].values.astype(np.float32)
                X_test = data["df_teste"][feature_set].values.astype(np.float32)
                y_test = data["df_teste"]['pr'].values.astype(np.float32)

                # 🔹 Instancia trainer (usa GPU se disponível)
                self.model_trainer = Trainer(
                    model_type=model_type,
                    input_size=X_train.shape[1],
                    device='cpu'  # auto escolhe cuda/cpu
                )

                # 🔹 Ajusta para 3D (samples, timesteps, features)
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                # 🔹 Treina e prediz
                y_pred_test = self.model_trainer.train_and_predict(
                    X_train, y_train, X_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr
                )

                # 🔹 Formata resultados
                for i, (true, pred) in enumerate(zip(y_test, y_pred_test)):
                    result = format_prediction_result(
                        data["lat"], data["lon"], float(true), float(pred),
                        ano_alvo, quadrimestre_alvo, feature_set,
                        data["df_teste"][feature_set], i, data["cluster"]
                    )
                    results.append(result)

            except Exception as e:
                print(f"Ocorreu um erro no treinamento com features {feature_set}: {e}")
                continue

        return results



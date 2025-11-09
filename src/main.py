from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
from src.data import DataProcessor, DataLoader
from src.processing import LocationProcessor
from src.utils import setup_logging, save_results, filter_by_shapefile
from src.utils.geographic import select_regions_from_shapefile


def process_location_wrapper(args):
    location_processor, group, ano_alvo, season, model_type = args
    return location_processor.process_location(group, ano_alvo, season, model_type)


class PrecipitationPredictor:
    """Classe principal que coordena todo o processo de predição"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.data_loader = DataLoader()
        self.location_processor = LocationProcessor()
        setup_logging()

    def run_predictions(self):
        df = self.data_loader.load_and_prepare_data()
        anos_disponiveis = self.data_processor.get_available_years(df)

        # Interface de seleção (mantido do código original)
        print("Estações disponíveis:")
        for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON'], start=1):
            print(f"{i}. {season}")

        while True:
            try:
                selected_index = int(input("Selecione um trimestre (número): "))
                if 1 <= selected_index <= 4:
                    selected_season = ['DJF', 'MAM', 'JJA', 'SON'][selected_index - 1]
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
        # Pergunta qual modelo usar (similar à escolha de estação/clusters)
        models = ['lstm', 'gru', 'rnn']
        print("\nModelos disponíveis:")
        for i, md in enumerate(models, start=1):
            print(f"{i}. {md}")

        while True:
            model_input = input("Selecione o modelo (número ou nome): ").strip()
            # aceita número ou nome
            if model_input.isdigit():
                idx = int(model_input)
                if 1 <= idx <= len(models):
                    selected_model = models[idx - 1]
                    break
            else:
                if model_input.lower() in models:
                    selected_model = model_input.lower()
                    break
            print("Entrada inválida. Digite o número ou o nome do modelo (lstm/gru/rnn).")

        select_regions = select_regions_from_shapefile('SNIRH_RHI/SNIRH_RHI/SNIRH_RegioesHidrograficas.shp')
        for ano_alvo in tqdm(anos_disponiveis, desc=f"{selected_season} - múltiplos pares"):
                self._process_year_season(df, ano_alvo, selected_season, selected_pairs, select_regions, selected_model)

    def _process_year_season(self, df, ano_alvo, season, selected_pairs, region, model_type='lstm'):
        import logging
        logging.info(f"Iniciando {season}/{ano_alvo}")

        df_filtered = self.data_processor.filter_data_by_season_year(
            df, season, ano_alvo, selected_pairs
        )

        if df_filtered is None or df_filtered.empty:
            return

        grouped_data = self.data_processor.group_data_by_coordinates(df_filtered)
        grouped_data = filter_by_shapefile(grouped_data, 'SNIRH_RHI/SNIRH_RHI/SNIRH_RegioesHidrograficas.shp',region)
        logging.info(f"Grupos com dados para {season}/{ano_alvo}: {len(grouped_data)}")

        resultados_quadrimestre = self._process_locations_parallel(
            grouped_data, ano_alvo, season, model_type
        )

        save_results(resultados_quadrimestre, season, ano_alvo)

    def _process_locations_parallel(self, grouped_data, ano_alvo, season, model_type='lstm'):
        """Processa localizações em paralelo usando multiprocessing"""
        args_list = [(self.location_processor, group, ano_alvo, season, model_type) for _, group in grouped_data]
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
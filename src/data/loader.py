import pandas as pd
import os
import logging
from src.config.settings import DTYPES, QUARTER_MAP

class DataLoader:
    """Classe responsável pelo carregamento de dados"""

    def load_and_prepare_data(self, processed_path='data/MWC_v3_processed.parquet'):
        """Carrega e prepara os dados básicos, com cache no arquivo processado"""
        
        if os.path.exists(processed_path):
            df = pd.read_parquet(processed_path).astype(DTYPES)
            return df
        
        raw_path = input("Insira o dataset path: ")
        logging.info("Arquivo processado não encontrado. Carregando bruto e processando...")

        df = pd.read_parquet(raw_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['month'].map(QUARTER_MAP)
        df['meteorological_year'] = df['year']
        df = df.astype(DTYPES)
        
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logging.info(f"Arquivo processado salvo em: {processed_path}")

        return df
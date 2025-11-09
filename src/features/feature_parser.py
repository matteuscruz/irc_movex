"""
Módulo para parsing e processamento de features.
"""

import re
from typing import List, Tuple
from itertools import combinations


class FeatureParser:
    """Classe responsável pelo parsing e processamento de features."""

    @staticmethod
    def parse_movs(movs_str: str) -> List[str]:
        """
        Parse the MoVs string into individual feature names.

        Args:
            movs_str: String contendo features separadas por vírgula ou '+'

        Returns:
            Lista de nomes de features únicos
        """
        if not isinstance(movs_str, str) or not movs_str.strip():
            return []

        # Divide por vírgula ou sinal de mais
        features = [f.strip() for f in re.split(r',|\+', movs_str)]

        # Remove strings vazias e duplicatas, mantém ordem
        unique_features = []
        seen = set()
        for feature in features:
            if feature and feature not in seen:
                unique_features.append(feature)
                seen.add(feature)

        return unique_features

    @staticmethod
    def generate_feature_combinations(feature_cols: List[str]) -> List[Tuple[str, ...]]:
        """
        Gera todas as combinações possíveis de features.

        Args:
            feature_cols: Lista de colunas de features

        Returns:
            Lista de todas as combinações possíveis de features
        """
        if not feature_cols:
            return []

        feature_cols_combs = []
        for r in range(1, len(feature_cols) + 1):
            feature_cols_combs.extend(combinations(feature_cols, r))

        return feature_cols_combs

    @staticmethod
    def validate_features(features: List[str], available_columns: List[str]) -> Tuple[List[str], List[str]]:
        """
        Valida se as features estão disponíveis nas colunas do DataFrame.

        Args:
            features: Lista de features para validar
            available_columns: Lista de colunas disponíveis

        Returns:
            Tupla com (features_válidas, features_inválidas)
        """
        valid_features = []
        invalid_features = []

        for feature in features:
            if feature in available_columns:
                valid_features.append(feature)
            else:
                invalid_features.append(feature)

        return valid_features, invalid_features

    @staticmethod
    def get_feature_importance_ranking(feature_importance_dict: dict,
                                       feature_combinations: List[Tuple]) -> List[Tuple]:
        """
        Ranqueia combinações de features por importância.

        Args:
            feature_importance_dict: Dicionário com importância de cada feature
            feature_combinations: Lista de combinações de features

        Returns:
            Lista de combinações ordenadas por importância total (descendente)
        """
        ranked_combinations = []

        for combo in feature_combinations:
            total_importance = sum(
                feature_importance_dict.get(feature, 0) for feature in combo
            )
            ranked_combinations.append((combo, total_importance))

        # Ordena por importância total (descendente)
        ranked_combinations.sort(key=lambda x: x[1], reverse=True)

        return [combo for combo, _ in ranked_combinations]

    @staticmethod
    def filter_combinations_by_size(feature_combinations: List[Tuple],
                                    min_size: int = 1,
                                    max_size: int = None) -> List[Tuple]:
        """
        Filtra combinações por tamanho mínimo e máximo.

        Args:
            feature_combinations: Lista de combinações de features
            min_size: Tamanho mínimo das combinações
            max_size: Tamanho máximo das combinações (None = sem limite)

        Returns:
            Lista filtrada de combinações
        """
        filtered_combinations = []

        for combo in feature_combinations:
            combo_size = len(combo)
            if combo_size >= min_size:
                if max_size is None or combo_size <= max_size:
                    filtered_combinations.append(combo)

        return filtered_combinations

    @classmethod
    def process_movs_column(cls, movs_values: List[str],
                            available_columns: List[str]) -> dict:
        """
        Processa uma coluna inteira de valores MoVs.

        Args:
            movs_values: Lista de strings MoVs
            available_columns: Colunas disponíveis no DataFrame

        Returns:
            Dicionário com estatísticas do processamento
        """
        all_features = set()
        valid_movs = 0
        invalid_features_count = 0

        for movs_str in movs_values:
            if pd.isna(movs_str):
                continue

            features = cls.parse_movs(movs_str)
            if features:
                valid_movs += 1
                valid_features, invalid_features = cls.validate_features(
                    features, available_columns
                )
                all_features.update(valid_features)
                invalid_features_count += len(invalid_features)

        return {
            'total_movs_entries': len([m for m in movs_values if not pd.isna(m)]),
            'valid_movs_entries': valid_movs,
            'unique_features_found': len(all_features),
            'all_features': sorted(list(all_features)),
            'invalid_features_count': invalid_features_count
        }


# Importa pandas apenas se necessário para a função process_movs_column
try:
    import pandas as pd
except ImportError:
    pass
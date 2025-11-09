import re
from itertools import combinations

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
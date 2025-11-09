# Data types configuration
DTYPES = {
    'date': 'int64',
    'time': 'int64',
    'Date': 'int64',
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
    'year': 'int16',
    'month': 'int8',
    'meteorological_year': 'int16',
    'DJF_k_means': 'int8',
    'JJA_k_means': 'int8',
    'MAM_k_means': 'int8',
    'SON_k_means': 'int8',
    'DJF_movs': 'category',
    'JJA_movs': 'category',
    'MAM_movs': 'category',
    'SON_movs': 'category',
    'quarter': 'category'
}

# Season mapping
QUARTER_MAP = {
    12: 'DJF', 1: 'DJF', 2: 'DJF',
    3: 'MAM', 4: 'MAM', 5: 'MAM',
    6: 'JJA', 7: 'JJA', 8: 'JJA',
    9: 'SON', 10: 'SON', 11: 'SON'
}

SEASON_LIST = ['DJF', 'MAM', 'JJA', 'SON']
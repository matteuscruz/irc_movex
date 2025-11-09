import pandas as pd

def series_to_supervised(df, target_col, n_in=1, n_out=1, dropnan=True):
    """
    Converte um DataFrame de série temporal em formato supervisionado.
    """
    cols = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i).add_suffix(f"_t-{i}"))
    for i in range(0, n_out):
        cols.append(
            df[[target_col]].shift(-i).rename(columns={target_col: f"{target_col}_t+{i}"})
        )

    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg
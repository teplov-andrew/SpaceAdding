import pandas as pd

def filter_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df_filtered = df.drop_duplicates(subset=[column])
    df_filtered = df_filtered[df_filtered[column].astype(str).str.split().str.len() > 1]
    return df_filtered.reset_index(drop=True)

def replace_space(el: str) -> str:
    return str(el).replace(" ", "")

def ensure_no_space_column(df: pd.DataFrame, col_with_space: str, col_no_space: str) -> pd.DataFrame:
    if col_no_space not in df.columns:
        df[col_no_space] = df[col_with_space].astype(str).apply(replace_space)
    return df
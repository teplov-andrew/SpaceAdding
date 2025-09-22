import re
import pandas as pd
from typing import Optional
from .preprocessing import filter_text_column, ensure_no_space_column

def load_from_txt(path: str) -> pd.DataFrame:
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = re.split(r',', line.strip(), maxsplit=1)
            if len(parts) == 2:
                id_val, text = parts
                data.append({'id': id_val, 'text': text})
    df = pd.DataFrame(data)
    if len(df) and df.iloc[0, 0] != df.columns[0]:
        new_header = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df.columns = new_header
    return df

def load_train_dataframe(
    train_csv: Optional[str] = None,
    title_col: str = "base_title",
    use_txt_file: bool = False,
    raw_txt_path: Optional[str] = None,
) -> pd.DataFrame:
    if use_txt_file:
        if not raw_txt_path:
            raise ValueError("raw_txt_path must be provided when use_txt_file=True")
        df = load_from_txt(raw_txt_path)
        if title_col not in df.columns:
            if "text" in df.columns:
                df = df.rename(columns={"text": title_col})
            else:
                raise ValueError(f"Column {title_col} not found in TXT file.")
    else:
        if not train_csv:
            raise ValueError("train_csv must be provided when use_txt_file=False")
        df_texts = pd.read_csv(train_csv)
        if title_col not in df_texts.columns:
            raise ValueError(f"Column {title_col} not found in {train_csv}.")
        df = df_texts[[title_col]].copy()

    df = filter_text_column(df, title_col)
    df = ensure_no_space_column(df, title_col, f"{title_col}_no_space")
    return df

def load_test_dataframe(
    use_test_txt: bool,
    test_txt_path: Optional[str],
    test_csv: Optional[str],
    title_col: str,
    test_texts_col: Optional[str] = None,
) -> pd.DataFrame:
    if use_test_txt:
        if not test_txt_path:
            raise ValueError("test_txt_path must be provided when use_test_txt=True")
        df_raw = load_from_txt(test_txt_path)
        if "text" not in df_raw.columns:
            text_col = df_raw.columns[1] if len(df_raw.columns) > 1 else None
            if not text_col:
                raise ValueError("TXT must contain a text column.")
            df_raw = df_raw.rename(columns={text_col: "text"})
        df = pd.DataFrame()
        df["id"] = df_raw["id"] if "id" in df_raw.columns else range(len(df_raw))
        df["input_text"] = df_raw["text"].astype(str)
        if title_col in df_raw.columns:
            df["target_text"] = df_raw[title_col].astype(str)
        return df.reset_index(drop=True)
    else:
        if not test_csv:
            raise ValueError("Either use_test_txt=True or provide test_csv")
        df_raw = pd.read_csv(test_csv)
        df = pd.DataFrame()
        df["id"] = df_raw[title_col] if "id" not in df_raw.columns else df_raw["id"]
        if test_texts_col and test_texts_col in df_raw.columns:
            df["input_text"] = df_raw[test_texts_col].astype(str)
        else:
            if title_col not in df_raw.columns:
                raise ValueError(f"Column {title_col} not found in test CSV.")
            df["input_text"] = df_raw[title_col].astype(str).str.replace(" ", "", regex=False)
            df["target_text"] = df_raw[title_col].astype(str)
        if title_col in df_raw.columns and "target_text" not in df.columns:
            df["target_text"] = df_raw[title_col].astype(str)
        return df.reset_index(drop=True)

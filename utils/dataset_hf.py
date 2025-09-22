from datasets import Dataset
from typing import Dict, Any

def build_hf_splits(df, title_col: str, max_length: int, tokenizer, test_size: float = 0.2, seed: int = 42):
    dataset = Dataset.from_pandas(df)
    train_test = dataset.train_test_split(test_size=test_size, seed=seed)

    def preprocess(examples: Dict[str, Any]):
        src_col = f"{title_col}_no_space"
        inputs = tokenizer(examples[src_col], max_length=max_length, truncation=True)
        labels = tokenizer(examples[title_col], max_length=max_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = train_test.map(preprocess, batched=True)
    return tokenized
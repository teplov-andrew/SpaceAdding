import os
import hydra
import torch
import pandas as pd
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loading import load_test_dataframe
from utils.metrics import get_space_positions

def predict_space_positions(model, tokenizer, texts, device=None, max_length=128, num_beams=5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions, positions = [], []
    for text in tqdm(texts):
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        positions.append([i for i, ch in enumerate(pred) if ch == " "])
    return predictions, positions

@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    # модель (если обучали — берём outputs/final)
    # model_dir = os.path.join(cfg.paths.output_dir, "final")
    # model_name_or_path = model_dir if os.path.isdir(model_dir) else cfg.model.pretrained_name
    model_path = cfg.generate.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    # загрузка теста (TXT по умолчанию)
    df = load_test_dataframe(
        use_test_txt=cfg.generate.use_test_txt,
        test_txt_path=cfg.generate.test_txt_path,
        test_csv=cfg.generate.test_csv,
        title_col=cfg.data.columns.title_col,
        test_texts_col=cfg.generate.test_texts_col,
    )

    # предсказания
    texts = df["input_text"].tolist()
    preds, pos = predict_space_positions(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=cfg.generate.max_length,
        num_beams=cfg.generate.num_beams
    )

    # сохранение
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    out_path = cfg.generate.output_csv
    pd.DataFrame({
        "id": df["id"],
        "predicted_text": preds,
        "predicted_positions": pos
    }).to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()

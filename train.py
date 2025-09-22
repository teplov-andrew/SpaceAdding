import os
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.modeling import load_model_and_tokenizer
from utils.data_loading import load_train_dataframe
from utils.dataset_hf import build_hf_splits
from utils.metrics import compute_space_metrics_factory

@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1) данные
    df = load_train_dataframe(
        train_csv=cfg.data.train_csv,
        title_col=cfg.data.columns.title_col,
        use_txt_file=cfg.data.use_txt_file,
        # raw_txt_path=cfg.data.raw_txt_path,
    )

    # 2) модель/токенайзер
    model, tokenizer = load_model_and_tokenizer(cfg.model.pretrained_name)

    # 3) датасеты
    tokenized = build_hf_splits(
        df=df,
        title_col=cfg.data.columns.title_col,
        max_length=cfg.data.max_length,
        tokenizer=tokenizer,
        test_size=cfg.data.split.test_size,
        seed=cfg.data.split.seed,
    )

    # 4) модели
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = compute_space_metrics_factory(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.paths.output_dir,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        logging_steps=cfg.train.logging_steps,
        save_total_limit=cfg.train.save_total_limit,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        fp16=cfg.train.fp16,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        eval_strategy=cfg.train.evaluation_strategy,
        save_strategy=cfg.train.save_strategy,
        report_to=cfg.train.report_to,
        predict_with_generate=False,
        push_to_hub=False,
        seed=cfg.train.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    # 5) обучение
    trainer.train()

    # 6) сохранение
    final_dir = os.path.join(cfg.paths.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved model to: {final_dir}")

if __name__ == "__main__":
    main()

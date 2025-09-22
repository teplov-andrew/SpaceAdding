import numpy as np
from typing import Dict, Tuple

def get_space_positions(text: str) -> set:
    return {i for i, ch in enumerate(text) if ch == " "}

def compute_space_metrics_factory(tokenizer):
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_prec, all_rec, all_f1 = [], [], []
        for pred, true in zip(decoded_preds, decoded_labels):
            pred_spaces = get_space_positions(pred)
            true_spaces = get_space_positions(true)

            tp = len(pred_spaces & true_spaces)
            fp = len(pred_spaces - true_spaces)
            fn = len(true_spaces - pred_spaces)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1)

        return {
            "precision": float(np.mean(all_prec)) if all_prec else 0.0,
            "recall": float(np.mean(all_rec)) if all_rec else 0.0,
            "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
        }
    return compute_metrics

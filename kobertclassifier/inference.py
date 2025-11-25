# inference.py
import pandas as pd
import torch


def show_misclassified(model, loader, device, label_map=None, output_path="misclassified_results.csv"):
    model.eval()
    all_preds, all_trues, all_texts = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].cpu().tolist()
            texts = batch["text"]

            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = outputs.argmax(1).cpu().tolist()

            all_preds.extend(preds)
            all_trues.extend(labels)
            all_texts.extend(texts)

    # ✅ 오분류만 추출
    misclassified = []
    for text, pred, true in zip(all_texts, all_preds, all_trues):
        if pred != true:
            misclassified.append({
                "text": text,
                "pred_label": label_map[pred] if label_map else pred,
                "true_label": label_map[true] if label_map else true
            })

    # ✅ CSV 저장
    df = pd.DataFrame(misclassified)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"💾 Misclassified samples saved to '{output_path}' ({len(df)} rows).")

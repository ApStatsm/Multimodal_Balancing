import torch
import torch.nn as nn
import torch.nn.functional as F

def run_epoch(model, loader, optimizer, device, mode="train"):
    if mode == "train":
        model.train()
    else:
        model.eval()

    total, correct = 0, 0
    total_loss = 0
    ce = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(mode == "train"):
        for text_input, bio_input, label in loader:
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            if mode == "train":
                optimizer.zero_grad()

            logits = model(text_input, bio_input)
            loss = ce(logits, label)

            if mode == "train":
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return acc, avg_loss

def test_multimodal(model, loader, device, shuffle_mode="none"):
    """
    shuffle_mode: "none", "text", "bio"
    Returns: acc, loss, labels, preds, probs (for AUC)
    """
    model.eval()
    total_loss = 0
    ce = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for text_input, bio_input, label in loader:
            
            for k in text_input:
                text_input[k] = text_input[k].to(device)
            bio_input = bio_input.to(device)
            label = label.to(device)

            # 🎲 셔플 로직
            idx = torch.randperm(bio_input.size(0)).to(device)
            if shuffle_mode == "text":
                for k in text_input:
                    text_input[k] = text_input[k][idx]
            elif shuffle_mode == "bio":
                bio_input = bio_input[idx]

            logits = model(text_input, bio_input)
            loss = ce(logits, label)
            total_loss += loss.item()

            # AUC 계산을 위한 확률값 (Class 1에 대한 확률)
            probs = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    acc = correct / len(all_labels)
    avg_loss = total_loss / len(loader)

    return acc, avg_loss, all_labels, all_preds, all_probs
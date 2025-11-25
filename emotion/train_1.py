import torch
import torch.nn as nn
from tqdm import tqdm
import time

def run_epoch(model, loader, optimizer, device, mode="train"):

    if mode == "train":
        model.train()
    else:
        model.eval()

    total, correct = 0, 0
    total_loss = 0
    ce = nn.CrossEntropyLoss()

    start = time.time()

    with torch.set_grad_enabled(mode == "train"):
        for text_input, bio_input, label in tqdm(loader, desc=mode.upper()):

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
    return acc, total_loss / len(loader), time.time() - start

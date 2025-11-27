# train.py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, trues = 0, [], []
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds += outputs.argmax(1).detach().cpu().tolist()
        trues += labels.cpu().tolist()
    acc = accuracy_score(trues, preds)
    return total_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, trues = 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds += outputs.argmax(1).cpu().tolist()
            trues += labels.cpu().tolist()
    acc = accuracy_score(trues, preds)
    return total_loss / len(loader), acc, preds, trues

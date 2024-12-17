import json
from pathlib import Path

METRICS_FILE = Path("metrics.json")

def save_metrics(metrics):
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)

def load_metrics():
    if not METRICS_FILE.exists():
        return {}
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)

def update_metric(key, value):
    metrics = load_metrics()
    metrics[key] = value
    save_metrics(metrics)

def get_metric(key):
    metrics = load_metrics()
    return metrics.get(key)

# 具体的更新函数
def update_current_epoch(epoch): update_metric('current_epoch', epoch)
def update_train_loss(loss): update_metric('train_loss', round(loss, 4))
def update_train_accuracy(acc): update_metric('train_accuracy', round(acc, 2))
def update_val_accuracy(acc): update_metric('val_accuracy', round(acc, 2)) 
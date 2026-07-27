import torch
import hashlib
import json
import datetime
import os


# ── 12.2.1 Versioned Checkpoint ──────────────────────────────────────────────

def save_versioned_checkpoint(model, optimizer, metrics: dict, config: dict,
                               save_dir: str = "checkpoints") -> str:
    """
    Simpan checkpoint dengan metadata lengkap + registry.
    Registry memungkinkan penelusuran kembali eksperimen yang menghasilkan angka tertentu.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "timestamp": timestamp,
    }

    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    filename = f"{save_dir}/model_{timestamp}_{config_hash}.pt"

    torch.save(state, filename)

    # Update registry
    registry_path = f"{save_dir}/registry.json"
    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except FileNotFoundError:
        registry = []

    registry.append({
        "filename": filename,
        "timestamp": timestamp,
        "config_hash": config_hash,
        "metrics": metrics
    })

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"✅ Checkpoint saved: {filename}")
    return filename


def load_best_checkpoint(save_dir: str = "checkpoints", metric: str = "val_loss",
                          minimize: bool = True):
    """Load checkpoint terbaik berdasarkan metrik dari registry."""
    registry_path = f"{save_dir}/registry.json"
    with open(registry_path, "r") as f:
        registry = json.load(f)

    valid = [r for r in registry if metric in r.get("metrics", {})]
    if not valid:
        raise ValueError(f"Tidak ada checkpoint dengan metrik '{metric}'")

    best = min(valid, key=lambda r: r["metrics"][metric]) if minimize \
        else max(valid, key=lambda r: r["metrics"][metric])

    print(f"✅ Best checkpoint: {best['filename']} ({metric}={best['metrics'][metric]})")
    return torch.load(best["filename"], map_location="cpu"), best
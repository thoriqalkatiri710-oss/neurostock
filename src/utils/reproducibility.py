import os
import random

import numpy as np
import torch

# ── 17.1.1 Global Seed ────────────────────────────────────────────────────────

def set_global_seed(seed: int = 42):
    """
    Set seed di SEMUA sumber randomness.
    Panggil di awal SETIAP script entry point — bukan hanya sekali.
    Catatan: deterministic=True bisa memperlambat training di GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Global seed set: {seed}")


def seed_worker(worker_id: int):
    """
    Dipanggil di DataLoader worker_init_fn agar tiap worker ter-seed konsisten.
    Penggunaan: DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── 17.1.2 Multi-Seed Evaluation ──────────────────────────────────────────────

def run_multi_seed_evaluation(train_fn, eval_fn,
                               seeds: list = [42, 123, 7, 2026, 99]) -> tuple:
    """
    Jalankan eksperimen dengan beberapa seed berbeda.
    Laporkan mean ± std — bukan hasil satu run terbaik saja.
    Multi-agent RL punya variance antar run meski seed sama (CUDA non-deterministic).
    """
    results = []
    for seed in seeds:
        print(f"\n── Seed {seed} ──")
        set_global_seed(seed)
        model = train_fn(seed=seed)
        metrics = eval_fn(model)
        results.append(metrics)
        print(f"  Metrics: {metrics}")

    # Agregasi mean ± std
    aggregated = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        aggregated[key] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4)
        }

    print("\n── Multi-Seed Summary ──")
    for k, v in aggregated.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    return aggregated, results


# ── 17.2.1 Environment Info ───────────────────────────────────────────────────

def print_environment_info():
    """Print informasi environment untuk dokumentasi reproducibility."""
    import platform
    import sys

    print("── Environment Info ──")
    print(f"Python   : {sys.version}")
    print(f"Platform : {platform.platform()}")
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA ver : {torch.version.cuda}")
        print(f"GPU      : {torch.cuda.get_device_name(0)}")


def save_environment_info(path: str = "docs/environment_info.md"):
    """Simpan environment info ke markdown file."""
    import platform
    import sys

    lines = [
        "# Environment Info\n",
        f"- Python: `{sys.version}`\n",
        f"- Platform: `{platform.platform()}`\n",
        f"- PyTorch: `{torch.__version__}`\n",
        f"- CUDA available: `{torch.cuda.is_available()}`\n",
    ]
    if torch.cuda.is_available():
        lines.append(f"- CUDA version: `{torch.version.cuda}`\n")
        lines.append(f"- GPU: `{torch.cuda.get_device_name(0)}`\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"✅ Environment info saved: {path}")


# ── 17.3.1 Reproducibility Checklist ─────────────────────────────────────────

def print_reproducibility_checklist():
    checklist = [
        "Seed di-set eksplisit dan didokumentasikan untuk setiap eksperimen",
        "requirements_lock.txt ter-generate dan tersimpan di repo",
        "Hasil dilaporkan sebagai mean ± std dari minimal 3 run berbeda",
        "Split data (walk-forward fold) konsisten di seluruh eksperimen",
        "Checkpoint model final tersimpan dan bisa di-load ulang",
        "Skrip end-to-end bisa dijalankan ulang tanpa langkah manual tersembunyi",
    ]
    print("\n── Reproducibility Checklist ──")
    for item in checklist:
        print(f"  [ ] {item}")


if __name__ == "__main__":
    # Test seed setting
    set_global_seed(42)

    # Verifikasi reproducibility
    print("\n── Verifikasi Reproducibility ──")
    set_global_seed(42)
    a = torch.randn(3)
    set_global_seed(42)
    b = torch.randn(3)
    print(f"Run 1: {a.tolist()}")
    print(f"Run 2: {b.tolist()}")
    print(f"Identical: {torch.allclose(a, b)} ✅" if torch.allclose(a, b)
          else "❌ Not identical — check seed setup")

    # Environment info
    print()
    print_environment_info()
    save_environment_info()

    # Checklist
    print_reproducibility_checklist()

    # Demo multi-seed
    print("\n── Demo Multi-Seed Evaluation ──")
    def mock_train(seed): return {"seed": seed}
    def mock_eval(model):
        return {
            "mape": np.random.uniform(10, 15),
            "service_level": np.random.uniform(0.88, 0.96)
        }

    aggregated, _ = run_multi_seed_evaluation(
        mock_train, mock_eval, seeds=[42, 123, 7]
    )
"""
BAGIAN 18 — ETIKA DAN TANGGUNG JAWAB SISTEM AI
Audit fairness, explainability, dan human-in-the-loop.
Bagian yang membedakan project portofolio serius dari yang asal jalan.
"""

import pandas as pd
import torch

# ── 18.1.1 Fairness Audit ─────────────────────────────────────────────────────

def audit_service_level_fairness(simulation_results: dict,
                                  store_metadata: dict) -> dict:
    """
    Audit disparitas service level antar kelompok toko.
    simulation_results: {store_id: service_level}
    store_metadata: {store_id: {"region": str, "size": str, ...}}

    Jika disparitas >10 poin persentase → flag untuk ditinjau.
    Bukan berarti sistem dihentikan, tapi didokumentasikan sebagai
    keterbatasan yang harus dipantau.
    """
    rows = []
    for sid, sl in simulation_results.items():
        row = {"store_id": sid, "service_level": sl}
        row.update(store_metadata.get(sid, {}))
        rows.append(row)

    df = pd.DataFrame(rows)

    if "region" not in df.columns:
        df["region"] = "unknown"

    group_stats = df.groupby("region")["service_level"].agg(["mean", "std", "min"])
    disparity = group_stats["mean"].max() - group_stats["mean"].min()

    result = {
        "group_stats": group_stats.round(4).to_dict(),
        "max_disparity": round(float(disparity), 4),
        "flag_for_review": bool(disparity > 0.10),
    }

    print("\n── 18.1 Service Level Fairness Audit ──")
    print(group_stats.round(4).to_string())
    print(f"\nMax disparity: {disparity:.4f}")
    if result["flag_for_review"]:
        print("⚠️  Disparitas >10% — pertimbangkan fairness component di reward function")
    else:
        print("✅ Disparitas dalam batas wajar (<10%)")

    return result


# ── 18.2.1 Attention Explainability ──────────────────────────────────────────

def visualize_attention_explanation(attn_weights: torch.Tensor,
                                     feature_names: list,
                                     timestep_labels: list,
                                     top_k: int = 5) -> list:
    """
    Ekstrak top-k timestep yang paling berkontribusi terhadap forecast.
    attn_weights shape: (n_heads, seq_len, seq_len) atau (n_heads, seq_len)

    Tampilkan di dashboard sebagai:
    "Forecast naik terutama karena pola penjualan akhir pekan 2 minggu lalu
    dan sinyal suhu hari ini" — jauh lebih dipercaya pengguna bisnis.
    """
    if attn_weights.dim() == 3:
        # Ambil attention dari posisi terakhir (prediksi hari ini)
        avg_attn = attn_weights.mean(dim=0)[-1]
    else:
        avg_attn = attn_weights.mean(dim=0)

    # Pastikan top_k tidak melebihi panjang sequence
    k = min(top_k, len(avg_attn))
    top_k_idx = avg_attn.topk(k).indices.tolist()

    explanation = []
    for i in top_k_idx:
        label = timestep_labels[i] if i < len(timestep_labels) else f"t-{len(avg_attn)-i}"
        explanation.append({
            "timestep": label,
            "contribution": round(float(avg_attn[i]), 4),
            "contribution_pct": round(float(avg_attn[i] / avg_attn.sum()) * 100, 1)
        })

    explanation = sorted(explanation, key=lambda x: -x["contribution"])

    print("\n── 18.2 Attention Explanation (Top contributors) ──")
    for i, e in enumerate(explanation):
        print(f"  {i+1}. {e['timestep']:<20} contribution={e['contribution']:.4f} "
              f"({e['contribution_pct']}%)")

    return explanation


# ── 18.3.1 Human-in-the-Loop ─────────────────────────────────────────────────

def flag_for_human_review(recommendation: dict,
                           threshold_qty: float = 500,
                           threshold_uncertainty: float = 0.4) -> dict:
    """
    Flag keputusan yang butuh review manusia sebelum dieksekusi.
    Prinsip desain: sistem memberi REKOMENDASI, bukan eksekusi otomatis penuh.
    Terutama untuk: order bernilai besar, forecast dengan uncertainty tinggi.
    """
    high_value = recommendation["recommended_order_qty"] > threshold_qty

    mu = recommendation["forecast_mu"]
    sigma = recommendation["forecast_sigma"]
    mu_val = mu[0] if isinstance(mu, list) else float(mu)
    sigma_val = sigma[0] if isinstance(sigma, list) else float(sigma)
    uncertainty_ratio = sigma_val / (mu_val + 1e-6)
    high_uncertainty = uncertainty_ratio > threshold_uncertainty

    needs_review = bool(high_value or high_uncertainty)

    reasons = []
    if high_value:
        reasons.append(f"Order qty {recommendation['recommended_order_qty']:.0f} > threshold {threshold_qty}")
    if high_uncertainty:
        reasons.append(f"Uncertainty ratio {uncertainty_ratio:.2f} > threshold {threshold_uncertainty}")

    result = {
        "needs_human_review": needs_review,
        "reasons": reasons,
        "high_value_order": high_value,
        "high_uncertainty": high_uncertainty,
        "uncertainty_ratio": round(uncertainty_ratio, 4),
    }

    print("\n── 18.3 Human-in-the-Loop Check ──")
    if needs_review:
        print("⚠️  Review manusia diperlukan:")
        for r in reasons:
            print(f"   - {r}")
    else:
        print("✅ Keputusan dalam batas otomatis — tidak perlu review manual")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  BAGIAN 18 — AI ETHICS & RESPONSIBILITY")
    print("=" * 60)

    # 18.1 Fairness audit
    simulation_results = {
        "store_0": 0.94, "store_1": 0.91, "store_2": 0.88,
        "store_3": 0.93, "store_4": 0.76,  # toko di region C lebih rendah
    }
    store_metadata = {
        "store_0": {"region": "North", "size": "large"},
        "store_1": {"region": "North", "size": "medium"},
        "store_2": {"region": "South", "size": "medium"},
        "store_3": {"region": "South", "size": "large"},
        "store_4": {"region": "East",  "size": "small"},
    }
    audit_result = audit_service_level_fairness(simulation_results, store_metadata)

    # 18.2 Attention explanation
    attn_weights = torch.softmax(torch.randn(4, 14, 14), dim=-1)
    timestep_labels = [f"t-{13-i}" for i in range(14)]
    explanation = visualize_attention_explanation(
        attn_weights, feature_names=[], timestep_labels=timestep_labels
    )

    # 18.3 Human-in-the-loop
    rec_normal = {
        "recommended_order_qty": 250.0,
        "forecast_mu": [62.4],
        "forecast_sigma": [2.46],
    }
    rec_flagged = {
        "recommended_order_qty": 750.0,
        "forecast_mu": [62.4],
        "forecast_sigma": [35.0],  # uncertainty tinggi
    }

    print("\nSkenario normal:")
    flag_for_human_review(rec_normal)

    print("\nSkenario perlu review:")
    flag_for_human_review(rec_flagged)
"""
BAGIAN 14 — BUSINESS CASE & ANALISIS ROI
Menerjemahkan hasil teknis ke dampak finansial yang dipahami pengambil keputusan bisnis.
"""

import pandas as pd

# ── 14.1.1 Cost Savings Calculation ──────────────────────────────────────────

def calculate_cost_savings(baseline_metrics: dict, neurostock_metrics: dict,
                            n_stores: int,
                            annualization_factor: float = 365 / 90) -> dict:
    """
    Ekstrapolasi penghematan dari simulasi 90 hari ke estimasi tahunan.
    CATATAN: angka ini ESTIMASI berdasarkan simulasi, bukan deployment nyata.
    Selalu beri label 'estimasi' dan jelaskan asumsi di baliknya.
    """
    baseline_annual = baseline_metrics["total_cost"] * annualization_factor * n_stores
    neurostock_annual = neurostock_metrics["total_cost"] * annualization_factor * n_stores
    savings = baseline_annual - neurostock_annual
    savings_pct = (savings / baseline_annual) * 100 if baseline_annual > 0 else 0

    return {
        "baseline_annual_cost": round(baseline_annual, 2),
        "neurostock_annual_cost": round(neurostock_annual, 2),
        "estimated_annual_savings": round(savings, 2),
        "savings_percentage": round(savings_pct, 2),
    }


# ── 14.1.2 Sensitivity Analysis ───────────────────────────────────────────────

def cost_sensitivity_analysis(holding_cost_range: list,
                               stockout_penalty_range: list,
                               evaluate_fn) -> pd.DataFrame:
    """
    Uji robustness kesimpulan terhadap variasi asumsi biaya.
    Visualisasikan sebagai heatmap — lebih meyakinkan dari satu angka tunggal.
    """
    results = []
    for h_cost in holding_cost_range:
        for s_penalty in stockout_penalty_range:
            metrics = evaluate_fn(holding_cost=h_cost, stockout_penalty=s_penalty)
            results.append({
                "holding_cost": h_cost,
                "stockout_penalty": s_penalty,
                "neurostock_advantage_pct": metrics.get("savings_percentage", 0)
            })

    df = pd.DataFrame(results)
    pivot = df.pivot(index="holding_cost", columns="stockout_penalty",
                     values="neurostock_advantage_pct")

    print("\n── Sensitivity Analysis Heatmap (NeuroStock advantage %) ──")
    print("holding_cost \\ stockout_penalty")
    print(pivot.round(1).to_string())
    return df


# ── 14.2.1 NPV Calculation ────────────────────────────────────────────────────

def calculate_npv(initial_investment: float, annual_savings: float,
                  discount_rate: float = 0.10, years: int = 3) -> dict:
    """
    NPV = -Investasi_awal + Σ (Penghematan_tahunan / (1+r)^t) untuk t=1..years
    Sertakan biaya implementasi realistis — tidak hanya sisi penghematan.
    """
    npv = -initial_investment
    cash_flows = []
    for t in range(1, years + 1):
        discounted_cf = annual_savings / ((1 + discount_rate) ** t)
        npv += discounted_cf
        cash_flows.append(round(discounted_cf, 2))

    payback_period = initial_investment / annual_savings if annual_savings > 0 else float("inf")

    return {
        "npv": round(npv, 2),
        "discounted_cash_flows": cash_flows,
        "payback_period_years": round(payback_period, 2),
        "is_profitable": npv > 0,
    }


# ── 14.3.1 Metrics Translation Table ─────────────────────────────────────────

def print_business_translation():
    """
    Tabel translasi metrik teknis ke bahasa bisnis.
    Berguna untuk README dan presentasi ke audiens campuran teknis-non teknis.
    """
    translations = [
        ("MAPE turun dari 18% ke 12%",
         "Perkiraan permintaan lebih akurat, mengurangi kesalahan perencanaan stok"),
        ("Stockout rate turun 23%",
         "Lebih sedikit kehilangan penjualan akibat barang habis"),
        ("Service level naik ke 94%",
         "94 dari 100 permintaan pelanggan terpenuhi tanpa kehabisan stok"),
        ("Holding cost turun 11%",
         "Modal yang terikat di gudang berkurang, bisa dialokasikan ke hal lain"),
        ("Coverage probability 78% (target 80%)",
         "Estimasi ketidakpastian cukup terkalibrasi, bisa dipercaya untuk perencanaan"),
    ]

    print("\n── 14.3.1 Translasi Metrik Teknis ke Bahasa Bisnis ──")
    print(f"{'Metrik Teknis':<40} {'Arti Bisnis'}")
    print("-" * 100)
    for technical, business in translations:
        print(f"{technical:<40} {business}")


if __name__ == "__main__":
    print("=" * 60)
    print("  BAGIAN 14 — BUSINESS CASE & ROI ANALYSIS")
    print("=" * 60)

    # 14.1.1 Cost savings
    print("\n── 14.1.1 Estimasi Penghematan Biaya (ESTIMASI) ──")
    baseline_metrics = {"total_cost": 10406.56}
    neurostock_metrics = {"total_cost": 8455.36}

    savings = calculate_cost_savings(
        baseline_metrics, neurostock_metrics,
        n_stores=5, annualization_factor=365/90
    )
    for k, v in savings.items():
        print(f"  {k:<35}: {v:,.2f}")

    # 14.1.2 Sensitivity analysis
    def mock_evaluate(holding_cost, stockout_penalty):
        # Simulasi: NeuroStock lebih unggul saat stockout_penalty tinggi
        advantage = 15 + (stockout_penalty - 2) * 3 - (holding_cost - 0.5) * 2
        return {"savings_percentage": round(advantage, 1)}

    cost_sensitivity_analysis(
        holding_cost_range=[0.3, 0.5, 0.7, 1.0],
        stockout_penalty_range=[1.0, 2.0, 3.0, 5.0],
        evaluate_fn=mock_evaluate
    )

    # 14.2.1 NPV
    print("\n── 14.2.1 NPV Analysis ──")
    annual_savings = savings["estimated_annual_savings"]

    # Biaya implementasi realistis
    implementation_costs = {
        "Cloud compute (training)": 500,
        "Engineering time (2 minggu)": 3000,
        "Maintenance tahunan": 1000,
    }
    total_investment = sum(implementation_costs.values())

    print("Asumsi biaya implementasi:")
    for item, cost in implementation_costs.items():
        print(f"  {item:<30}: Rp {cost:,}")
    print(f"  {'Total investasi':<30}: Rp {total_investment:,}")

    npv_result = calculate_npv(
        initial_investment=total_investment,
        annual_savings=annual_savings,
        discount_rate=0.10,
        years=3
    )
    print("\nHasil NPV (3 tahun, discount rate 10%):")
    print(f"  NPV                          : {npv_result['npv']:,.2f}")
    print(f"  Payback period               : {npv_result['payback_period_years']:.2f} tahun")
    print(f"  Profitable                   : {'✅ Ya' if npv_result['is_profitable'] else '❌ Tidak'}")
    print(f"  Discounted cash flows        : {npv_result['discounted_cash_flows']}")

    # 14.3.1 Translation table
    print_business_translation()

    print("\n⚠️  DISCLAIMER: Seluruh angka adalah ESTIMASI berdasarkan simulasi.")
    print("   Asumsi: holding_cost=0.5/unit/hari, stockout_penalty=2.0/unit")
    print("   Hasil aktual bergantung pada skala deployment dan data nyata.")
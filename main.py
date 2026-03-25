"""
main.py – Example script to validate the Beneish M-Score implementation.

Fictional data based on plausible accounting figures for illustration only.
Run: python main.py
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from advisor_brain_fsa import BeneishMScore
from advisor_brain_fsa.beneish_mscore import FinancialData


# ---------------------------------------------------------------------------
# Scenario 1 – Healthy Company ("SafeCo")
# ---------------------------------------------------------------------------

safecos_t = FinancialData(
    revenues=1_200_000,
    cost_of_goods_sold=720_000,
    sales_general_admin_expenses=120_000,
    receivables=100_000,
    total_assets=900_000,
    current_assets=300_000,
    pp_and_e=400_000,
    securities=50_000,
    total_long_term_debt=180_000,
    current_liabilities=90_000,
    depreciation=40_000,
    net_income=120_000,
    cash_from_operations=150_000,   # CFO > NI → low accruals (healthy signal)
)

safecos_t1 = FinancialData(
    revenues=1_000_000,
    cost_of_goods_sold=600_000,
    sales_general_admin_expenses=100_000,
    receivables=80_000,
    total_assets=800_000,
    current_assets=260_000,
    pp_and_e=360_000,
    securities=40_000,
    total_long_term_debt=160_000,
    current_liabilities=80_000,
    depreciation=36_000,
    net_income=100_000,
    cash_from_operations=130_000,
)

# ---------------------------------------------------------------------------
# Scenario 2 – Aggressive Accounting ("RiskyInc")
# ---------------------------------------------------------------------------

riskyinc_t = FinancialData(
    revenues=1_500_000,
    cost_of_goods_sold=1_200_000,        # Very thin gross margin (deteriorating)
    sales_general_admin_expenses=300_000,
    receivables=400_000,                  # Receivables growing much faster than sales
    total_assets=1_000_000,
    current_assets=200_000,
    pp_and_e=200_000,
    securities=20_000,
    total_long_term_debt=500_000,         # Heavy leverage
    current_liabilities=200_000,
    depreciation=10_000,                  # Suspiciously low depreciation rate
    net_income=200_000,
    cash_from_operations=20_000,          # Huge gap between NI and CFO → large accruals
)

riskyinc_t1 = FinancialData(
    revenues=1_000_000,
    cost_of_goods_sold=700_000,
    sales_general_admin_expenses=150_000,
    receivables=100_000,
    total_assets=800_000,
    current_assets=250_000,
    pp_and_e=280_000,
    securities=30_000,
    total_long_term_debt=200_000,
    current_liabilities=100_000,
    depreciation=30_000,
    net_income=80_000,
    cash_from_operations=100_000,
)


def run_analysis(name: str, current: FinancialData, prior: FinancialData) -> None:
    print(f"\n{'#' * 55}")
    print(f"  Company: {name}")
    print(f"{'#' * 55}")

    model = BeneishMScore(current=current, prior=prior)
    result = model.calculate()

    print(result)

    print("\n  Detail as Pandas Series:")
    print(result.to_series().to_string())


def main() -> None:
    print("\n" + "=" * 55)
    print("     ADVISOR-BRAIN-FSA | Beneish M-Score Demo")
    print("=" * 55)

    run_analysis("SafeCo (Non-Manipulator expected)", safecos_t, safecos_t1)
    run_analysis("RiskyInc (Potential Manipulator expected)", riskyinc_t, riskyinc_t1)

    # Summary table
    results = []
    for name, t, t1 in [
        ("SafeCo", safecos_t, safecos_t1),
        ("RiskyInc", riskyinc_t, riskyinc_t1),
    ]:
        r = BeneishMScore(current=t, prior=t1).calculate().to_dict()
        r["Company"] = name
        results.append(r)

    df = pd.DataFrame(results).set_index("Company")
    print("\n\n  Comparative Summary Table")
    print("-" * 55)
    print(df.to_string())
    print()


if __name__ == "__main__":
    main()

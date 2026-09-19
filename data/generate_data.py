"""
ML Sentinel — synthetic customer-churn dataset generator.

Produces a realistic, intentionally-messy tabular dataset so the
preprocessing / drift / risk modules have real problems to catch:
missing values, duplicates, outliers, class imbalance, a leaking
column, a PII column, and a categorical feature whose distribution
we deliberately shift between "training" and "production" splits
(so the drift engine has something genuine to detect).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).parent


def _base_frame(n: int, plan_shift: bool = False) -> pd.DataFrame:
    age = RNG.normal(42, 12, n).clip(18, 85)
    tenure_months = RNG.exponential(24, n).clip(0, 120)
    monthly_charges = RNG.normal(70, 25, n).clip(10, 200)
    income = RNG.normal(55000, 20000, n).clip(12000, 250000)
    debt = (income * RNG.uniform(0.05, 0.6, n)).round(0)
    credit_score = RNG.normal(650, 80, n).clip(300, 850)
    support_tickets = RNG.poisson(1.5, n)

    if plan_shift:
        # production traffic skews toward the "premium" and "family"
        # plans much more than training data did -> real feature drift
        plan = RNG.choice(
            ["basic", "standard", "premium", "family"],
            size=n, p=[0.10, 0.25, 0.45, 0.20],
        )
    else:
        plan = RNG.choice(
            ["basic", "standard", "premium", "family"],
            size=n, p=[0.35, 0.35, 0.20, 0.10],
        )

    contract = RNG.choice(["month-to-month", "one-year", "two-year"], size=n, p=[0.55, 0.25, 0.20])
    payment_method = RNG.choice(
        ["credit_card", "bank_transfer", "e_check", "mailed_check"], size=n
    )
    region = RNG.choice(["north", "south", "east", "west"], size=n)

    # churn probability driven by a real underlying signal
    logit = (
        -1.2
        + 0.9 * (contract == "month-to-month")
        - 0.6 * (contract == "two-year")
        + 0.015 * (monthly_charges - 70)
        - 0.02 * (tenure_months - 24)
        + 0.25 * support_tickets
        - 0.004 * (credit_score - 650)
        + 0.3 * (payment_method == "e_check")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = RNG.binomial(1, prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{100000+i}" for i in range(n)],
            "email": [f"user{i}@example.com" for i in range(n)],  # PII column
            "age": age.round(0),
            "tenure_months": tenure_months.round(1),
            "monthly_charges": monthly_charges.round(2),
            "annual_income": income.round(0),
            "outstanding_debt": debt,
            "credit_score": credit_score.round(0),
            "support_tickets_90d": support_tickets,
            "plan": plan,
            "contract_type": contract,
            "payment_method": payment_method,
            "region": region,
            "churned": churn,
        }
    )
    return df


def make_datasets(n_train: int = 8000, n_prod: int = 2000, out_dir: Path = OUT_DIR) -> None:
    train = _base_frame(n_train, plan_shift=False)
    prod = _base_frame(n_prod, plan_shift=True)  # simulates real-world drift

    # inject realistic messiness into TRAIN only (prod assumed already "cleaned" at ingest,
    # so preprocessing module has real work to do on train, drift module compares clean features)
    idx = RNG.choice(train.index, size=int(0.04 * len(train)), replace=False)
    train.loc[idx, "credit_score"] = np.nan
    idx2 = RNG.choice(train.index, size=int(0.02 * len(train)), replace=False)
    train.loc[idx2, "annual_income"] = np.nan
    # outliers
    out_idx = RNG.choice(train.index, size=20, replace=False)
    train.loc[out_idx, "monthly_charges"] = RNG.uniform(500, 900, len(out_idx))
    # exact duplicates
    dupes = train.sample(30, random_state=1)
    train = pd.concat([train, dupes], ignore_index=True)
    # a near-leaking column (near-perfect proxy for target) to test leakage detection
    train["last_invoice_status"] = np.where(
        train["churned"] == 1,
        RNG.choice(["overdue", "unpaid"], size=len(train), p=[0.7, 0.3]),
        RNG.choice(["paid", "overdue"], size=len(train), p=[0.95, 0.05]),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train_raw.csv", index=False)
    prod.to_csv(out_dir / "production_traffic.csv", index=False)
    print(f"Wrote {len(train)} training rows -> train_raw.csv")
    print(f"Wrote {len(prod)} production rows -> production_traffic.csv")


if __name__ == "__main__":
    make_datasets()

import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────
N_HIST = 10000   # Dataset 1 — historical, có label
N_CURR = 5000    # Dataset 2 — current, không có label

HIST_OBS_DATE  = datetime(2024, 6, 30)
HIST_LAB_START = datetime(2024, 7, 1)
HIST_LAB_END   = datetime(2024, 9, 30)

CURR_OBS_DATE  = datetime(2025, 6, 30)


# ─────────────────────────────────────────────────────────────
# GENERATE FEATURE STORE DIRECTLY
# ─────────────────────────────────────────────────────────────
def generate_feature_store(n, obs_date, id_start=0, seed=42):
    np.random.seed(seed)

    # ── Demographics ──────────────────────────────────────────
    monthly_salary      = np.random.lognormal(16.4, 0.5, n).clip(5e6, 100e6)
    age                 = np.random.normal(35, 8, n).clip(22, 60).astype(int)
    gender              = np.random.choice(['M', 'F'], n, p=[0.55, 0.45])
    province            = np.random.choice(
                              ['HN', 'HCM', 'DN', 'HP', 'CT', 'OTHER'], n,
                              p=[0.25, 0.30, 0.10, 0.08, 0.07, 0.20])
    tenure_months       = np.random.exponential(24, n).clip(6, 120).astype(int)
    has_mobile_banking  = np.random.binomial(1, 0.7, n)
    num_products        = np.random.randint(1, 6, n)
    has_loan            = np.random.binomial(1, 0.2, n)

    # ── Nhóm 1: Income ────────────────────────────────────────
    salary_mean_12m     = monthly_salary * np.random.normal(1.0, 0.02, n)
    salary_std_12m      = salary_mean_12m * np.random.uniform(0.01, 0.08, n)
    salary_cv_12m       = (salary_std_12m / salary_mean_12m).round(4)
    salary_min_12m      = salary_mean_12m * np.random.uniform(0.85, 0.98, n)
    salary_max_12m      = salary_mean_12m * np.random.uniform(1.02, 1.20, n)
    salary_trend_12m    = np.random.normal(50000, 200000, n)   # VND/tháng
    salary_delay_rate   = np.random.beta(1, 19, n).round(4)    # thường < 10%

    has_term_deposit         = np.random.binomial(1, 0.4, n)
    term_deposit_count       = has_term_deposit * np.random.randint(1, 4, n)
    term_deposit_total       = has_term_deposit * salary_mean_12m * np.random.uniform(2, 12, n)
    term_deposit_interest    = (term_deposit_total * np.random.uniform(0.04, 0.07, n)).round(0)

    has_savings_account      = np.random.binomial(1, 0.4, n)
    savings_avg_balance_12m  = has_savings_account * salary_mean_12m * np.random.uniform(1, 3, n)
    savings_min_balance_avg  = savings_avg_balance_12m * np.random.uniform(0.3, 0.8, n)
    savings_trend_12m        = np.random.normal(100000, 500000, n)

    total_income_proxy       = (salary_mean_12m * 12 + term_deposit_interest).round(0)

    # ── Nhóm 2: Spending ──────────────────────────────────────
    cost_level          = np.random.uniform(0.8, 1.2, n)
    shopping_tendency   = np.random.uniform(0.5, 1.5, n)
    edu_tendency        = np.random.uniform(0.0, 1.0, n)
    entertain_tendency  = np.random.uniform(0.3, 1.3, n)

    bills_electric_avg  = (np.random.uniform(200_000, 800_000, n)   * cost_level).round(0)
    bills_water_avg     = (np.random.uniform(50_000,  200_000, n)   * cost_level).round(0)
    bills_internet_avg  = (np.random.uniform(150_000, 400_000, n)   * cost_level).round(0)
    bills_total_avg     = (bills_electric_avg + bills_water_avg + bills_internet_avg).round(0)
    bills_to_income     = (bills_total_avg / salary_mean_12m).round(4)

    edu_spend_avg       = (np.random.uniform(0, 5_000_000, n) * edu_tendency * 0.4).round(0)
    entertainment_avg   = (np.random.uniform(200_000, 2_000_000, n) * entertain_tendency).round(0)
    grocery_spend_avg   = (np.random.uniform(500_000, 3_000_000, n) * shopping_tendency).round(0)
    dining_spend_avg    = (np.random.uniform(200_000, 2_000_000, n) * shopping_tendency).round(0)
    shopping_spend_avg  = (np.random.uniform(300_000, 5_000_000, n) * shopping_tendency).round(0)
    transport_spend_avg = (np.random.uniform(100_000, 1_000_000, n) * shopping_tendency).round(0)
    total_shopping_avg  = (grocery_spend_avg + dining_spend_avg +
                           shopping_spend_avg + transport_spend_avg).round(0)
    transfer_avg        = (np.random.uniform(0, 5_000_000, n) * np.random.binomial(1, 0.3, n)).round(0)

    total_spend_avg     = (bills_total_avg + edu_spend_avg + entertainment_avg +
                           total_shopping_avg + transfer_avg).round(0)
    discretionary_avg   = (total_spend_avg - bills_total_avg).round(0)
    essential_ratio     = (bills_total_avg / total_spend_avg).round(4)

    # ── Nhóm 3a: Financial Stress ─────────────────────────────
    spend_to_income     = (total_spend_avg / salary_mean_12m).round(4)
    monthly_surplus_avg = (salary_mean_12m - total_spend_avg).round(0)
    cashflow_volatility = np.abs(monthly_surplus_avg * np.random.uniform(0.1, 0.5, n)).round(0)
    months_neg_balance  = np.where(
        monthly_surplus_avg < 0,
        np.random.randint(1, 12, n),
        np.random.randint(0, 3, n)
    )
    overdraft_rate      = (months_neg_balance / 12).round(4)

    # ── Nhóm 3b: Credit Need ──────────────────────────────────
    high_spend_months   = np.where(
        spend_to_income > 0.9,
        np.random.randint(3, 12, n),
        np.random.randint(0, 3, n)
    )
    end_month_shortfall = np.where(
        monthly_surplus_avg < 0,
        monthly_surplus_avg * np.random.uniform(0.8, 1.2, n),
        0
    ).round(0)
    spending_accel      = np.random.normal(0, 0.005, n).round(6)

    # ── Nhóm 3c: Savings Capacity ─────────────────────────────
    savings_rate_avg    = (monthly_surplus_avg / salary_mean_12m).round(4)
    total_wealth        = savings_avg_balance_12m + term_deposit_total
    wealth_accumulation = (total_wealth / (salary_mean_12m * 12)).round(4)

    # ── Nhóm 3d: Life Stage ───────────────────────────────────
    edu_spend_trend     = np.random.normal(0, 50000, n).round(0)
    large_transfer_count= np.random.poisson(1.5, n)
    large_transfer_avg  = (large_transfer_count * np.random.uniform(5e6, 20e6, n)).round(0)
    income_jump         = np.random.beta(1, 10, n).round(4)
    lifestyle_inflation = np.random.uniform(-0.3, 0.9, n).round(4)

    # ── Assemble ──────────────────────────────────────────────
    df = pd.DataFrame({
        # IDs
        'customer_id'              : [f"KH{str(i).zfill(6)}" for i in range(id_start, id_start + n)],
        'observation_date'         : obs_date.date(),

        # Demographics
        'age'                      : age,
        'gender'                   : gender,
        'province'                 : province,
        'tenure_months'            : tenure_months,
        'has_mobile_banking'       : has_mobile_banking,
        'num_products'             : num_products,
        'has_loan'                 : has_loan,

        # Nhóm 1 — Income
        'salary_mean_12m'          : salary_mean_12m.round(0),
        'salary_std_12m'           : salary_std_12m.round(0),
        'salary_cv_12m'            : salary_cv_12m,
        'salary_min_12m'           : salary_min_12m.round(0),
        'salary_max_12m'           : salary_max_12m.round(0),
        'salary_trend_12m'         : salary_trend_12m.round(0),
        'salary_delay_rate_12m'    : salary_delay_rate,
        'has_term_deposit'         : has_term_deposit,
        'term_deposit_count'       : term_deposit_count,
        'term_deposit_total'       : term_deposit_total.round(0),
        'term_deposit_interest_annual': term_deposit_interest,
        'has_savings_account'      : has_savings_account,
        'savings_avg_balance_12m'  : savings_avg_balance_12m.round(0),
        'savings_min_balance_avg'  : savings_min_balance_avg.round(0),
        'savings_trend_12m'        : savings_trend_12m.round(0),
        'total_income_proxy'       : total_income_proxy,

        # Nhóm 2 — Spending
        'bills_electric_avg'       : bills_electric_avg,
        'bills_water_avg'          : bills_water_avg,
        'bills_internet_avg'       : bills_internet_avg,
        'bills_total_avg'          : bills_total_avg,
        'bills_to_income_ratio'    : bills_to_income,
        'edu_spend_avg'            : edu_spend_avg,
        'entertainment_avg'        : entertainment_avg,
        'grocery_spend_avg'        : grocery_spend_avg,
        'dining_spend_avg'         : dining_spend_avg,
        'shopping_spend_avg'       : shopping_spend_avg,
        'transport_spend_avg'      : transport_spend_avg,
        'total_shopping_avg'       : total_shopping_avg,
        'transfer_avg'             : transfer_avg,
        'total_spend_avg'          : total_spend_avg,
        'discretionary_spend_avg'  : discretionary_avg,
        'essential_spend_ratio'    : essential_ratio,

        # Nhóm 3a — Financial Stress
        'spend_to_income_ratio'    : spend_to_income,
        'monthly_surplus_avg'      : monthly_surplus_avg,
        'cashflow_volatility'      : cashflow_volatility,
        'months_negative_balance'  : months_neg_balance,
        'overdraft_rate'           : overdraft_rate,

        # Nhóm 3b — Credit Need
        'high_spend_months'        : high_spend_months,
        'end_month_shortfall_avg'  : end_month_shortfall,
        'spending_acceleration'    : spending_accel,

        # Nhóm 3c — Savings Capacity
        'savings_rate_avg'         : savings_rate_avg,
        'wealth_accumulation'      : wealth_accumulation,

        # Nhóm 3d — Life Stage
        'edu_spend_trend'          : edu_spend_trend,
        'large_transfer_count'     : large_transfer_count,
        'large_transfer_avg'       : large_transfer_avg,
        'income_jump_detected'     : income_jump,
        'lifestyle_inflation'      : lifestyle_inflation,
    })

    return df


# ─────────────────────────────────────────────────────────────
# GENERATE LABEL — historical only
# ─────────────────────────────────────────────────────────────
def generate_label(feature_store, label_start, label_end):
    df = feature_store[['customer_id']].copy()

    df['label_window_start'] = label_start.date()
    df['label_window_end']   = label_end.date()

    # Conversion probability phụ thuộc vào features
    fs = feature_store
    log_odds = (
        -1.5
        + 0.02  * (fs['age'] - 35)
        - 0.001 * (fs['age'] - 35) ** 2
        + 0.3   * (1 - fs['salary_cv_12m'])          # lương ổn định
        + 0.2   * np.log(fs['salary_mean_12m'] / 1e7)
        + 0.5   * fs['spend_to_income_ratio']         # cần credit
        + 0.3   * (1 - fs['overdraft_rate'])           # không overdraft
        + 0.2   * fs['has_mobile_banking']
        + 0.15  * fs['lifestyle_inflation']            # spending tăng cùng lương
        + 0.1   * (fs['num_products'] - 1)
        + 0.1   * fs['has_savings_account']
        + np.random.normal(0, 0.5, len(fs))
    )

    conversion_prob  = 1 / (1 + np.exp(-log_odds))
    df['converted']  = np.random.binomial(1, conversion_prob)

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Dataset 1: Historical — train model ───────────────────
    print("=" * 55)
    print("DATASET 1 — Historical (obs: 30/6/2024)")
    print("=" * 55)

    hist_features = generate_feature_store(
        n        = N_HIST,
        obs_date = HIST_OBS_DATE,
        id_start = 0,
        seed     = 42
    )
    hist_features.to_csv("data/feature_store_historical.csv", index=False)
    print(f"feature_store_historical.csv : {hist_features.shape[0]:,} rows × {hist_features.shape[1]} cols")

    hist_label = generate_label(hist_features, HIST_LAB_START, HIST_LAB_END)
    hist_label.to_csv("data/label_historical.csv", index=False)
    print(f"label_historical.csv         : {hist_label.shape[0]:,} rows × {hist_label.shape[1]} cols")
    print(f"Conversion rate              : {hist_label['converted'].mean():.2%}")

    # ── Dataset 2: Current — deploy model ─────────────────────
    print("\n" + "=" * 55)
    print("DATASET 2 — Current (obs: 30/6/2025) — no label")
    print("=" * 55)

    curr_features = generate_feature_store(
        n        = N_CURR,
        obs_date = CURR_OBS_DATE,
        id_start = 20000,
        seed     = 99
    )
    curr_features.to_csv("data/feature_store_current.csv", index=False)
    print(f"feature_store_current.csv    : {curr_features.shape[0]:,} rows × {curr_features.shape[1]} cols")
    print(f"No label — apply model first, send scores back")

    print("\nDone. Files saved to data/")
    print("\nNext steps:")
    print("  1. Train propensity model on feature_store_historical.csv + label_historical.csv")
    print("  2. Apply model on feature_store_current.csv → propensity scores")
    print("  3. Send scores back → generate campaign outcomes for A/B test")
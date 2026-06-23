import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_customers(n, start_date, end_date):
    """Generate customer base với behavioral features"""
    
    # IDs 
    customer_ids = [f"KH{str(i).zfill(6)}" for i in range(n)]
    
    # Demographics 
    age = np.random.normal(35, 8, n).clip(22, 60)
    gender = np.random.choice(['M', 'F'], n, p=[0.55, 0.45])
    province = np.random.choice(
        ['HN', 'HCM', 'DN', 'HP', 'CT', 'OTHER'],
        n, p=[0.25, 0.30, 0.10, 0.08, 0.07, 0.20]
    )
    
    # Tenure 
    tenure_months = np.random.exponential(24, n).clip(6, 120)
    
    # Salary features 
    monthly_salary = np.random.lognormal(16.4, 0.5, n).clip(5e6, 100e6)
    salary_regularity = np.random.beta(5, 2, n)
    salary_growth_3m = np.random.normal(0.02, 0.05, n)
    
    #  Spending behavior 
    avg_monthly_spend = (monthly_salary * np.random.beta(2, 3, n)).clip(1e6, 50e6)
    spend_to_income_ratio = avg_monthly_spend / monthly_salary
    transaction_frequency = np.random.poisson(15, n).clip(1, 60)
    avg_ticket_size = avg_monthly_spend / transaction_frequency.clip(1)
    
    # Product holding
    num_products = np.random.randint(1, 6, n)
    has_savings = np.random.binomial(1, 0.7, n)
    has_loan = np.random.binomial(1, 0.2, n)
    has_mobile_banking = np.random.binomial(1, 0.7, n)
    
    # Repayment behavior
    bill_payment_ontime_rate = np.random.beta(4, 1.5, n)
    overdraft_count_6m = np.random.poisson(0.5, n)
    
    # Digital engagement 
    app_login_monthly = np.random.poisson(8, n).clip(0, 60)
    online_txn_rate = np.random.beta(2, 3, n)
    
    # Observation date 
    date_range = (end_date - start_date).days
    obs_dates = [
        start_date + timedelta(days=np.random.randint(0, date_range))
        for _ in range(n)
    ]
    
    # Conversion probability
    log_odds = (
        -1.5
        + 0.02  * (age - 35)
        - 0.001 * (age - 35) ** 2
        + 0.3   * salary_regularity
        + 0.2   * np.log(monthly_salary / 1e7)
        + 0.5   * spend_to_income_ratio
        + 0.3   * bill_payment_ontime_rate
        - 0.2   * overdraft_count_6m
        + 0.2   * has_mobile_banking
        + 0.1   * np.log1p(app_login_monthly)
        + 0.15  * online_txn_rate
        + 0.1   * (num_products - 1)
        + 0.15  * salary_growth_3m
        + np.random.normal(0, 0.5, n)
    )
    conversion_prob = 1 / (1 + np.exp(-log_odds))
    converted = np.random.binomial(1, conversion_prob)
    
    df = pd.DataFrame({
        # IDs và dates
        'customer_id': customer_ids,
        'observation_date': obs_dates,
        
        # Demographics
        'age': age.round(0).astype(int),
        'gender': gender,
        'province': province,
        'tenure_months': tenure_months.round(0).astype(int),
        
        # Salary
        'monthly_salary': monthly_salary.round(0),
        'salary_regularity': salary_regularity.round(4),
        'salary_growth_3m': salary_growth_3m.round(4),
        
        # Spending
        'avg_monthly_spend': avg_monthly_spend.round(0),
        'spend_to_income_ratio': spend_to_income_ratio.round(4),
        'transaction_frequency': transaction_frequency,
        'avg_ticket_size': avg_ticket_size.round(0),
        
        # Products
        'num_products': num_products,
        'has_savings': has_savings,
        'has_loan': has_loan,
        'has_mobile_banking': has_mobile_banking,
        
        # Repayment
        'bill_payment_ontime_rate': bill_payment_ontime_rate.round(4),
        'overdraft_count_6m': overdraft_count_6m,
        
        # Digital
        'app_login_monthly': app_login_monthly,
        'online_txn_rate': online_txn_rate.round(4),
        
        # Target
        'conversion_prob': conversion_prob.round(4),  # ground truth, không dùng trong model
        'converted': converted
    })
    
    return df


if __name__ == "__main__":
    
    # Data cho train model (historical: 2022-2024) 
    print("Generating model training data...")
    df_model = generate_customers(
        n=50000,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
    df_model.to_csv("data/data_model.csv", index=False)
    print(f"  Shape: {df_model.shape}")
    print(f"  Conversion rate: {df_model['converted'].mean():.2%}")
    
    #  Data cho A/B test (recent cohort: 2025) 
    print("\nGenerating A/B test data...")
    df_abtest = generate_customers(
        n=5000,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30)
    )
    
    # Assign treatment/control (70/30 split, stratified by salary band)
    df_abtest['salary_band'] = pd.cut(
    df_abtest['monthly_salary'],
    bins=[0, 10e6, 20e6, float('inf')],
    labels=['low', 'mid', 'high']
)
    
    # Random assignment within each salary band
    np.random.seed(123)
    df_abtest['group'] = 'control'
    for band in ['low', 'mid', 'high']:
        mask = df_abtest['salary_band'] == band
        idx = df_abtest[mask].index
        n_treatment = int(len(idx) * 0.7)
        treatment_idx = np.random.choice(idx, n_treatment, replace=False)
        df_abtest.loc[treatment_idx, 'group'] = 'treatment'
    
    # Treatment effect: campaign tăng conversion thêm 5%
    treatment_mask = df_abtest['group'] == 'treatment'
    uplift = np.random.binomial(
        1,
        0.05,
        treatment_mask.sum()
    )
    df_abtest.loc[treatment_mask, 'converted'] = (
        df_abtest.loc[treatment_mask, 'converted'].values | uplift
    ).astype(int)
    
    df_abtest.to_csv("data/data_abtest.csv", index=False)
    print(f"  Shape: {df_abtest.shape}")
    print(f"  Treatment group: {(df_abtest['group']=='treatment').sum()}")
    print(f"  Control group: {(df_abtest['group']=='control').sum()}")
    print(f"  Conversion - Treatment: {df_abtest[df_abtest['group']=='treatment']['converted'].mean():.2%}")
    print(f"  Conversion - Control: {df_abtest[df_abtest['group']=='control']['converted'].mean():.2%}")
    
    print("\nDone. Files saved to data/")
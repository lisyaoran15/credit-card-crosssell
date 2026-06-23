```markdown
# Data Documentation

## Project Overview

**Bài toán**: Cross-sell thẻ tín dụng cho KH đổ lương tại BIDV chưa có thẻ tín dụng.

**Mục tiêu**: Build propensity model để predict KH nào có khả năng activate thẻ và tiêu ≥ 1.5 triệu VND trong tháng đầu tiên, sau đó validate bằng A/B test.

---

## Temporal Structure

```
Feature window:   6/2024 → 5/2025   (12 tháng hành vi — input của model)
Buffer month:     6/2025             (không dùng — tránh leakage)
Observation date: 30/6/2025          (thời điểm snapshot, chốt features)
Label window:     1/7/2025 → 30/9/2025 (3 tháng observe conversion — target Y)
```

**Critical rule**: Không được dùng bất kỳ thông tin nào sau observation_date làm feature. Mọi aggregation phải kết thúc tại FEATURE_END (31/5/2025).

---

## Two Datasets

| Dataset | Mục đích | Obs date | Feature window | N customers |
|---|---|---|---|---|
| Dataset 1 (no prefix) | Train propensity model | 30/6/2025 | 6/2024–5/2025 | 10,000 |
| Dataset 2 (ab_ prefix) | Validate A/B test | 30/6/2024 | 6/2023–5/2024 | 3,000 |

Dataset 2 là cohort cũ hơn — simulate việc đã chạy campaign trước đó và có kết quả để validate.

---

## File Structure

```
data/
├── customers.csv                  # Master table — Dataset 1
├── salary_history.csv             # Lịch sử lương — Dataset 1
├── savings_balance.csv            # Balance không kỳ hạn — Dataset 1
├── term_deposits.csv              # Tiết kiệm có kỳ hạn — Dataset 1
├── monthly_behavior.csv           # Spending aggregates — Dataset 1
├── feature_store.csv              # Features đã engineer — Dataset 1
├── campaign_assignments.csv       # A/B test + target — Dataset 1
│
├── ab_customers.csv               # Master table — Dataset 2
├── ab_salary_history.csv          # Lịch sử lương — Dataset 2
├── ab_savings_balance.csv         # Balance không kỳ hạn — Dataset 2
├── ab_term_deposits.csv           # Tiết kiệm có kỳ hạn — Dataset 2
├── ab_monthly_behavior.csv        # Spending aggregates — Dataset 2
├── ab_feature_store.csv           # Features đã engineer — Dataset 2
└── ab_campaign_assignments.csv    # A/B test + target — Dataset 2
```

---

## Table Schemas

### customers.csv

Master table. Mỗi row là một KH, capture tại observation_date.

| Column | Type | Mô tả |
|---|---|---|
| customer_id | string | Mã KH (KH000000...) |
| observation_date | date | Ngày snapshot |
| age | int | Tuổi (22–60) |
| gender | string | M = Nam, F = Nữ |
| province | string | HN, HCM, DN, HP, CT, OTHER |
| tenure_months | int | Số tháng có tài khoản tại BIDV (6–120) |
| monthly_salary | float | Thu nhập lương TB/tháng (VND) |
| has_mobile_banking | int | Đã đăng ký mobile banking: 1/0 |
| num_products | int | Số sản phẩm đang có (1–5) |
| has_loan | int | Đang có khoản vay: 1/0 |

---

### salary_history.csv

Lịch sử lương theo tháng trong feature window. Join với customers qua customer_id.

| Column | Type | Mô tả |
|---|---|---|
| customer_id | string | FK → customers |
| year_month | string | Tháng (YYYY-MM) |
| salary_amount | float | Số tiền lương được credit (VND) |
| credit_date | date | Ngày lương thực tế đổ vào tài khoản |
| is_regular | int | Lương đúng hạn: 1 = đúng, 0 = trễ |

**Notes**: 5% tháng có `is_regular = 0`. Lương tăng nhẹ ~0.1%/tháng.

---

### savings_balance.csv

Balance tài khoản tiết kiệm không kỳ hạn, snapshot cuối tháng. Chỉ 40% KH có bảng này.

| Column | Type | Mô tả |
|---|---|---|
| customer_id | string | FK → customers |
| year_month | string | Tháng (YYYY-MM) |
| closing_balance | float | Balance cuối tháng (VND) |
| min_balance | float | Balance thấp nhất trong tháng (VND) |
| avg_balance | float | Balance trung bình trong tháng (VND) |

---

### term_deposits.csv

Sổ tiết kiệm có kỳ hạn. Chỉ giữ các sổ còn active tại observation_date. Chỉ 40% KH có, mỗi KH 1–3 sổ.

| Column | Type | Mô tả |
|---|---|---|
| customer_id | string | FK → customers |
| deposit_id | string | Mã sổ tiết kiệm (TD + 6 số) |
| start_date | date | Ngày mở sổ |
| maturity_date | date | Ngày đáo hạn |
| amount | float | Số tiền gửi (VND) |
| term_months | int | Kỳ hạn: 1, 3, 6, 12, 24 tháng |
| interest_rate | float | Lãi suất/năm |

**Interest rate by term**:

| Term | Rate |
|---|---|
| 1 tháng | 4.0% |
| 3 tháng | 4.5% |
| 6 tháng | 5.5% |
| 12 tháng | 6.5% |
| 24 tháng | 7.0% |

---

### monthly_behavior.csv

Spending aggregates theo tháng. Mỗi row là một KH × một tháng.

| Column | Type | Mô tả |
|---|---|---|
| customer_id | string | FK → customers |
| year_month | string | Tháng (YYYY-MM) |
| total_inflow | float | Tổng tiền vào tài khoản trong tháng (VND) |
| electric | float | Tiền điện (VND) |
| water | float | Tiền nước (VND) |
| internet | float | Internet/điện thoại (VND) |
| total_bills | float | Tổng hóa đơn cố định (VND) |
| edu_spend | float | Chi phí giáo dục (VND) — 0 nếu không có |
| entertainment | float | Giải trí, streaming (VND) |
| groceries | float | Thực phẩm (VND) |
| dining | float | Ăn uống nhà hàng (VND) |
| shopping | float | Mua sắm thời trang, đồ dùng (VND) |
| transport | float | Di chuyển, xăng, grab (VND) |
| total_shopping | float | Tổng shopping (groceries+dining+shopping+transport) |
| transfer_amount | float | Chuyển khoản lớn trong tháng (VND) — 0 nếu không có |
| total_spend | float | Tổng chi tiêu (bills+edu+entertainment+shopping+transfer) |
| monthly_surplus | float | Income - Spend (âm = thiếu tiền tháng đó) |
| spend_ratio | float | total_spend / monthly_salary |

---

### feature_store.csv

Features đã được engineer sẵn, ready để đưa vào model. Join với campaign_assignments qua customer_id để có target Y.

#### Nhóm 1 — Income Features

| Feature | Mô tả |
|---|---|
| salary_mean_12m | Lương TB 12 tháng (VND) |
| salary_std_12m | Std dev lương 12 tháng |
| salary_cv_12m | Coefficient of variation lương = std/mean. Cao = không ổn định |
| salary_min_12m | Lương thấp nhất trong 12 tháng |
| salary_max_12m | Lương cao nhất trong 12 tháng |
| salary_trend_12m | Slope của lương theo thời gian. Dương = đang tăng lương |
| salary_delay_rate_12m | Tỷ lệ tháng lương đổ trễ (0–1). Cao = không đáng tin |
| has_term_deposit | Có tiết kiệm có kỳ hạn: 1/0 |
| term_deposit_count | Số sổ tiết kiệm có kỳ hạn đang active |
| term_deposit_total | Tổng tiền gửi có kỳ hạn (VND) |
| term_deposit_interest_annual | Lãi tiết kiệm ước tính/năm (VND) |
| has_savings_account | Có tài khoản không kỳ hạn: 1/0 |
| savings_avg_balance_12m | Balance TB tài khoản không kỳ hạn (VND) |
| savings_min_balance_avg | Balance thấp nhất TB/tháng (VND) |
| savings_trend_12m | Slope của balance theo thời gian. Dương = đang tích lũy |
| total_income_proxy | Tổng income ước tính/năm = lương × 12 + lãi tiết kiệm |

#### Nhóm 2 — Spending Features

| Feature | Mô tả |
|---|---|
| bills_electric_avg | Tiền điện TB/tháng |
| bills_water_avg | Tiền nước TB/tháng |
| bills_internet_avg | Internet TB/tháng |
| bills_total_avg | Tổng hóa đơn cố định TB/tháng |
| bills_to_income_ratio | Hóa đơn / lương. Cao = gánh nặng fixed cost lớn |
| edu_spend_avg | Chi phí giáo dục TB/tháng |
| entertainment_avg | Giải trí TB/tháng |
| grocery_spend_avg | Thực phẩm TB/tháng |
| dining_spend_avg | Ăn uống TB/tháng |
| shopping_spend_avg | Mua sắm TB/tháng |
| transport_spend_avg | Di chuyển TB/tháng |
| total_shopping_avg | Tổng shopping TB/tháng |
| transfer_avg | Chuyển khoản TB/tháng |
| total_spend_avg | Tổng chi tiêu TB/tháng |
| discretionary_spend_avg | Chi tiêu tùy ý TB/tháng = total - bills |
| essential_spend_ratio | Bills / total spend. Cao = chi tiêu thiết yếu chiếm phần lớn |

#### Nhóm 3a — Financial Stress

| Feature | Mô tả |
|---|---|
| spend_to_income_ratio | Tổng chi tiêu / lương TB. > 1 = đang chi tiêu vượt thu nhập |
| monthly_surplus_avg | Thu nhập - Chi tiêu TB/tháng. Âm = thường xuyên thiếu tiền |
| cashflow_volatility | Std dev của monthly_surplus. Cao = dòng tiền không ổn định |
| months_negative_balance | Số tháng surplus âm trong 12 tháng |
| overdraft_rate | Tỷ lệ tháng surplus âm = months_negative / 12 |

#### Nhóm 3b — Credit Need

| Feature | Mô tả |
|---|---|
| high_spend_months | Số tháng spend > 90% income — KH đang dùng gần hết lương |
| end_month_shortfall_avg | Mức thiếu hụt TB các tháng âm (VND) — 0 nếu không có tháng âm |
| spending_acceleration | Slope của spend_ratio theo thời gian. Dương = spending đang tăng nhanh hơn income |

#### Nhóm 3c — Savings Capacity

| Feature | Mô tả |
|---|---|
| savings_rate_avg | monthly_surplus / lương. Tỷ lệ tiết kiệm TB |
| wealth_accumulation | Tổng tài sản / annual income. Cao = đã tích lũy tốt |

#### Nhóm 3d — Life Stage Signals

| Feature | Mô tả |
|---|---|
| edu_spend_trend | Slope của chi phí giáo dục. Tăng = có thể đang có con đi học |
| large_transfer_count | Số lần chuyển khoản > 5tr trong 12 tháng |
| large_transfer_avg | Giá trị TB các chuyển khoản lớn (VND) |
| income_jump_detected | Max % tăng lương month-over-month. Cao = vừa được thăng chức/tăng lương |
| lifestyle_inflation | Correlation giữa spending và lương. Cao = spending tăng cùng lúc lương tăng |

---

### campaign_assignments.csv

Kết quả A/B test. Join với feature_store qua customer_id để có X và Y đầy đủ cho model.

| Column | Type | Mô tả |
|---|---|---|
| customer_id | string | FK → customers |
| feature_window_start | date | Bắt đầu feature window |
| feature_window_end | date | Kết thúc feature window |
| observation_date | date | Ngày snapshot |
| label_window_start | date | Bắt đầu observe target |
| label_window_end | date | Kết thúc observe target |
| group | string | treatment = nhận campaign, control = không |
| contacted_date | date | Ngày contact (treatment only) |
| converted | int | **Target Y**: activate thẻ và tiêu ≥ 1.5tr trong tháng đầu: 1/0 |
| conversion_date | date | Ngày convert (converted = 1 only) |

**A/B test design**:

| | Treatment | Control |
|---|---|---|
| Tỷ lệ | 70% | 30% |
| Channel | Push noti + gọi điện | Không contact |
| Contact window | 1–15/7/2025 | — |
| Label window | 1/7/2025 – 30/9/2025 | 1/7/2025 – 30/9/2025 |
| Treatment uplift | Baseline + 5% | Baseline |

**Stratification**: phân bổ treatment/control được stratify theo salary band (low < 10tr, mid 10–20tr, high > 20tr).

---

## Key Relationships

```
customers (10,000)
    ├── salary_history          [1 KH : 12 rows]
    ├── savings_balance         [40% KH : 12 rows mỗi KH]
    ├── term_deposits           [40% KH : 1–3 rows mỗi KH, active only]
    ├── monthly_behavior        [1 KH : 12 rows]
    ├── feature_store           [1 KH : 1 row — ready for model]
    └── campaign_assignments    [1 KH : 1 row — target Y + A/B group]
```

---

## How to Use

### Build propensity model

```python
import pandas as pd

features  = pd.read_csv("data/feature_store.csv")
campaign  = pd.read_csv("data/campaign_assignments.csv")
customers = pd.read_csv("data/customers.csv")

# Merge features + demographics + target
df = features \
    .merge(customers[['customer_id', 'age', 'gender', 'province',
                       'tenure_months', 'has_mobile_banking',
                       'num_products', 'has_loan']], on='customer_id') \
    .merge(campaign[['customer_id', 'converted', 'group']], on='customer_id')

X = df.drop(columns=['customer_id', 'converted', 'group'])
y = df['converted']
```

### A/B test analysis

```python
campaign = pd.read_csv("data/campaign_assignments.csv")

treatment = campaign[campaign['group'] == 'treatment']['converted']
control   = campaign[campaign['group'] == 'control']['converted']

from scipy import stats
z_stat, p_value = stats.ttest_ind(treatment, control)
print(f"Treatment conv: {treatment.mean():.2%}")
print(f"Control conv  : {control.mean():.2%}")
print(f"Incremental   : {treatment.mean() - control.mean():.2%}")
print(f"p-value       : {p_value:.4f}")
```

### Validate A/B test with Dataset 2

```python
ab_campaign = pd.read_csv("data/ab_campaign_assignments.csv")
ab_features = pd.read_csv("data/ab_feature_store.csv")
# Same analysis flow as above, dùng ab_ prefix files
```

---

## Important Notes

**Temporal cutoff**: tuyệt đối không dùng thông tin sau `observation_date` làm feature. `feature_store.csv` đã enforce điều này — chỉ aggregate data trong `[feature_window_start, feature_window_end]`.

**Target variable**: `converted` trong `campaign_assignments.csv` là Y duy nhất. Không có ground truth conversion probability trong production data.

**Missing values**: KH không có savings account sẽ có `has_savings_account = 0` và các savings features = 0. Tương tự với term deposits. Đây là intentional, không phải missing data.

**Join key**: tất cả tables join qua `customer_id`. Dataset 1 dùng KH000000–KH009999, Dataset 2 dùng KH020000–KH022999 để tránh overlap.

**Synthetic data disclaimer**: data này được generate bằng statistical simulation dựa trên domain knowledge banking Việt Nam. Các relationships giữa features và target được hardcode trong data generation — model sẽ học lại những relationships đó từ data.
```
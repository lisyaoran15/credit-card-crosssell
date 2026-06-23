# Data Dictionary

## Hai datasets

| File            | Mô tả                                                     | Số rows|
|-----------------|-----------------------------------------------------------|--------|
| data_model.csv  | Historical data 2022-2024, dùng để train propensity model | 50,000 |
| data_abtest.csv | Recent cohort 2025, dùng để simulate A/B test             | 5,000  |

---

## Columns

### IDs và Dates

| Column           | Type   | Mô tả                                                       |
|------------------|--------|-------------------------------------------------------------|
| customer_id      | string | Mã KH định danh duy nhất (KH000001...)                      |
| observation_date | date   | Ngày quan sát — tất cả features được tính tại thời điểm này |

### Demographics

| Column        | Type   | Mô tả                                     |
|---------------|--------|-------------------------------------------|
| age           | int    | Tuổi KH (22-60)                           |
| gender        | string | Giới tính: M = Nam, F = Nữ                |
| province      | string | Tỉnh thành: HN, HCM, DN, HP, CT, OTHER    |
| tenure_months | int    | Số tháng KH có tài khoản tại BIDV (6-120) |

### Salary Features

| Column | Type | Mô tả |
|-------------------|-------|------------------------------------------------------------------------------|
| monthly_salary    | float | Thu nhập lương trung bình/tháng (VND) — proxy từ salary credit vào tài khoản |
| salary_regularity | float | Độ đều đặn của lương (0-1) — cao = lương vào đúng ngày mỗi tháng             |
| salary_growth_3m  | float | Tốc độ tăng lương 3 tháng gần nhất — dương = đang tăng lương                 |

### Spending Behavior

| Column                | Type  | Mô tả                                                                       |
|-----------------------|-------|-----------------------------------------------------------------------------|
| avg_monthly_spend     | float | Chi tiêu trung bình/tháng (VND) — tính từ debit transactions                |
| spend_to_income_ratio | float | Tỷ lệ chi tiêu/thu nhập — cao = đang spend gần hết lương, có nhu cầu credit |
| transaction_frequency | int   | Số giao dịch trung bình/tháng                                               |
| avg_ticket_size       | float | Giá trị trung bình mỗi giao dịch (VND)                                      |

### Product Holding

| Column             | Type| Mô tả                              |
|--------------------|-----|------------------------------------|
| num_products       | int | Số sản phẩm đang có tại BIDV (1-5) |
| has_savings        | int | Có tài khoản tiết kiệm không (0/1) |
| has_loan           | int | Đang có khoản vay không (0/1)      |
| has_mobile_banking | int | Đã đăng ký                         |
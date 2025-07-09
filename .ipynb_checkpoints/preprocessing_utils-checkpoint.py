import pandas as pd
import numpy as np
import re

def preprocess_credit_data(df):
    # --- Name ---
    df = df[df["name"].notna() & (df["name"].astype(str).str.strip() != "")]
    df.loc[:, "name"] = df["name"].astype(str).str.replace(r"[^\w\s]", "", regex=True).str.strip()

    # --- Age ---
    df.loc[:, "age"] = df["age"].astype(str).str.extract(r"(\d+)", expand=False)
    df.loc[:, "age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]

    # --- Occupation ---
    df.loc[:, "occupation"] = df["occupation"].astype(str).replace(["_______", "nan"], np.nan)

    # --- Annual Income ---
    df.loc[:, "annual_income"] = df["annual_income"].astype(str).str.extract(r"([\d.]+)", expand=False)
    df.loc[:, "annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce").round(2)
    df = df[df["annual_income"].notna() & (df["annual_income"] <= 5e6)]

    # --- Monthly Inhand Salary ---
    df.loc[:, "monthly_inhand_salary"] = pd.to_numeric(df["monthly_inhand_salary"], errors="coerce").round(2)
    df = df[df["monthly_inhand_salary"].notna() & (df["monthly_inhand_salary"] <= 1e6)]

    # --- Num Bank Accounts ---
    df.loc[:, "num_bank_accounts"] = pd.to_numeric(df["num_bank_accounts"], errors="coerce")
    df = df[(df["num_bank_accounts"] >= 0) & (df["num_bank_accounts"] <= 50)]

    # --- Num Credit Card ---
    df.loc[:, "num_credit_card"] = df["num_credit_card"].astype(str).str.extract(r"(\d+)", expand=False)
    df.loc[:, "num_credit_card"] = pd.to_numeric(df["num_credit_card"], errors="coerce")
    df = df[(df["num_credit_card"] >= 0) & (df["num_credit_card"] <= 50)]

    # --- Interest Rate ---
    df.loc[:, "interest_rate"] = df["interest_rate"].astype(str).str.extract(r"(\d+)", expand=False)
    df.loc[:, "interest_rate"] = pd.to_numeric(df["interest_rate"], errors="coerce")
    df = df[(df["interest_rate"] >= 0) & (df["interest_rate"] <= 100)]

    # --- Num of Loan ---
    df.loc[:, "num_of_loan"] = df["num_of_loan"].astype(str).str.extract(r"(\d+)", expand=False)
    df.loc[:, "num_of_loan"] = pd.to_numeric(df["num_of_loan"], errors="coerce")
    df = df[(df["num_of_loan"] >= 0) & (df["num_of_loan"] <= 50)]

    # --- Delay from due date ---
    df.loc[:, "delay_from_due_date"] = pd.to_numeric(df["delay_from_due_date"], errors="coerce").astype("Int64")
    df = df[(df["delay_from_due_date"] >= 0) & (df["delay_from_due_date"] <= 100)]

    # --- Num of Delayed Payment ---
    df.loc[:, "num_of_delayed_payment"] = df["num_of_delayed_payment"].astype(str).str.extract(r"(\d+)", expand=False)
    df.loc[:, "num_of_delayed_payment"] = pd.to_numeric(df["num_of_delayed_payment"], errors="coerce").astype("Int64")
    df = df[(df["num_of_delayed_payment"] >= 0) & (df["num_of_delayed_payment"] <= 100)]

    # --- Changed Credit Limit ---
    if "changed_credit_limit" in df.columns:
        df = df.drop(columns=["changed_credit_limit"])

    # --- Num Credit Inquiries ---
    df.loc[:, "num_credit_inquiries"] = pd.to_numeric(df["num_credit_inquiries"], errors="coerce")
    df.loc[(df["num_credit_inquiries"] < 0) | (df["num_credit_inquiries"] > 100), "num_credit_inquiries"] = np.nan

    # --- Outstanding Debt ---
    df.loc[:, "outstanding_debt"] = df["outstanding_debt"].astype(str).str.extract(r"(-?[\d.]+)", expand=False)
    df.loc[:, "outstanding_debt"] = pd.to_numeric(df["outstanding_debt"], errors="coerce")

    # --- Feature Engineering: Credit Utilization Rate ---
    df["credit_utilization_rate"] = df.apply(
        lambda row: row["outstanding_debt"] / row["annual_income"]
        if pd.notnull(row["outstanding_debt"]) and pd.notnull(row["annual_income"]) and row["annual_income"] != 0
        else np.nan,
        axis=1
    )
    df["credit_utilization_rate"] = df["credit_utilization_rate"].replace([np.inf, -np.inf], np.nan).round(2)

    # --- Credit Utilization Ratio ---
    df.loc[:, "credit_utilization_ratio"] = pd.to_numeric(df["credit_utilization_ratio"], errors="coerce").round(2)

    # --- Credit Mix ---
    df.loc[:, "credit_mix"] = df["credit_mix"].astype(str).str.lower().str.strip()
    df.loc[:, "credit_mix"] = df["credit_mix"].replace({"nan": np.nan, "": np.nan})

    # --- Credit History Age → credit_history_months ---
    def extract_credit_months(text):
        if pd.isna(text):
            return np.nan
        text = str(text).lower()
        years = months = 0
        year_match = re.search(r"(\d+)\s*year", text)
        month_match = re.search(r"(\d+)\s*month", text)
        if year_match:
            years = int(year_match.group(1))
        if month_match:
            months = int(month_match.group(1))
        return years * 12 + months

    df.loc[:, "credit_history_age"] = df["credit_history_age"].astype(str)
    df.loc[:, "credit_history_months"] = df["credit_history_age"].apply(extract_credit_months).astype("Int64")

    # --- Payment of Min Amount ---
    df.loc[:, "payment_of_min_amount"] = df["payment_of_min_amount"].astype(str).str.lower().str.strip()
    df.loc[:, "payment_of_min_amount"] = df["payment_of_min_amount"].replace({"nm": np.nan, "": np.nan, "nan": np.nan})

    # --- Total EMI per Month ---
    df.loc[:, "total_emi_per_month"] = df["total_emi_per_month"].astype(str).str.extract(r"([\d.]+)", expand=False)
    df.loc[:, "total_emi_per_month"] = pd.to_numeric(df["total_emi_per_month"], errors="coerce").round(2)
    df = df[(df["total_emi_per_month"] >= 0) & (df["total_emi_per_month"] <= 5000)]

    # --- Amount Invested Monthly ---
    df.loc[:, "amount_invested_monthly"] = df["amount_invested_monthly"].astype(str).str.extract(r"([\d.]+)", expand=False)
    df.loc[:, "amount_invested_monthly"] = pd.to_numeric(df["amount_invested_monthly"], errors="coerce").round(2)
    df = df[(df["amount_invested_monthly"] >= 0) & (df["amount_invested_monthly"] <= 50000)]

    # --- Payment Behaviour ---
    df.loc[:, "payment_behaviour"] = df["payment_behaviour"].astype(str).str.lower().str.strip()
    df.loc[:, "payment_behaviour"] = df["payment_behaviour"].replace("!@9#%8", np.nan)

    # --- Monthly Balance ---
    df.loc[:, "monthly_balance"] = df["monthly_balance"].astype(str).str.extract(r"(-?[\d.]+)", expand=False)
    df.loc[:, "monthly_balance"] = pd.to_numeric(df["monthly_balance"], errors="coerce").round(2)
    df.loc[(df["monthly_balance"] < -1e6) | (df["monthly_balance"] > 1e6), "monthly_balance"] = np.nan

    # --- Credit Score ---
    if "credit_score" in df.columns:
        df.loc[:, "credit_score"] = df["credit_score"].astype(str).str.lower().str.strip()
        df.loc[:, "credit_score"] = df["credit_score"].replace({"nan": np.nan, "": np.nan})
        df = df[df["credit_score"].notna()]

    # --- Преобразуем только нужные поля в string ---
    columns_to_str = [
        "id", "customer_id", "month", "name", "ssn", "occupation",
        "type_of_loan", "credit_mix", "payment_of_min_amount",
        "payment_behaviour", "credit_score", "credit_history_age"
    ]
    
    for col in columns_to_str:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # --- Приведение числовых колонок к нужным типам ---
    int_columns = [
        "age", "num_bank_accounts", "num_credit_card", "interest_rate",
        "num_of_loan", "delay_from_due_date", "num_of_delayed_payment",
        "credit_history_months"
    ]
    float_columns = [
        "annual_income", "monthly_inhand_salary", "num_credit_inquiries",
        "outstanding_debt", "credit_utilization_ratio", "total_emi_per_month",
        "amount_invested_monthly", "monthly_balance", "credit_utilization_rate"
    ]
    
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    
    return df

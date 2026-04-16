import pandas as pd

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)

    return df
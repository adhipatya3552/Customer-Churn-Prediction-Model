import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Convert TotalCharges to numeric
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors='coerce')

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Drop customerID
    df.drop("CustomerID", axis=1, inplace=True)

    return df
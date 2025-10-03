# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(input_csv="data/loan_data.csv", output_csv="data/loan_data_processed.csv"):
    # Load dataset
    df = pd.read_csv(input_csv)
    print(f"📥 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Encode categorical column
    le = LabelEncoder()
    df['purpose'] = le.fit_transform(df['purpose'])

    # Scale numeric columns (excluding target)
    scaler = StandardScaler()
    numeric_cols = df.drop(columns=['not.fully.paid']).select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save processed dataset
    df.to_csv(output_csv, index=False)
    print(f"✅ Preprocessed data saved to {output_csv}")

    # Preview results
    print("\n🔎 Preview of processed data:")
    print(df.head())

    # Show class balance
    print("\n📊 Class distribution (not.fully.paid):")
    print(df['not.fully.paid'].value_counts(normalize=True))

if __name__ == "__main__":
    preprocess()

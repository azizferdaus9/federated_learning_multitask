import pandas as pd

# Load dataset
df = pd.read_csv("Dataset/parkinsons_updrs.csv")

# Basic information
print("✅ Dataset loaded successfully!\n")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# Missing values
print("\nMissing values per column:\n", df.isnull().sum())

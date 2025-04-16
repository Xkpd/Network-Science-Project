import pandas as pd
import os

# Create output folder if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

log = []  # Collect logs here

# Load the CSV
papers_df = pd.read_csv("./output/papers_cleaned.csv")
log.append(f"Initial shape: {papers_df.shape[0]} rows, {papers_df.shape[1]} columns")

# Step 1: Remove exact duplicate rows
duplicate_rows = papers_df.duplicated()
log.append(f"Total exact duplicate rows: {duplicate_rows.sum()}")
papers_df = papers_df.drop_duplicates()

# Step 2: Check and remove duplicates by key columns
for col in ['Title']:
    if col in papers_df.columns:
        count = papers_df.duplicated(subset=col, keep=False).sum()
        log.append(f"- Duplicate entries in \"{col}\": {count}")
        papers_df = papers_df.drop_duplicates(subset=col, keep='first')

# Step 3: Check for missing (NaN) values
na_counts = papers_df.isna().sum()
log.append("Missing (NaN) values per column:")
log.extend([f"  - {col}: {na_counts[col]}" for col in na_counts.index if na_counts[col] > 0])

# Step 4: Check for empty strings
empty_counts = (papers_df == '').sum()
log.append("Empty string entries per column:")
log.extend([f"  - {col}: {empty_counts[col]}" for col in empty_counts.index if empty_counts[col] > 0])

# Step 5: Check for any rows that are partially empty
partially_empty = papers_df.isna().any(axis=1) | (papers_df == '').any(axis=1)
log.append(f"Rows with any missing (NaN) or empty strings: {partially_empty.sum()}")

# Final shape
log.append(f"Final shape after cleaning: {papers_df.shape[0]} rows")

# Save cleaned file
cleaned_path = os.path.join(output_dir, "papers_cleaned_final.csv")
papers_df.to_csv(cleaned_path, index=False)
log.append(f"Cleaned data saved to '{cleaned_path}'")

# Save log
log_path = os.path.join(output_dir, "cleaning_log.txt")
with open(log_path, "w", encoding='utf-8') as f:
    for line in log:
        f.write(line + "\n")

print(f"Cleaning complete. Outputs written to '{output_dir}' folder.")


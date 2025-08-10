import pandas as pd

# Read your CSV files
df = pd.read_csv("horeca_pomorskie.csv", encoding="utf-8")
mapping = pd.read_csv("PL_POSTAL_CODE_CS.csv", dtype=str)  # replace with actual filename

# Clean and standardize postal codes
df["postal_code"] = (
    df["Grantobiorca Kod pocztowy"].astype(str).str.strip().str.replace("-", "")
)
mapping["postal_code"] = mapping["postal_code"].astype(str).str.strip()

# Merge datasets on full postal code
df = df.merge(
    mapping[["postal_code", "counties", "state"]],
    how="left",
    left_on="postal_code",
    right_on="postal_code"
)

# Drop helper postal_code column
df = df.drop(columns=["postal_code", "state"])

# Rename counties → Powiat
df = df.rename(columns={"counties": "Powiat"})

# Save enriched CSV
df.to_csv("horeca_pomorskie_with_powiat.csv", index=False, encoding="utf-8-sig")

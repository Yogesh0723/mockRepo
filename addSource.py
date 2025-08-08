import pandas as pd
import json
from pathlib import Path

# ========= CONFIG =========
INPUT_CSV = "./data/input_addresses.csv"  # <-- replace with your file path
OUTPUT_CSV = "./data/uk_address_bio.csv"
OUTPUT_JSON = "./data/uk_address_bio.json"
# ==========================

def clean_address_no_postcode(address):
    """Remove postcode from address string."""
    tokens = address.strip().strip(',').split(',')
    # Remove postcode if last part is alphanumeric (e.g., EX4 2JB)
    if tokens and tokens[-1].strip().replace(" ", "").isalnum():
        tokens = tokens[:-1]
    return ", ".join([t.strip() for t in tokens])  


def tokenize_address(address):
    address = address.strip().strip(',')
    tokens = address.split(',')
    # Remove postcode (last token if alphanumeric like 'EX4 2JB')
    if tokens and tokens[-1].strip().replace(" ", "").isalnum():
        tokens = tokens[:-1]
    return " ".join(tokens).split()

def bio_tag_address(row):
    tokens = tokenize_address(row['SINGLE_LINE_ADDRESS'])
    tags = ['O'] * len(tokens)
    used = set()

    def match_and_tag(value, label_prefix):
        if pd.isna(value) or str(value).strip() == "":
            return
        comp_tokens = str(value).strip().split()
        n = len(comp_tokens)
        for i in range(len(tokens) - n + 1):
            if any(idx in used for idx in range(i, i + n)):
                continue
            if tokens[i:i+n] == comp_tokens:
                tags[i] = f'B-{label_prefix}'
                for j in range(1, n):
                    tags[i+j] = f'I-{label_prefix}'
                used.update(range(i, i+n))
                break

    # Priority: If SUB_BUILDING exists → BUILDING_NUMBER = HOUSE_NUM
    if not pd.isna(row['SUB_BUILDING']) and str(row['SUB_BUILDING']).strip():
        match_and_tag(row['SUB_BUILDING'], 'FLAT')
        match_and_tag(row['BUILDING_NUMBER'], 'HOUSE_NUM')
    else:
        match_and_tag(row['BUILDING_NUMBER'], 'HOUSE_NUM')

    match_and_tag(row['BUILDING_NAME'], 'HOUSE_NAME')
    match_and_tag(row['STREET_NAME'], 'STREET')

    return tokens, tags

def generate_labeled_dataset(df):
    rows = []
    json_data = []

    for _, row in df.iterrows():
        tokens, tags = bio_tag_address(row)
        for i, (tok, tag) in enumerate(zip(tokens, tags)):
            rows.append({
                'address': clean_address_no_postcode(row['SINGLE_LINE_ADDRESS']),
                'token_id': i,
                'token': tok,
                'bio_tag': tag
            })

        json_data.append({
            'address': clean_address_no_postcode(row['SINGLE_LINE_ADDRESS']),
            'tokens': tokens,
            'bio_tags': tags
        })

    return pd.DataFrame(rows), json_data

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    csv_df, json_output = generate_labeled_dataset(df)

    csv_df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2)

    print(f"✅ Token-level CSV saved to {Path(OUTPUT_CSV).resolve()}")
    print(f"✅ JSON saved to {Path(OUTPUT_JSON).resolve()}")

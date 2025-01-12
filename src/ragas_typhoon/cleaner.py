from pathlib import Path
import json

def clean(INPUT_DIR, OUTPUT_DIR, keys):
    with open(INPUT_DIR, 'r') as file:
        data = json.load(file)
    
    cleaned_data = []

    for item in data:
        cleaned_item = {key: value for key, value in item.items() if key in keys}
        cleaned_data.append(cleaned_item)
        
    with open(OUTPUT_DIR, 'w') as file:
        json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    keys = ["type", "question", "ground_truth"]
    
    DIR = Path(__file__).parent / "data" / "input" / "general.json"
    DIRC = Path(__file__).parent / "data" / "input" / "general_cleaned.json"
    clean(DIR, DIRC, keys)
    
    DIR = Path(__file__).parent / "data" / "input" / "finance.json"
    DIRC = Path(__file__).parent / "data" / "input" / "finance_cleaned.json"
    clean(DIR, DIRC, keys)
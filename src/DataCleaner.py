import os
import re
import json
import random
import glob
import pandas as pd
import logging
from typing import List, Dict

os.chdir(os.path.dirname(__file__))

def read_data(fpath: str) -> List[Dict]:
    files = glob.glob(f"{fpath}/**")
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df.drop_duplicates()

    df["question"] = df["question"].str.replace("\"", "").str.replace("\'", "")
    df["answer"] = df["answer"].str.replace("\"", "").str.replace("\'", "")

    data = list(df.apply(lambda x: x.to_dict(), axis=1))

    return data

def is_valid(data: Dict) -> bool:
    is_how_much = False if re.search("how much", data["question"], re.IGNORECASE) else True

    return all([
        is_how_much,
    ])

def validate_entries(entries: List[Dict]) -> List[Dict]:
    return [entry for entry in entries if is_valid(entry)]

def transform_to_chat(data: Dict) -> Dict:
    system_message = (
        "Imagine you are an experienced and professional auditor with extensive knowledge in your field."
        "When responding, it is crucial that the information you provide is both accurate and precise."
        "Please ensure your replies are concise, professional, and exude confidence."
    )

    user_message = data["question"]
    assistant_message = data['answer']
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    return {"messages": messages}

def transform_entries(entries: List[Dict]) -> List[Dict]:
    return [transform_to_chat(entry) for entry in entries]

def remove_long_entries(entries: List[Dict]) -> List[Dict]:
    return [entry for entry in entries if sum(len(message['content']) for message in entry['messages']) <= 2048]

def transform_all(format):

    entries_folder = '../data/processed_data'
    data = read_data(entries_folder)
    data = validate_entries(data)
    data = transform_entries(data)
    data = remove_long_entries(data)

    n = len(data)

    os.makedirs(f'../dataset/{format}/', exist_ok=True)

    with open(f'../dataset/{format}/all.jsonl', 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    test_data = data[int(n * 0.9):]
    data = data[:int(n * 0.9)]

    random.shuffle(test_data)
    random.shuffle(data)

    train_data = data[:int(n * 0.8)]
    val_data = data[int(n * 0.8):]

    with open(f'../dataset/{format}/train.jsonl', 'w', encoding='utf-8') as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open(f'../dataset/{format}/valid.jsonl', 'w', encoding='utf-8') as f:
        for d in val_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open(f'../dataset/{format}/test.jsonl', 'w', encoding='utf-8') as f:
        for d in test_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    logger.info(f"Transformed {n} entries and saved to ./dataset/{format}/")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='../log/DataCleaner.log', encoding='utf-8', level=logging.DEBUG, filemode="w")
    transform_all("chat")
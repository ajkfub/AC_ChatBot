import os
import re
import json
import random
import glob
import pandas as pd
import logging
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dataset_directory = os.path.join(parent_dir, "dataset")
data_directory = os.path.join(parent_dir, "data/raw_data")

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(__file__))


def read_data(fpath: str) -> List[Dict]:
    """
    Read CSV files from a specified directory and return a list of dictionaries.

    Args:
        fpath (str): The path to the directory containing CSV files.

    Returns:
        List[Dict]: A list of dictionaries representing the data.
    """
    files = glob.glob(f"{fpath}/**")  # Get all files in the specified directory
    df = pd.concat([pd.read_csv(f) for f in files])  # Concatenate all CSV files into a single DataFrame
    df = df.drop_duplicates()  # Remove duplicate entries

    # Clean the 'question' and 'answer' columns by removing quotes
    df["question"] = df["question"].str.replace("\"", "").str.replace("\'", "")
    df["answer"] = df["answer"].str.replace("\"", "").str.replace("\'", "")

    # Convert DataFrame rows to a list of dictionaries
    data = list(df.apply(lambda x: x.to_dict(), axis=1))

    return data


def is_valid(data: Dict) -> bool:
    """
    Validate the entry based on specific criteria.

    Args:
        data (Dict): A dictionary representing a single data entry.

    Returns:
        bool: True if the entry is valid, False otherwise.
    """
    is_how_much = False if re.search("how much", data["question"], re.IGNORECASE) else True

    return all([
        is_how_much,
    ])


def validate_entries(entries: List[Dict]) -> List[Dict]:
    """
    Validate a list of entries.

    Args:
        entries (List[Dict]): A list of dictionaries representing data entries.

    Returns:
        List[Dict]: A list of valid entries.
    """
    return [entry for entry in entries if is_valid(entry)]


def transform_to_chat(data: Dict) -> Dict:
    """
    Transform a data entry into a chat message format.

    Args:
        data (Dict): A dictionary representing a single data entry.

    Returns:
        Dict: A dictionary containing the chat message format.
    """
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
    """
    Transform a list of entries into a chat message format.

    Args:
        entries (List[Dict]): A list of dictionaries representing data entries.

    Returns:
        List[Dict]: A list of transformed entries in chat format.
    """
    return [transform_to_chat(entry) for entry in entries]


def remove_long_entries(entries: List[Dict]) -> List[Dict]:
    """
    Remove entries that exceed a certain length limit.

    Args:
        entries (List[Dict]): A list of entries in chat format.

    Returns:
        List[Dict]: A list of entries that are within the length limit.
    """
    return [entry for entry in entries if sum(len(message['content']) for message in entry['messages']) <= 2048]


def transform_all(format: str) -> None:
    """
    Perform the full transformation process on the dataset.

    Args:
        format (str): The format to save the transformed data.
    """
    entries_folder = data_directory
    data = read_data(data_directory)  # Read data from CSV files
    data = validate_entries(data)  # Validate the entries
    data = transform_entries(data)  # Transform the entries to chat format
    data = remove_long_entries(data)  # Remove long entries

    n = len(data)

    # Create output directory if it doesn't exist
    os.makedirs(f'{dataset_directory}/{format}/', exist_ok=True)

    # Save all transformed entries to a JSONL file
    with open(f'{dataset_directory}/{format}/all.jsonl', 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # Split the data into training, validation, and testing sets
    test_data = data[int(n * 0.9):]
    data = data[:int(n * 0.9)]

    random.shuffle(test_data)  # Shuffle the test data
    random.shuffle(data)  # Shuffle the remaining data

    train_data = data[:int(n * 0.8)]  # 80% for training
    val_data = data[int(n * 0.8):]  # 20% for validation

    # Save the training, validation, and test sets to JSONL files
    with open(f'{dataset_directory}/{format}/train.jsonl', 'w', encoding='utf-8') as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open(f'{dataset_directory}/{format}/valid.jsonl', 'w', encoding='utf-8') as f:
        for d in val_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open(f'{dataset_directory}/{format}/test.jsonl', 'w', encoding='utf-8') as f:
        for d in test_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    logger.info(f"Transformed {n} entries and saved to ./dataset/{format}/")


if __name__ == "__main__":
    # Initialize the logger for logging information
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='/log/DataCleaner.log', encoding='utf-8', level=logging.DEBUG, filemode="w")

    # Execute the transformation process
    transform_all("chat")
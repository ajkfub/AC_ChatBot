# !git clone https://github.com/amontgomerie/question_generator

import json
import sys
import logging
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from question_generator.questiongenerator import QuestionGenerator
import pandas as pd
from tqdm import tqdm
import glob
from typing import List


class DataGenerator:
    """
    A class to generate questions from text data using the QuestionGenerator.

    Attributes:
        qg (QuestionGenerator): An instance of the QuestionGenerator class.
    """

    def __init__(self) -> None:
        """Initialize the DataGenerator with a QuestionGenerator instance."""
        self.qg = QuestionGenerator()

    def _get_text_data(self, filepath: str, name: str) -> None:
        """
        Load text data from a JSON file.

        Args:
            filepath (str): The path to the JSON file.
            name (str): The name of the dataset for later use.
        """
        self.filepath = filepath
        self.name = name

        with open(self.filepath) as fp:
            data = json.load(fp)

        self.text_data = data

    def _generate_question(self, text: str, num_questions: int = 1) -> List[dict]:
        """
        Generate questions from a given text.

        Args:
            text (str): The text from which to generate questions.
            num_questions (int): The number of questions to generate.

        Returns:
            List[dict]: A list of generated questions.
        """
        qa_list = self.qg.generate(
            text, num_questions=num_questions, answer_style="sentences"
        )

        logging.info(f"{len(qa_list)} questions are generated")

        return qa_list

    def generate_data(self) -> None:
        """Generate questions for the loaded text data and save them to a CSV file."""
        text_list = [v for _, v in self.text_data.items()]

        result = []

        logging.info(f"Start generating {len(text_list)} questions")

        for text in tqdm(text_list[:5]):
            question_list = self._generate_question(text)
            result = result + question_list

        df = pd.DataFrame(result)
        df.to_csv(f"{parent_dir}/data/processed_data/{self.name}.csv", index=False)

    def run(self, filepath: str, name: str) -> None:
        """
        Execute the data generation process.

        Args:
            filepath (str): The path to the JSON file containing text data.
            name (str): The name of the dataset for saving the output.
        """
        self._get_text_data(filepath, name)
        self.generate_data()


if __name__ == "__main__":
    # Initialize the logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename="log/generator.log",
        encoding="utf-8",
        level=logging.DEBUG,
        filemode="w",
    )

    # Create an instance of DataGenerator
    generator = DataGenerator()

    # Get all JSON files from the raw data directory
    directory = os.path.join(parent_dir, "data/raw_data")
    files = os.listdir(directory)
    files = [f for f in files if not f.startswith(".")]

    for file in files:
        fname = file.replace(".json", "")
        generator.run(filepath=os.path.join(directory, file), name=fname)
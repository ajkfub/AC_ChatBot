import json
import sys
import logging
import torch
# sys.path.append("/content/question_generator")
# sys.path.append("/content/raw_data")
from questiongenerator import QuestionGenerator
import pandas as pd
from tqdm import tqdm


class DataGenerator:

    def __init__(self):
        self.qg = QuestionGenerator()

    def _get_text_data(self, filepath, name):
        self.filepath = filepath
        self.name = name

        with open(self.filepath) as fp:
            data = json.load(fp)

        self.text_data = data

    def _generate_question(self, text: str, num_questions=1) -> list:
        qa_list = self.qg.generate(
            text,
            num_questions=num_questions,
            answer_style="sentences"
        )

        logging.info(f"{len(qa_list)} questions are generated")

        return qa_list

    def generate_data(self):
        text_list = [v for _, v in self.text_data.items()]

        result = []

        logging.info(f"Start generating {len(text_list)} questions")

        for text in tqdm(text_list):
            question_list = self._generate_question(text)
            result.append(question_list)

        df = pd.DataFrame(result)
        df.to_csv(f"data/processed_data/{self.name}.csv", index=False)

    def run(self, filepath, name):
        self._get_text_data(filepath, name)
        self.generate_data()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='log/gerenator.log', encoding='utf-8', level=logging.DEBUG, filemode="w")

    generator = DataGenerator()
    generator.run(filepath="data/raw_data/accountingcoach.json", name="accountingcoach")
    generator.run(filepath="data/raw_data/ready_ratio.json", name="ready_ratio")
# !pip install llama-cpp-python
import os
from Prediction import find_optimal_k, suggest_similar_questions
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from llama_cpp import Llama
from DataVisualizer import DataVisualizer
import pandas as pd
import argparse
import seaborn as sns
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_directory = os.path.join(parent_dir, "data/enquiries_data")

# Set the style for seaborn visualizations
sns.set(font="Times")
sns.set_style("ticks")
sns.set_context("poster", font_scale=0.75, rc={"grid.linewidth": 0.75})


class Response:
    """
    A class to generate text responses using a language model and visualize financial data.

    Attributes:
        _repo_id (str): The repository ID for the pretrained model.
        _filename (str): The filename of the pretrained model.
        model (Llama): The language model instance.
    """

    def __init__(self) -> None:
        """Initialize the Response class and load the pretrained Llama model."""
        self._repo_id = "SamHunghaha/AC_ChatBot"
        self._filename = "unsloth.F16.gguf"

        self.model = Llama.from_pretrained(
            repo_id=self._repo_id, filename=self._filename
        )
        self._optimal_k = self._get_optimal_k()

    def _get_optimal_k(self) -> int:

        nlp = spacy.load("en_core_web_sm")

        # Step 1: Text Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')

        data_path = os.path.join(data_directory, "enquiries.csv")
        self.enquiries = list(pd.read_csv(data_path)["enquiries"])

        X = vectorizer.fit_transform(self.enquiries)
        optimal_k = find_optimal_k(X,1450, plot=False)

        return optimal_k

    def generate_text_from_prompt(
        self,
        user_prompt: str,
        max_tokens: int = 40,
        temperature: float = 0.3,
        top_p: float = 0.1,
        echo: bool = True,
    ) -> str:
        """
        Generate text response from a user prompt using the Llama model.

        Args:
            user_prompt (str): The prompt provided by the user.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 100.
            temperature (float): The temperature for sampling. Defaults to 0.3.
            top_p (float): The cumulative probability for nucleus sampling. Defaults to 0.1.
            echo (bool): Whether to echo the prompt in the output. Defaults to True.

        Returns:
            str: The generated text response.
        """
        model_output = self.model(
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )

        result = model_output["choices"][0]["text"].strip()
        result = result.split("\n")[1]
        print(f"Prompt: {user_prompt}")
        print(f"Response: {result}")

        suggestions = suggest_similar_questions(user_prompt, self.enquiries, self._optimal_k)

        print(f"Suggested questions for '{user_prompt}':")
        for question in suggestions:
            print(f"- {question}")

        return result

    def visualize_data(self, ticker: str, item: str, freq: str) -> None:
        """
        Visualize financial data for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.
            item (str): The financial item to visualize (e.g., "totalAssets").
            freq (str): The frequency of the data ("A" for Annual, "Q" for Quarterly).
        """
        self.data_visualizer = DataVisualizer(ticker)
        self.data_visualizer.display(item, freq)


if __name__ == "__main__":
    response = Response()

    parser = argparse.ArgumentParser(description="Accounting Chat Bot")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["prompt", "data"],
        help='Mode of operation: "prompt" to generate text or "data" to visualize financial data.',
    )
    parser.add_argument(
        "--prompt",
        required=False,
        help="The prompt to generate a response from the model (required in prompt mode).",
    )
    parser.add_argument(
        "--stock_code",
        required=False,
        help='Stock ticker symbol (e.g., "AAPL") - required in data mode.',
    )
    parser.add_argument(
        "--item",
        required=False,
        help='Financial item to visualize (e.g., "totalAssets") - required in data mode.',
    )
    parser.add_argument(
        "--freq",
        required=False,
        choices=["A", "Q"],
        help='Frequency of the data: "A" for Annual, "Q" for Quarterly - required in data mode.',
    )

    # Parse command line arguments
    args = parser.parse_args()
    mode = args.mode
    stock_code: Optional[str] = args.stock_code
    item: Optional[str] = args.item
    freq: Optional[str] = args.freq

    if mode == "prompt":
        if args.prompt is None:
            raise Exception("Argument --prompt is required in Prompt mode.")
        else:
            prompt = args.prompt
            result = response.generate_text_from_prompt(prompt, max_tokens=30)
            print(result)
    elif mode == "data":
        if stock_code is None or item is None or freq is None:
            raise Exception(
                "Arguments --stock_code, --item, and --freq are all required in Data mode."
            )
        else:
            response.visualize_data(stock_code, item, freq)
    else:
        raise Exception(
            "Incorrect mode input -- only 'prompt' or 'data' mode is accepted."
        )

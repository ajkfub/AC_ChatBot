# Import required packages
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize

#nltk.download("punkt")
#nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from nltk import ngrams
import os
from typing import List, Dict

wordnet = WordNetLemmatizer()

# Create the transform
vectorizer = TfidfVectorizer(ngram_range=(1, 3))


def cleansing(text: str, wordnet: WordNetLemmatizer) -> str:
    """Clean the input text by removing unwanted characters and normalizing it.

    Args:
        text (str): The input text to be cleaned.
        wordnet (WordNetLemmatizer): An instance of the WordNet lemmatizer.

    Returns:
        str: The cleaned and processed text.
    """
    # Remove space after "IFRS" or "IAS" if the next string is a number
    text_ifrs = re.sub(r"(IFRS)\s+(\d+)", r"\1\2", text)
    text_ias = re.sub(r"(IAS)\s+(\d+)", r"\1\2", text_ifrs)
    # Remove square brackets and text between them
    text_no_square = re.sub(r"\[[^\]]*\]", "", text_ias)
    # Remove round brackets and text between them
    text_no_round = re.sub(r"\([^\)]*\)", "", text_no_square)
    # Convert to lower case
    text_lower_case = text_no_round.lower()
    # Remove punctuations, whitespace, and numbers
    text_no_punctuation = re.sub(r"[^\w\s]", "", text_lower_case)
    text_no_wspace = text_no_punctuation.strip()
    text_no_num = re.sub(r"\s\d+\s", "", text_no_wspace)
    text_non_single = re.sub(r"(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)", "", text_no_num)

    # Word tokenize and clean stopwords with lemmatization
    text_tokenize = word_tokenize(text_non_single)
    text_no_stop = [
        wordnet.lemmatize(token) for token in text_tokenize if token not in (stopwords.words("english") + ["shall"])
    ]

    # Remove the word 'paragraph'
    text_processed = re.sub(r"\b{}\b".format("paragraph"), "", " ".join(text_no_stop))

    return text_processed


def munti_gram(cleaned_txt: List[str], gram: int) -> List[str]:
    """Generate bag of words for n-gram analysis.

    Args:
        cleaned_txt (List[str]): The cleaned text data.
        gram (int): The n-gram size.

    Returns:
        List[str]: A list of n-grams generated from the cleaned text.
    """
    mulgram_lst = []

    for sentence in cleaned_txt:
        mulgram = list(ngrams(sentence.split(), gram))
        mulgram_lst_temp = [" ".join(item) for item in mulgram]
        mulgram_lst.append(mulgram_lst_temp)

    return [item for sublist in mulgram_lst for item in sublist]


def noun_phrase(mulgram_lst_ttl_pos: List[str]) -> nltk.Tree:
    """Identify noun phrases based on Part of Speech (PoS) tagging.

    Args:
        mulgram_lst_ttl_pos (List[str]): List of tokenized words to analyze.

    Returns:
        nltk.Tree: A tree structure of identified noun phrases.
    """
    tagged_words = nltk.pos_tag(mulgram_lst_ttl_pos)

    # Define the grammar for noun phrases
    grammar = "NP:{<DT>?<JJ>*<NN>}"
    parser = nltk.RegexpParser(grammar)

    return parser.parse(tagged_words)


def vectorize(cleaned_txt: List[str]) -> List[List[float]]:
    """Vectorizes the cleaned text using TF-IDF.

    Args:
        cleaned_txt (List[str]): The cleaned text data.

    Returns:
        List[List[float]]: A list of vectorized representations for each cleaned text.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    vectorizer.fit(cleaned_txt)

    return [(vectorizer.transform([x])).toarray() for x in cleaned_txt]


if __name__ == "__main__":
    # Set the directory for processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    processed_data_dir = os.path.join(parent_dir, "data/processed_data")
    cleaned_data_dir = os.path.join(parent_dir, "data/cleaned_data")

    files = [f for f in os.listdir(processed_data_dir) if not f.startswith(".")]

    # Generate the dictionary of dataframes
    file_dict: Dict[str, pd.DataFrame] = {
        file.replace(".csv", ""): pd.read_csv(os.path.join(processed_data_dir, file))
        for file in files
    }

    # Extract questions and answers for further preparation
    for name, table in file_dict.items():

        question_lst = list(table["question"])
        answer_lst = list(table["answer"])
        res_file_cleanse = pd.DataFrame()
        res_file_vectorized = pd.DataFrame()

        qa_dict = {"question": question_lst, "answer": answer_lst}

        # Prepare data for further analysis by columns
        for record, details in qa_dict.items():
            # Cleanse data
            cleaned_record = [cleansing(str(text), wordnet) for text in details]
            res_file_cleanse[record] = cleaned_record

            # Perform frequency analysis
            monogram_lst_ttl = munti_gram(cleaned_record, 1)
            bigram_lst_ttl = munti_gram(cleaned_record, 2)
            trigram_lst_ttl = munti_gram(cleaned_record, 3)

            dist_dict = {
                "monogram": FreqDist(monogram_lst_ttl),
                "bigram": FreqDist(bigram_lst_ttl),
                "trigram": FreqDist(trigram_lst_ttl),
            }
            for key, val in dist_dict.items():
                print(val.most_common(5))
                val.plot(20)

            # Vectorize string
            vec_lst = [str(list(x[0])) for x in vectorize(cleaned_record)]
            res_file_vectorized[record] = vec_lst

        # Save cleansed and vectorized data as CSV files
        res_file_cleanse.to_csv(f"{cleaned_data_dir}/{name}_cleansed.csv")
        json_string = res_file_vectorized.to_json()
        with open(f"{cleaned_data_dir}/{name}_vectorized.txt", "w") as f:
            f.write(str(json_string))

        # Perform Part of Speech (PoS) tagging
        noun_phrases = noun_phrase(monogram_lst_ttl)

        # Save noun phrases with PoS tagging as a text file
        with open(f"{cleaned_data_dir}/{name}_pos.txt", "w") as f:
            f.write(str(noun_phrases))

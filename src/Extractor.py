import json
import logging
import re
import unicodedata
import itertools
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt
from tqdm import tqdm
import pandas as pd

class Extractor:
    '''
    A class to extract data from Accounting Coach and Ready Ratios websites.

    It retrieves URLs from the Accounting Coach blog and extracts question-answer
    data, as well as extracts content from the Ready Ratios website.
    '''

    def __init__(self) -> None:

        '''
        Initializes the Extractor with base URLs and retrieves all relevant URLs.
        '''
        self.accounting_coach_url: str = "https://www.accountingcoach.com"
        self.ready_ratio_url: str = "https://www.readyratios.com"
        self.njobs: int = 5
        self.all_ac_urls = self.__get_all_ac_urls()
        self.all_rr_urls = self.__get_all_rr_urls()

    @retry(stop=stop_after_attempt(3))
    def __get_all_ac_urls(self) -> list[str]:
        """
        Retrieves all blog URLs from the Accounting Coach website.

        Returns:
            list: A list of blog URLs found on the Accounting Coach website.
        """
        response = requests.get(f"{self.accounting_coach_url}/blog")
        soup = BeautifulSoup(response.content, 'html.parser')
        topics = soup.find("div", {"id": "archive-topics"})
        topics = topics.find_all("li", {"class": "all-topics__topics__topic sorted"})

        logging.info(f"[Task: Account Coach]# of topics available: {len(topics)}")

        with ThreadPoolExecutor(max_workers=self.njobs) as executor:
            results = executor.map(self._get_url_from_topic, topics)

        all_urls = [res for res in results]
        all_urls = list(itertools.chain(*all_urls))

        logging.info(f"[Task: Account Coach]# of urls available: {len(all_urls)}")

        return all_urls

    def _get_url_from_topic(self, topic: str) -> list[str]:
        '''
        Extracts all blog URLs from a given topic.

        Args:
            topic (BeautifulSoup object): A BeautifulSoup object representing a blog topic.

        Returns:
            list: A list of blog URLs associated with the given topic.
        '''
        object = topic.find("a", href=True)
        href = object["href"]
        title = object["title"]

        response = requests.get(f"{self.accounting_coach_url}{href}")
        soup = BeautifulSoup(response.content, 'html.parser')

        urls = [
            i["href"]
            for i in soup.find_all("a", href=True)
            if re.match(f"{self.accounting_coach_url}/blog/.*", i["href"])
        ]

        return urls

    def _get_qna_data(self, url: str) -> dict[str, str]:
        """
        Extracts question and answer data from a specific blog URL.

        Args:
            url (str): The URL of the blog to extract data from.

        Returns:
            dict: A dictionary containing headers as keys and corresponding paragraphs as values.
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        soup = soup.find_all("div", {"class": "col-12 col-md-8"})[1]

        qna_data = self._get_header_and_paragraph(soup)

        return qna_data

    def _get_header_and_paragraph(self, soup: BeautifulSoup) -> dict[str, str]:
        """
        Retrieves headers and their associated paragraphs from a BeautifulSoup object.

        Args:
            soup (BeautifulSoup object): A BeautifulSoup object containing the content to parse.

        Returns:
            dict: A dictionary mapping headers to lists of associated paragraphs.
        """
        data = {}
        all_headers = soup.find_all("h2")
        for e in all_headers:
            data[e.text] = []
            for s in e.find_next_siblings():
                if s.name == "p":
                    data[e.text].append(s.text)
                else:
                    break

        return data

    @retry(stop=stop_after_attempt(3))
    def extract_ac_data(self) -> dict[str, str]:
        """
        Extracts question-answer data from all Accounting Coach blog URLs and saves it to a JSON file.

        Returns:
            dict: A dictionary containing all combined question-answer data from the blog.
        """
        with ThreadPoolExecutor(max_workers=self.njobs) as executor:
            results = executor.map(self._get_qna_data, self.all_ac_urls)

        data = [i for i in results]
        all_qna_data = reduce(lambda a, b: dict(a, **b), data)
        all_qna_data = {
            k: " ".join(v) for k, v in all_qna_data.items() if len(" ".join(v)) > 0
        }

        logging.info(f"[Task: Accounting Coach] Extracted {len(all_qna_data)} Question and Answers data")

        with open("data/raw_data/accountingcoach.json", "w") as fp:
            json.dump(all_qna_data, fp)

        return all_qna_data

    # ================================================
    # Function for ready ratios website scrapping
    # ================================================

    def __get_all_rr_urls(self) -> list[str]:
        """
        Retrieves all reference URLs from the Ready Ratios website.

        Returns:
            list: A list of reference URLs found on the Ready Ratios website.
        """
        response = requests.get(self.ready_ratio_url + "/reference")
        soup = BeautifulSoup(response.content, 'html.parser')

        urls = [
            i["href"]
            for i in soup.find_all("a", href=True)
            if re.match("/reference/.*/\Z", i["href"])
        ]

        logging.info(f"[Task: Ready Ratio]# of urls available: {len(urls)}")

        return urls

    def _get_rr_website_content(self, href):
        """
        Extracts content from a specific Ready Ratios reference URL.

        Args:
            href (str): The relative URL of the reference to extract content from.

        Returns:
            dict: A dictionary containing the title and text extracted from the reference.
        """
        data = {}

        url = self.ready_ratio_url + href
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "lxml")

        items = soup.find_all("div", {"class", "item"})

        for item in items:
            title = item.find_all("h2")[0].text
            text = item.find_all("p")[0].text
            text = unicodedata.normalize("NFKD", text)

            data[title] = text

        return data

    def extract_rr_data(self) -> dict[str, str]:
        """
        Extracts content from all Ready Ratios reference URLs and saves it to a JSON file.

        Returns:
            dict: A dictionary containing all combined content from the references.
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(self._get_rr_website_content, self.all_rr_urls)

        data = [i for i in results]
        all_qna_data = reduce(lambda a, b: dict(a, **b), data)

        logging.info(f"[Task: Ready Ratio] Extracted {len(all_qna_data)} Question and Answers data")

        with open("data/raw_data/ready_ratio.json", "w") as fp:
            json.dump(all_qna_data, fp)

        return all_qna_data

    def run(self):
        """
        Executes the extraction process for both Accounting Coach and Ready Ratios.

        This method orchestrates the extraction of data from both websites.
        """
        self.extract_ac_data()
        self.extract_rr_data()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='log/extractor.log', encoding='utf-8', filemode='w', level=logging.DEBUG)

    extractor = Extractor()
    extractor.run()
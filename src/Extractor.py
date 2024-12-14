from abc import ABC, abstractmethod
import requests
import re
from bs4 import BeautifulSoup
import json
import unicodedata
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from typing import Dict, List
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_directory = os.path.join(parent_dir, "data/raw_data")


class WebScrapper(ABC):
    """
    Abstract base class for web scrapers.

    Attributes:
        index_page (str): The URL of the main page to scrape.
    """

    def __init__(self, index_page: str):
        self.index_page = index_page

    @abstractmethod
    def _get_all_urls(self) -> List[str]:
        """Retrieve all URLs from the main page."""
        pass

    @abstractmethod
    def _get_subpage_data(self, url: str) -> Dict[str, List[str]]:
        """Extract data from a subpage given its URL."""
        pass

    def export_data(self, data: List[Dict[str, str]]) -> None:
        """
        Export scraped data to a JSON file.

        Args:
            data (List[Dict[str, str]]): The data to export.
        """
        with open(
            f"{data_directory}/{self.__class__.__name__}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def run(self) -> None:
        """Execute the scraping process."""
        print(f"Scraping {self.__class__.__name__}...")
        urls = self._get_all_urls()

        data = {}
        for url in tqdm(urls):
            data.update(self._get_subpage_data(url))

        self.export_data(data)
        print(f"Exported {len(data)} items to {self.__class__.__name__}.json")


class IFRSScraper(WebScrapper):
    """A web scraper for extracting IFRS standards from the official IFRS website."""

    def __init__(
        self,
        index_page: str = "https://www.ifrs.org",
        username: str = "dilac63789@chainds.com",
        password: str = "Stat7008",
    ):
        super().__init__(index_page)
        self.username = username
        self.password = password
        self.chrome_options = Options()
        self.service = Service(ChromeDriverManager().install())
        self.driver = None

    def __enter__(self):
        """Initialize WebDriver and login."""
        self.driver = webdriver.Chrome(
            service=self.service, options=self.chrome_options
        )
        self._login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and ensure driver is closed."""
        if self.driver:
            self.driver.quit()

    def _login(self) -> None:
        """Log in to the IFRS website."""
        self.driver.get(f"{self.index_page}/login/?resource=/content/ifrs/home.html")
        cookie_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="cc-accept-cookies"]'))
        )
        cookie_button.click()

        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='email']"))
        ).send_keys(self.username)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='password']"))
        ).send_keys(self.password)
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        ).click()

        time.sleep(5)  # wait for browser to redirect

    def _get_all_urls(self) -> List[str]:
        """Get all IFRS standard URLs from the main page."""
        self.driver.get(f"{self.index_page}/issued-standards/list-of-standards/")
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#ifrs-cmp-standards-results")
            )
        )

        section = self.driver.find_element(
            By.XPATH, '//*[@id="ifrs-cmp-standards-results"]/div/div[4]'
        )
        soup = BeautifulSoup(section.get_attribute("outerHTML"), "html.parser")
        urls = [
            link["href"].replace(
                "/content/ifrs/home/issued-standards/",
                f"{self.index_page}/issued-standards/",
            )
            for link in soup.find_all("a", href=True)
        ]
        return urls

    def _get_subpage_data(self, url: str) -> Dict[str, str]:
        """Get content from an IFRS standard page."""
        self.driver.get(url)
        section_xpath = "/html/body/div/div/div[2]/div/div/div/div[3]/div/div/div/div/div/div/div/div/div/section"
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, section_xpath))
        )

        section = self.driver.find_element(By.XPATH, section_xpath)
        soup = BeautifulSoup(section.get_attribute("outerHTML"), "html.parser")
        ifrs_number = int(url.split("/")[-1].replace("ifrs", "").replace(".html", ""))
        table = soup.find("div", id=f"IFRS{ifrs_number:02d}_TI")

        data = {}
        for div in table.find_all("div", class_="body"):
            text = " ".join(div.text.split()).replace("\u2060", "")
            if len(text) > 0:
                data[f"IFRS{ifrs_number}_Item{len(data)+1}"] = text

        return data


class AccountingCoachScrapper(WebScrapper):
    """Scrape accountingcoach.com for Q&A data."""

    def __init__(self, index_page: str = "https://www.accountingcoach.com"):
        super().__init__(index_page)

    def _get_all_urls(self) -> List[str]:
        """Get all blog post URLs from the main page."""
        response = requests.get(f"{self.index_page}/blog", timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        topics = soup.find("div", {"id": "archive-topics"})
        topics = topics.find_all("li", {"class": "all-topics__topics__topic sorted"})

        all_urls = []
        for topic in tqdm(topics):
            link = topic.find("a", href=True)
            url = f"https://www.accountingcoach.com{link['href']}"

            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            urls = [
                a["href"]
                for a in soup.find_all("a", href=True)
                if re.match("https://www.accountingcoach.com/blog/.*", a["href"])
            ]
            all_urls.extend(urls)

        return all_urls

    def _get_subpage_data(self, url: str) -> Dict[str, str]:
        """Get Q&A data from a blog post URL by extracting headers and their associated paragraphs."""
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find_all("div", {"class": "col-12 col-md-8"})[1]

        data = {}
        for header in content.find_all("h2"):
            paragraphs = []
            for sibling in header.find_next_siblings():
                if sibling.name == "p":
                    paragraphs.append(sibling.text)
                else:
                    break
            data[header.text] = paragraphs

        data = {k: " ".join(v) for k, v in data.items() if len(" ".join(v)) > 0}
        return data


class ReadyRatiosScrapper(WebScrapper):
    """Scrape readyratios.com for Q&A data."""

    def __init__(self, index_page: str = "https://www.readyratios.com/reference/"):
        super().__init__(index_page)

    def _get_all_urls(self) -> List[str]:
        """Get all URLs from the main page."""
        response = requests.get(self.index_page, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        urls = [
            i["href"]
            for i in soup.find_all("a", href=True)
            if re.match("/reference/.*/\Z", i["href"])
        ]
        return [f"https://www.readyratios.com{url}" for url in urls]

    def _get_subpage_data(self, url: str) -> Dict[str, str]:
        """Get content from readyratios.com URL."""
        data = {}
        # URL is already a full URL from _get_all_urls, don't prepend domain again
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "lxml")

        for item in soup.find_all("div", {"class": "item"}):
            title = item.find("h2").text
            text = item.find("p").text
            text = unicodedata.normalize("NFKD", text)
            data[title] = text

        return data


if __name__ == "__main__":
    with IFRSScraper() as scraper:  # Needs "with" clause since it manages browser resources
        scraper.run()

    AccountingCoachScrapper().run()

    ReadyRatiosScrapper().run()

"""Scraper for UFV news portal that extracts latest news articles from the university website."""

import asyncio
import logging
import re
from datetime import datetime
from typing import List

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ValidationError

BASE_URL = "https://www2.dti.ufv.br/noticias/scripts/"
NEWS_LIST_URL = BASE_URL + "listaNoticiasMulti.php"


class NewsItem(BaseModel):
    """Pydantic model representing a news article."""

    code: str
    category: str
    title: str
    date: datetime
    content: str
    location: str


class NewsListItem(BaseModel):
    """Pydantic model representing a news list item."""

    code: str
    url: str


async def fetch_html(url: str, client: httpx.AsyncClient) -> str:
    """Fetch HTML content from a URL using an async HTTP client.

    Args:
        url: The URL to fetch.
        client: The httpx AsyncClient instance.

    Returns:
        The HTML content as a string.
    """
    response = await client.get(url)
    response.raise_for_status()
    return response.text


def parse_news_list(html: str) -> List[NewsListItem]:
    """Parse the news list page to extract news item codes and URLs.

    Args:
        html: HTML content of the news list page.

    Returns:
        List of NewsListItem objects with code and URL for each news item.
    """
    soup = BeautifulSoup(html, "html.parser")
    news_items = []
    # Find all table rows with id starting with 'noticia_'
    for tr in soup.find_all("tr", id=re.compile("^noticia_")):
        if not isinstance(tr, Tag):
            logging.warning("Skipping non-tag element in table row %s", tr)
            continue
        code = str(tr.get("id")).split("_")[-1]
        a_tag = tr.find("a", href=True)
        if not isinstance(a_tag, Tag):
            logging.warning("Skipping non-tag element in link inside table row %s", tr)
            continue
        if a_tag:
            href = a_tag["href"]
            # Build full URL if needed
            full_url = (
                BASE_URL + str(href) if not str(href).startswith("http") else str(href)
            )
            try:
                news_item = NewsListItem(code=code, url=full_url)
                news_items.append(news_item)
            except ValidationError as e:
                logging.error("Failed to validate news list item: %s", e)
    return news_items


def parse_news_detail(html: str, code: str) -> NewsItem:
    """Parse a news detail page to extract article information.

    Args:
        html: HTML content of the news detail page.
        code: The news item code.

    Returns:
        NewsItem model with article details.

    Raises:
        ValueError: If the news container cannot be found or is not a valid tag.
        ValidationError: If the parsed data fails Pydantic validation.
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="ExibeNoticia")
    if container is None:
        raise ValueError("Cannot find the news container.")
    if not isinstance(container, Tag):
        raise ValueError("The news container is not a tag.")
    category_tag = container.find("p", class_="Categoria")
    location_tag = container.find("p", class_="CampusDescricao")
    title_tag = container.find("h3")
    date_tag = container.find("p", id="Data")
    # Get all paragraphs with style="text-align:justify" for content.
    content_paragraphs = container.find_all("p", style=re.compile("text-align:justify"))
    content = "\n\n".join(p.get_text(strip=True) for p in content_paragraphs)

    date_text = date_tag.get_text(strip=True) if date_tag else ""

    # Convert date string to datetime object
    date_obj = None
    if date_text:
        try:
            date_obj = datetime.strptime(date_text, "%d/%m/%Y")
        except ValueError:
            try:
                date_obj = datetime.strptime(date_text, "%d/%m/%Y %H:%M:%S")
            except ValueError:
                logging.error("Invalid date format: %s, expected dd/mm/yyyy", date_text)
                # Use a default date if parsing fails
                date_obj = datetime.now()
    else:
        date_obj = datetime.now()

    data_dict = {
        "code": code,
        "category": category_tag.get_text(strip=True) if category_tag else "",
        "title": title_tag.get_text(strip=True) if title_tag else "",
        "date": date_obj,
        "content": content,
        "location": location_tag.get_text(strip=True) if location_tag else "",
    }

    try:
        return NewsItem(**data_dict)
    except ValidationError as e:
        logging.error("Validation error for news %s: %s", code, e)
        logging.debug("Data that failed validation: %s", data_dict)
        raise


async def crawl_news() -> List[NewsItem]:
    """Crawl the news portal to fetch and parse all available news articles.

    Returns:
        List of NewsItem objects containing details for each news article.
    """
    async with httpx.AsyncClient() as client:
        list_html = await fetch_html(NEWS_LIST_URL, client)
        news_list = parse_news_list(list_html)
        tasks = []
        for news in news_list:
            tasks.append(fetch_html(news.url, client))
        details_html_list = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for news, detail_html in zip(news_list, details_html_list):
            if isinstance(detail_html, Exception):
                logging.error(
                    "Error fetching news detail for %s: %s", news.url, detail_html
                )
                continue
            try:
                assert isinstance(detail_html, str), "detail_html must be a string"
                news_detail = parse_news_detail(detail_html, news.code)
                results.append(news_detail)
            except ValidationError as e:
                logging.error(
                    "Failed to validate news detail (code %s): %s", news.code, e
                )
                continue
            except ValueError as e:
                logging.error("Error parsing news detail (code %s): %s", news.code, e)
                continue
        return results


async def main() -> None:
    """Main entry point to run the news scraper and print results."""
    news_details = await crawl_news()
    for news in news_details:
        print(news.model_dump())


if __name__ == "__main__":
    asyncio.run(main())

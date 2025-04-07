"""Scraper for UFV news portal that extracts latest news articles from the university website,
respecting robots.txt rules.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup, Tag
from models import NewsModel, db_client
from pydantic import BaseModel, ValidationError

BASE_URL = "https://www2.dti.ufv.br/noticias/scripts/"
NEWS_LIST_URL = BASE_URL + "listaNoticiasMulti.php"
ROBOTS_URL = "https://www.ufv.br/robots.txt"


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
    """Fetch HTML content from a URL using an async HTTP client."""
    response = await client.get(url)
    response.raise_for_status()
    return response.text


def parse_news_list(html: str) -> List[NewsListItem]:
    """Parse the news list page to extract news item codes and URLs."""
    soup = BeautifulSoup(html, "html.parser")
    news_items = []
    for tr in soup.find_all("tr", id=re.compile("^noticia_")):
        if not isinstance(tr, Tag):
            logging.warning("Skipping non-tag element in table row %s", tr)
            continue
        code = str(tr.get("id")).rsplit("_", maxsplit=1)[-1]
        a_tag = tr.find("a", href=True)
        if not isinstance(a_tag, Tag):
            logging.warning("Skipping non-tag element in link inside table row %s", tr)
            continue
        href = a_tag["href"]
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
    """Parse a news detail page to extract article information with flexible content extraction."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="ExibeNoticia")
    if container is None or not isinstance(container, Tag):
        raise ValueError("Cannot find a valid news container.")

    # Fixed fields extraction
    category_tag = container.find("p", class_="Categoria")
    location_tag = container.find("p", class_="CampusDescricao")
    title_tag = container.find("h3")
    date_tag = container.find("p", id="Data")

    category = category_tag.get_text(strip=True) if category_tag else ""
    location = location_tag.get_text(strip=True) if location_tag else ""
    title = title_tag.get_text(strip=True) if title_tag else ""
    date_text = date_tag.get_text(strip=True) if date_tag else ""

    try:
        date_obj = datetime.strptime(date_text, "%d/%m/%Y")
    except ValueError:
        try:
            date_obj = datetime.strptime(date_text, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            logging.error("Invalid date format: %s, expected dd/mm/yyyy", date_text)
            date_obj = datetime.now()

    # Flexible content extraction
    # Define a starting element: prefer the date_tag, or if missing, the title_tag.
    start_elem = date_tag if date_tag else title_tag
    content_parts = []
    if start_elem:
        # Iterate over all siblings following the starting element.
        for sibling in start_elem.find_next_siblings():
            # Skip known non-content elements.
            if (
                isinstance(sibling, Tag)
                and sibling.name == "div"
                and sibling.get("class") is not None
                and "fb-share-button" in (sibling.get("class") or [])
            ):
                continue
            if (
                isinstance(sibling, Tag)
                and sibling.name == "p"
                and sibling.get("class") is not None
                and "Linha" in (sibling.get("class") or [])
            ):
                continue
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                content_parts.append(text)
    else:
        content_parts.append(container.get_text(separator=" ", strip=True))

    content = "\n\n".join(content_parts)

    data_dict = {
        "code": code,
        "category": category,
        "title": title,
        "date": date_obj,
        "content": content,
        "location": location,
    }
    try:
        return NewsItem(**data_dict)
    except ValidationError as e:
        logging.error("Validation error for news %s: %s", code, e)
        logging.debug("Data that failed validation: %s", data_dict)
        raise


async def load_robot_parser(robots_url: str) -> RobotFileParser:
    """Download and parse the robots.txt file from the given URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(robots_url)
        response.raise_for_status()
        content = response.text
    rp = RobotFileParser()
    rp.parse(content.splitlines())
    return rp


async def crawl_news() -> List[NewsItem]:
    """Crawl the news portal to fetch and parse all available news articles,
    considering robots.txt restrictions.
    """
    robot_parser = await load_robot_parser(ROBOTS_URL)
    # Check if crawling the news list page is allowed.
    if not robot_parser.can_fetch("*", NEWS_LIST_URL):
        logging.error("Crawling not allowed for NEWS_LIST_URL: %s", NEWS_LIST_URL)
        return []

    async with httpx.AsyncClient() as client:
        list_html = await fetch_html(NEWS_LIST_URL, client)
        news_list = parse_news_list(list_html)
        tasks = []
        for news in news_list:
            # Skip URLs disallowed by robots.txt.
            if not robot_parser.can_fetch("*", news.url):
                logging.info("Skipping URL %s due to robots.txt restrictions", news.url)
                continue
            tasks.append(fetch_html(news.url, client))
        details_html_list = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        # Note: we must match results to news items carefully if some tasks were skipped.
        # Here we iterate over the tasks that were actually launched.
        allowed_news = [
            news for news in news_list if robot_parser.can_fetch("*", news.url)
        ]
        for news, detail_html in zip(allowed_news, details_html_list):
            if isinstance(detail_html, Exception):
                logging.error(
                    "Error fetching news detail for %s: %s", news.url, detail_html
                )
                continue
            try:
                assert isinstance(detail_html, str), "detail_html must be a string"
                news_detail = parse_news_detail(detail_html, news.code)
                results.append(news_detail)
            except (ValidationError, ValueError) as e:
                logging.error("Error parsing news detail (code %s): %s", news.code, e)
                continue
        return results


async def store_news(news_items: List[NewsItem]) -> None:
    """Store the list of scraped news articles in the database using peewee."""

    def db_operations() -> None:
        db_client.connect(reuse_if_open=True)
        db_client.create_tables([NewsModel], safe=True)
        for item in news_items:
            NewsModel.insert(
                code=item.code,
                category=item.category,
                title=item.title,
                date=item.date,
                content=item.content,
                location=item.location,
            ).on_conflict_replace().execute()
        db_client.close()

    await asyncio.to_thread(db_operations)


async def main() -> None:
    """Main entry point to run the news scraper and store results in the database."""
    news_details = await crawl_news()
    await store_news(news_details)


if __name__ == "__main__":
    asyncio.run(main())

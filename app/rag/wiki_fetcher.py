"""Fetch Wikipedia pages for a city, returning plain text per page."""
import logging
from dataclasses import dataclass

import wikipediaapi

logger = logging.getLogger(__name__)


@dataclass
class WikiPage:
    title: str
    text: str
    url: str


class WikiFetcher:
    """Fetch a small fixed set of pages per city.

    We hand-pick page-name patterns rather than free search, so the agent
    consistently gets travel-relevant content (overview + tourism + districts).
    """

    PAGE_PATTERNS = [
        "{city}",
        "Tourism in {city}",
        "Culture of {city}",
    ]

    def __init__(self) -> None:
        # 'user_agent' is required by Wikipedia's API policy.
        self._wiki = wikipediaapi.Wikipedia(
            user_agent="voyage-agent/0.1 (hipster-assessment; contact: 1930013117a@gmail.com)",
            language="en",
        )

    def fetch_for_city(self, city: str) -> list[WikiPage]:
        pages: list[WikiPage] = []
        for pattern in self.PAGE_PATTERNS:
            title = pattern.format(city=city)
            page = self._wiki.page(title)
            if not page.exists():
                logger.info("Wikipedia page not found: %s", title)
                continue
            pages.append(
                WikiPage(title=page.title, text=page.text, url=page.fullurl)
            )
            logger.info("Fetched %s (%d chars)", page.title, len(page.text))
        return pages
import os
import requests
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("news")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

API_KEY = os.environ.get("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2"

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def get_top_headlines(query: str = "", country: str = "us", category: str = "general") -> list:
    """
    Gets top news headlines. You can specify a query, country, or category.
    :param query: Keywords or a phrase to search for.
    :param country: The 2-letter ISO 3166-1 code of the country you want to get headlines for.
    :param category: The category you want to get headlines for (e.g., business, entertainment, health).
    :return: A list of articles.
    """
    if not API_KEY:
        raise ValueError("NewsAPI key not configured. Please set the NEWS_API_KEY environment variable.")
    params = {
        "apiKey": API_KEY,
        "q": query,
        "country": country,
        "category": category,
    }
    response = requests.get(f"{BASE_URL}/top-headlines", params=params)
    response.raise_for_status()
    return response.json().get("articles", [])


@mcp.tool()
async def get_everything(query: str, language: str = "en", sort_by: str = "relevancy") -> list:
    """
    Searches for articles from a wide range of sources.
    :param query: Keywords or a phrase to search for.
    :param language: The 2-letter ISO-639-1 code of the language you want to get articles for.
    :param sort_by: The order to sort the articles in (relevancy, popularity, publishedAt).
    :return: A list of articles.
    """
    if not API_KEY:
        raise ValueError("NewsAPI key not configured. Please set the NEWS_API_KEY environment variable.")
    params = {
        "apiKey": API_KEY,
        "q": query,
        "language": language,
        "sortBy": sort_by,
    }
    response = requests.get(f"{BASE_URL}/everything", params=params)
    response.raise_for_status()
    return response.json().get("articles", [])

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

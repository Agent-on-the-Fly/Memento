"""
crawl_server.py – FastMCP server exposing a crawl‑and‑clean tool
----------------------------------------------------------------
* crawl_page(url) – Fetch a web page with **crawl4ai.AsyncWebCrawler**
                    and return the readable Markdown.

Dependencies
    pip install crawl4ai mcp fastmcp
"""

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #

from mcp.server.fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler
import argparse
from interpreters.logger import get_logger

# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("crawl")

# --------------------------------------------------------------------------- #
#  Tool
# --------------------------------------------------------------------------- #


@mcp.tool()
async def crawl_page(url: str) -> str:
    """Deep crawl and extract key content from a web page (Markdown format).

    This tool is designed to perform *deep analysis* on a specific link
    retrieved from an earlier search step (e.g., via a web search tool).
    Given a fully qualified HTTP(S) URL, it fetches the web page,
    removes boilerplate content (menus, ads, nav bars, etc.), and
    extracts the core readable content, returning it as a clean,
    structured Markdown string.

    This Markdown output is well-suited for downstream processing by
    large language models (LLMs) for tasks such as:
    - Answering user questions from a specific page
    - Summarizing long articles or reports
    - Extracting facts, definitions, lists, or instructions
    - Contextual search over high‑signal content

    This is often used as a **follow-up** step after a general-purpose
    search tool (e.g., via SearxNG), when the agent needs to "click through"
    to an individual link and analyze its full content in a readable form.

    Args:
        url (str): A valid, fully-qualified URL (http:// or https://) that
            points to a real and accessible web page (e.g. news article,
            blog post, research page).

    Returns:
        str: Markdown-formatted main content of the page. If the crawl fails
            (due to network errors, access restrictions, or page layout
            issues), a plain-text error message is returned instead.
    """
    logger.info(f"Crawling page: {url}")
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown
    except Exception as exc:
        logger.error(f"Crawl error on {url}: {exc}")
        return f"⚠️ Crawl error: {exc!s}"


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args, _ = parser.parse_known_args()

    log_level = args.log_level.upper()
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)

    # Use stdio when embedding inside an agent, or HTTP during development.
    mcp.run(transport="stdio")
"""
search_server.py – FastMCP server exposing a SearxNG search tool
----------------------------------------------------------------
* search(query, num_results?, category?, language?, time_range?,
          safe_search?, host?)
      – Privacy‑respecting web search powered by any SearxNG instance.

Dependencies
    pip install httpx mcp fastmcp
"""

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #

from typing import Any, Dict, List, Optional
import argparse
from interpreters.logger import get_logger

import httpx
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("search")

# --------------------------------------------------------------------------- #
#  Constants & simple validators
# --------------------------------------------------------------------------- #

DEFAULT_HOST = "http://127.0.0.1:8080"  # pick any public instance you like
DEFAULT_CATEGORY = "general"
SAFE_SEARCH_LEVELS = {0, 1, 2}
VALID_TIME_RANGES = {"day", "week", "month", "year"}


def _check_safe(level: int) -> None:
    if level not in SAFE_SEARCH_LEVELS:
        raise ValueError(f"safe_search must be 0, 1 or 2 (got {level}).")


def _check_time_range(rng: Optional[str]) -> None:
    if rng is not None and rng not in VALID_TIME_RANGES:
        raise ValueError(
            f"time_range must be one of {sorted(VALID_TIME_RANGES)} (got {rng})."
        )


# --------------------------------------------------------------------------- #
#  Tool
# --------------------------------------------------------------------------- #


@mcp.tool()
async def search(
    query: str,
    num_results: int = 10,
    category: str | None = None,
    language: str = "en",
    time_range: str | None = None,
    safe_search: int = 1,
    host: str = DEFAULT_HOST,
) -> List[Dict[str, str]]:
    """Run a web search via any SearxNG instance (defaults to *nicfab.eu*).

    Args:
        query: The search string.
        num_results: Max results to return (default 10 — max 20 is polite).
        category: SearxNG category: *general*, *images*, *videos*, *news*,
            *map*, *music*, *it*, *science*, *files*, *social media*. 
            Categories not listed here can still be searched with the Search syntax.
            Defaults to *general*.
        language: Two‑letter language code (default "en").
        time_range: Optional freshness filter: "day" | "week" | "month" | "year".
        safe_search: 0 = off • 1 = moderate • 2 = strict (default 1).
        host: Full base‑URL of the SearxNG instance to query.

    Returns:
        A list of dicts with **title**, **link**, **snippet** keys.
    """
    _check_safe(safe_search)
    _check_time_range(time_range)

    params: Dict[str, Any] = {
        "q": query,
        "format": "json",
        "language": language,
        "categories": category or DEFAULT_CATEGORY,
        "pageno": 1,
        "safe": safe_search,
    }
    if time_range:
        params["time_range"] = time_range

    url = f"{host.rstrip('/')}/search"
    logger.info("Searching for '%s' on %s", query, url)

    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "fastmcp-search"}) as client:
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            results = r.json().get("results", [])[: num_results]
        except Exception as exc:  # network / JSON / key errors
            logger.error("Search failed for query '%s': %s", query, exc)
            return [{"title": "Search error", "link": "", "snippet": str(exc)}]

    return [
        {
            "title": it.get("title", ""),
            "link": it.get("url", ""),
            "snippet": it.get("content", ""),
        }
        for it in results
    ]


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

    mcp.run(transport="stdio")  # or: mcp.run(host="0.0.0.0", port=5002)

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #

import os
from typing import Optional, Tuple

import requests
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("fetch")

# --------------------------------------------------------------------------- #
#  Custom Exception
# --------------------------------------------------------------------------- #


class FetchError(Exception):
    """User-friendly exception to describe fetch failure reasons."""
    pass


# --------------------------------------------------------------------------- #
#  Tool
# --------------------------------------------------------------------------- #


@mcp.tool()
async def fetch_markdown(
    url: str,
    api_key: Optional[str] = None,
    timeout: int = 90,
    accept: str = "text/markdown, text/plain",
) -> str:
    """Fetch cleaned Markdown content from a web page via Jina Reader API.

    This tool uses Jina Reader (r.jina.ai) to fetch and clean web page content,
    returning it as structured Markdown. The service automatically removes ads,
    navigation, and boilerplate, extracting only the main content.

    Use cases:
    - Extract article content from news sites, blogs, documentation pages
    - Get readable content from complex web pages with heavy JavaScript
    - Obtain clean text for downstream LLM processing or analysis
    - Alternative to web scraping for content extraction

    ⚠️ Important Notes:
    - Some sites (e.g., Google Scholar) may return 403 Forbidden due to
      anti-scraping measures. For academic data, use official APIs instead:
      OpenAlex, Semantic Scholar, Crossref, ORCID, arXiv, etc.
    - Requires a valid Jina API key (can be set via JINA_API_KEY env var)
    - Respects rate limits and robots.txt directives

    Args:
        url: The full HTTP(S) URL of the web page to fetch.
        api_key: Jina API key. If not provided, reads from JINA_API_KEY env var.
        timeout: Maximum time (in seconds) to wait for the response (default: 90).
            Increase for slow-loading pages.
        accept: HTTP Accept header value (default: "text/markdown, text/plain").
            Controls the output format from Jina Reader.

    Returns:
        str: Cleaned Markdown content of the web page. On error, returns a
            user-friendly error message with troubleshooting suggestions.

    Raises:
        None: All errors are caught and returned as formatted error messages
            within the returned string for better LLM interpretability.
    """
    reader_url = f"https://r.jina.ai/{url}"
    key = api_key or os.getenv("JINA_API_KEY")
    headers = {"Accept": accept}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        # Use synchronous requests within async function (fine for I/O-bound operations)
        resp = requests.get(
            reader_url,
            headers=headers,
            timeout=(5, timeout)  # (connect, read)
        )
    except requests.Timeout as e:
        return (
            f"⚠️ Request Timeout: Request exceeded {timeout} seconds without completion.\n\n"
            f"Suggestions:\n"
            f"- Retry with a larger timeout parameter\n"
            f"- Target page may be loading slowly or is inaccessible\n"
            f"- Consider using a more stable data source\n\n"
            f"Details: {e}"
        )
    except requests.RequestException as e:
        return f"⚠️ Network Error: Unable to connect to Jina Reader or target site.\nDetails: {e}"

    # Explicitly distinguish 403 (anti-scraping/forbidden) from other errors
    if resp.status_code == 403:
        return (
            f"⚠️ Target Site Returned 403 Forbidden\n\n"
            f"Reason: The target website may have detected scraping behavior or requires authentication/CAPTCHA.\n\n"
            f"Suggestions:\n"
            f"- Do not scrape sites with strict anti-scraping measures like Google Scholar\n"
            f"- For academic data, use official APIs instead:\n"
            f"  • OpenAlex (https://openalex.org/)\n"
            f"  • Semantic Scholar API (https://www.semanticscholar.org/product/api)\n"
            f"  • Crossref (https://www.crossref.org/)\n"
            f"  • ORCID (https://orcid.org/)\n"
            f"  • arXiv (https://arxiv.org/help/api)\n"
            f"- Switch to a web page that allows public access\n"
        )

    if not resp.ok:
        error_preview = resp.text[:200] if resp.text else "(empty response)"
        return (
            f"⚠️ Fetch Failed: HTTP {resp.status_code}\n\n"
            f"Response Content (first 200 characters):\n{error_preview}\n\n"
            f"Suggestions: Check if the URL is valid or if the target site is accessible."
        )

    return resp.text


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Use stdio when embedding inside an agent, or HTTP during development.
    mcp.run(transport="stdio")

import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

import colorlog
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("mcp.crawl_extract")

# --------------------------------------------------------------------------- #
#  Fixed model
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gpt-4.1"

EXTRACTOR_SYSTEM_PROMPT = (
    "You are a careful, concise information extraction assistant. "
    "Given a user query and a webpage in Markdown, do NOT summarize the whole page. "
    "Instead, extract ONLY the content that directly answers or is highly relevant to the query. "
    "If the answer is not present, say so explicitly.\n\n"
    "Output format:\n"
    "1) Direct Answer: 2–4 concise sentences (or 'Not found in page').\n"
    "2) Key Evidence: Bullet list of short quotes from the Markdown (quote verbatim, minimal trimming).\n"
    "3) Entities/Numbers: Bullet list of important names, dates, figures tied to the query.\n"
    "4) Uncertainties: Note any ambiguities or missing info.\n"
    "Stay grounded in the provided Markdown. Avoid fabrication."
)

# --------------------------------------------------------------------------- #
#  Chat backend
# --------------------------------------------------------------------------- #
class OpenAIBackend:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.model = DEFAULT_MODEL  
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 30000,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = await self.client.chat.completions.create(**payload)
        msg = resp.choices[0].message
        return {"content": msg.content or ""}

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
async def _extract_for_query(
    backend: OpenAIBackend,
    md: str,
    query: str,
    *,
    max_tokens: int = 30000,
    temperature: float = 0.1,
) -> str:
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query:\n{query}\n\nPage Markdown (verbatim):\n\n{md}"},
    ]
    res = await backend.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
    return (res["content"] or "").strip()

async def _crawl_markdown(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        md = (result.markdown or "").strip()
        if not md:
            return result
        return md

async def _crawl_and_extract(
    url: str,
    query: str,
    *,
    max_tokens: int = 30000,
    temperature: float = 0.1,
) -> str:
    logger.info(f"Crawling: {url}")
    md = await _crawl_markdown(url)
    logger.info(f"Markdown length: {len(md):,} characters")
    backend = OpenAIBackend()  
    return await _extract_for_query(
        backend,
        md,
        query=query,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# --------------------------------------------------------------------------- #
#  FastMCP server
# --------------------------------------------------------------------------- #
mcp = FastMCP("crawl-extract")

@mcp.tool()
async def crawl_extract(
    url: str,
    query: str,
    temperature: float = 0.1,
    max_tokens: int = 30000,
) -> str:
    """
    Crawl a URL to Markdown and extract only the content relevant to the query.

    Parameters
    ----------
    url : str
        Target URL to crawl.
    query : str
        The information need; used to extract only the most relevant snippets from the page Markdown.
    temperature : float, optional (default: 0.1)
        Sampling temperature for the extraction model.
    max_tokens : int, optional (default: 1400)
        Maximum tokens allowed in the extraction model response.

    Returns
    -------
    str
        A compact, four-part extraction (Direct Answer, Key Evidence, Entities/Numbers, Uncertainties).

    Notes
    -----
    - Exposed as the single MCP tool.
    - The underlying model is fixed to 'gpt-4.1'.
    """
    return await _crawl_and_extract(
        url=url,
        query=query,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mcp.run(transport="stdio")

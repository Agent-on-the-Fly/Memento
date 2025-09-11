import os
import requests
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("research_assistant")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

client = None
if os.getenv("OPENAI_API_KEY"):
    client = AsyncOpenAI()

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def find_related_papers(query: str, max_results: int = 5) -> list:
    """
    Finds related academic papers on ArXiv.
    :param query: The search query (e.g., 'quantum computing', 'author:einstein').
    :param max_results: The maximum number of papers to return.
    :return: A list of related papers with their titles, authors, and summary.
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }
    response = requests.get(ARXIV_API_URL, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        papers.append({"title": title, "authors": authors, "summary": summary.strip()})

    return papers


@mcp.tool()
async def generate_hypothesis(observation: str, background_knowledge: str) -> str:
    """
    Generates a testable hypothesis from an observation and background knowledge.
    :param observation: The observation to be explained.
    :param background_knowledge: Relevant background knowledge.
    :return: A testable hypothesis.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a scientific research assistant that helps formulate hypotheses."},
            {"role": "user", "content": f"Based on the following observation and background knowledge, generate a testable hypothesis.\n\nObservation: {observation}\n\nBackground Knowledge: {background_knowledge}"},
        ],
    )
    return response.choices[0].message.content


@mcp.tool()
async def design_experiment(hypothesis: str) -> str:
    """
    Proposes a basic experimental design to test a hypothesis.
    :param hypothesis: The hypothesis to be tested.
    :return: A basic experimental design.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a scientific research assistant that helps design experiments."},
            {"role": "user", "content": f"Propose a basic, high-level experimental design to test the following hypothesis: {hypothesis}"},
        ],
    )
    return response.choices[0].message.content

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

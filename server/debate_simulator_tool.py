import os
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP
import argparse
from interpreters.logger import get_logger

# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("debate_simulator")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

client = None
if os.getenv("OPENAI_API_KEY"):
    client = AsyncOpenAI()

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def generate_argument(topic: str, stance: str, persona: str) -> str:
    """
    Generates an argument for a topic from a specific stance and persona.
    :param topic: The topic of the debate.
    :param stance: The stance to take on the topic (e.g., 'pro', 'con').
    :param persona: The persona to adopt for the argument (e.g., 'tech-optimist', 'skeptic').
    :return: An argument for the topic.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a debate simulator. Your persona is '{persona}'. You are arguing '{stance}' the topic of '{topic}'."},
            {"role": "user", "content": "Please generate a strong opening argument for your position."},
        ],
    )
    return response.choices[0].message.content


@mcp.tool()
async def generate_counter_argument(argument: str) -> str:
    """
    Generates a counter-argument to a given argument.
    :param argument: The argument to counter.
    :return: A counter-argument.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a debate simulator. You are skilled at finding flaws in arguments and generating strong counter-arguments."},
            {"role": "user", "content": f"Please generate a strong counter-argument to the following argument: '{argument}'"},
        ],
    )
    return response.choices[0].message.content

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

    mcp.run(transport="stdio")

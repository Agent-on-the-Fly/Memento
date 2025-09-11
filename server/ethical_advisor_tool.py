import os
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("ethical_advisor")

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
async def analyze_action_ethics(action: str, framework: str) -> str:
    """
    Analyzes the ethical implications of a proposed action based on a specific ethical framework.
    :param action: The action to be analyzed.
    :param framework: The ethical framework to use (e.g., 'utilitarianism', 'deontology').
    :return: An analysis of the action's ethics.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are an ethics professor specializing in the '{framework}' framework. Analyze the following action from that perspective."},
            {"role": "user", "content": action},
        ],
    )
    return response.choices[0].message.content


@mcp.tool()
async def get_ethical_recommendation(dilemma: str) -> str:
    """
    Provides a recommendation for an ethical dilemma.
    :param dilemma: The ethical dilemma to be resolved.
    :return: A recommendation for the dilemma.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI ethics advisor. Provide a balanced and nuanced recommendation for the following ethical dilemma, considering various perspectives."},
            {"role": "user", "content": dilemma},
        ],
    )
    return response.choices[0].message.content

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

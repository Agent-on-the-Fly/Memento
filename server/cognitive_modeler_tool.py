import os
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("cognitive_modeler")

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
async def apply_cognitive_bias(text: str, bias_type: str) -> str:
    """
    Re-evaluates a piece of text from the perspective of a specific cognitive bias.
    :param text: The text to re-evaluate.
    :param bias_type: The type of cognitive bias to apply (e.g., 'confirmation', 'availability', 'negativity').
    :return: The re-evaluated text.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are an expert in cognitive biases. Re-write the following text from the perspective of someone exhibiting a strong '{bias_type}' bias. Explain the reasoning behind the biased interpretation."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


@mcp.tool()
async def simulate_decision_making(scenario: str, personality_profile: str) -> str:
    """
    Simulates how a person with a given personality profile might decide in a given scenario.
    :param scenario: The scenario for the decision.
    :param personality_profile: The personality profile of the decision-maker (e.g., 'cautious', 'impulsive', 'analytical').
    :return: The simulated decision-making process.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a decision-making simulator. Your persona is a '{personality_profile}' individual. Describe your thought process and final decision for the given scenario."},
            {"role": "user", "content": scenario},
        ],
    )
    return response.choices[0].message.content

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

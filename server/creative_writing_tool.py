import os
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("creative_writing")

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
async def generate_plot_twist(story_context: str) -> str:
    """
    Generates a surprising plot twist based on the current story context.
    :param story_context: A summary of the story so far.
    :return: A plot twist.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a creative writing assistant specializing in plot twists."},
            {"role": "user", "content": f"Here is the story so far: {story_context}. Now, give me a surprising and unexpected plot twist."},
        ],
    )
    return response.choices[0].message.content

@mcp.tool()
async def create_character_profile(archetype: str, traits: str) -> str:
    """
    Creates a detailed character profile, including backstory, motivations, and flaws.
    :param archetype: The character archetype (e.g., 'hero', 'villain', 'mentor').
    :param traits: A comma-separated list of character traits (e.g., 'brave, reckless, loyal').
    :return: A detailed character profile.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a character creation expert for creative writing."},
            {"role": "user", "content": f"Create a detailed character profile for a {archetype} with the following traits: {traits}. Include a backstory, motivations, and flaws."},
        ],
    )
    return response.choices[0].message.content

@mcp.tool()
async def generate_storyboard(scene_description: str) -> list:
    """
    Generates a sequence of images representing a scene for a storyboard.
    :param scene_description: A detailed description of the scene.
    :return: A list of URLs to the generated images.
    """
    if not client:
        raise ValueError("OpenAI API key not configured.")

    response = await client.images.generate(
        model="dall-e-3",
        prompt=f"A sequence of 4 storyboard panels for the following scene: {scene_description}. The images should have a consistent style.",
        n=1,
        size="1024x1024",
    )
    return [image.url for image in response.data]

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

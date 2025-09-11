"""
mcp_image_analysis.py
FastMCP server – vision tools (image → caption / VQA)
"""

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #
import base64
import io
import os
from typing import Optional
import argparse
import logging
from interpreters.logger import get_logger

import anyio
import openai
import requests
from PIL import Image
from urllib.parse import urlparse
from openai import AsyncOpenAI              # ← new

from mcp.server.fastmcp import FastMCP


from dotenv import load_dotenv
load_dotenv(".env")


# --------------------------------------------------------------------------- #
#  Helper class
# --------------------------------------------------------------------------- #
class ImageAnalysisToolkit:
    """
    Very small wrapper around OpenAI Vision GPT models.
    Provides two public coroutines:
        • image_to_text
        • ask_question_about_image
    """

    def __init__(self, timeout: float | None = None):
        self.timeout = timeout or 15

    # ---------------- public API ------------------------------------------ #
    async def image_to_text(
        self, image_path: str, sys_prompt: Optional[str] = None
    ) -> str:
        """
        Return a detailed caption of *image_path*.
        """
        default_sys = (
            "You are an expert image analyst. Provide a rich, concise "
            "description of everything visible, including any text."
        )
        return await self._chat_with_image(
            image_path,
            user_prompt="Please describe the contents of this image.",
            system_prompt=sys_prompt or default_sys,
        )

    async def ask_question_about_image(
        self,
        image_path: str,
        question: str,
        sys_prompt: Optional[str] = None,
    ) -> str:
        """
        Answer *question* about *image_path*.
        """
        default_sys = (
            "You answer questions about images by careful visual inspection, "
            "reading any text, and reasoning from what you see. Please consider the reqirements of the question carefully"
        )
        return await self._chat_with_image(
            image_path,
            user_prompt=question,
            system_prompt=sys_prompt or default_sys,
        )

    # ---------------- implementation -------------------------------------- #
    async def _chat_with_image(
        self, image_path: str, user_prompt: str, system_prompt: str
    ) -> str:
        """
        Core routine: prepare image, run OpenAI vision chat, return content.
        """
        image_url = await self._prepare_image(image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),  # works with Azure etc.
        )

        try:
            logger.info("Sending image to OpenAI ChatCompletion (vision)…")
            response = await openai_client.chat.completions.create(
                model="gemini-2.5-pro-preview-05-06",
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("OpenAI call failed: %s", e)
            raise

    async def _prepare_image(self, path: str) -> str:
        """
        Turn *path* (local path or URL) into a URL or data‑URL acceptable to
        the OpenAI Vision endpoint.
        """
        parsed = urlparse(path)

        # Remote URL – just return it (OpenAI fetches it directly)
        if parsed.scheme in ("http", "https"):
            logger.debug("Using remote image URL: %s", path)
            return path

        # Local file – read & encode
        logger.debug("Encoding local image: %s", path)
        data = await anyio.to_thread.run_sync(lambda: open(path, "rb").read())
        mime = Image.open(io.BytesIO(data)).get_format_mimetype()
        b64 = base64.b64encode(data).decode()
        return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP server
# --------------------------------------------------------------------------- #
mcp = FastMCP("image_analysis")
toolkit = ImageAnalysisToolkit()


@mcp.tool()
async def image_to_text(image_path: str, sys_prompt: Optional[str] = None) -> str:
    """
    Generates a detailed and descriptive caption of the image located at *image_path*.

    Parameters:
    - image_path (str): The file path or URL of the image to analyze.
    - sys_prompt (Optional[str]): An optional system prompt that can guide or influence the image captioning behavior, allowing for customization of the description style, detail level, or focus.

    Returns:
    - str: A detailed natural language description of the content, objects, scene, and relevant features in the image.
    """
    return await toolkit.image_to_text(image_path, sys_prompt)


@mcp.tool()
async def ask_question_about_image(
    image_path: str, question: str, sys_prompt: Optional[str] = None
) -> str:
    """
    Answers a specific question related to the content of the image located at *image_path*.

    Parameters:
    - image_path (str): The file path or URL of the image to analyze.
    - question (str): The question to be answered about the image content.
    - sys_prompt (Optional[str]): An optional system prompt to guide the reasoning or answering style, providing context or desired behavior for the image analysis.

    Returns:
    - str: The answer to the question based on visual analysis and understanding of the image content.
    """
    return await toolkit.ask_question_about_image(image_path, question, sys_prompt)

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
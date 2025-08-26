#!/usr/bin/env python
"""
FastMCP video‑helper server (async OpenAI edition).

Dependencies
------------
pip install \
  yt_dlp ffmpeg-python pillow \
  opencv-python numpy scenedetect \
  openai>=1.14.0  # must include AsyncOpenAI
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import List

import ffmpeg
import yt_dlp
from mcp.server.fastmcp import FastMCP
from PIL import Image
import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from openai import AsyncOpenAI              # ← new
from dotenv import load_dotenv


load_dotenv()  # picks up OPENAI_* variables from .env if present

# --------------------------------------------------------------------------- #
#  OpenAI client (async)                                                      #
# --------------------------------------------------------------------------- #

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),  # works with Azure etc.
)

# --------------------------------------------------------------------------- #
#  FastMCP instance                                                            #
# --------------------------------------------------------------------------- #

mcp = FastMCP("video_tools")

# --------------------------------------------------------------------------- #
#  Helper functions (unchanged except for async OpenAI calls)                 #
# --------------------------------------------------------------------------- #


def _capture_screenshot(video_file: str, timestamp: float, width: int = 320) -> Image.Image:
    if not os.path.isfile(video_file):
        raise ValueError(f"Invalid video file path: {video_file}")
    out, _ = (
        ffmpeg.input(video_file, ss=timestamp)
        .filter("scale", width, -1)
        .output("pipe:", vframes=1, format="image2", vcodec="png")
        .run(capture_stdout=True, capture_stderr=True)
    )
    return Image.open(io.BytesIO(out))


def _extract_audio(video_file: str, output_format: str = "mp3") -> str:
    if not os.path.isfile(video_file):
        raise ValueError(f"Invalid video file path: {video_file}")
    basename = os.path.splitext(video_file)[0]
    out_path = f"{basename}.{output_format}"
    (
        ffmpeg.input(video_file)
        .output(out_path, vn=None, acodec="libmp3lame")
        .run(quiet=True)
    )
    return out_path


async def _transcribe_audio_async(audio_path: str) -> str:
    """Whisper transcription via AsyncOpenAI; returns '' if disabled."""
    if not openai_client.api_key:
        return ""
    rsp = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
    )
    return rsp.text.strip()


def _normalize(img: Image.Image, target_width: int = 512) -> Image.Image:
    w, h = img.size
    return img.resize((target_width, int(target_width * h / w)), Image.Resampling.LANCZOS).convert("RGB")


def _extract_keyframes(
    video_path: str,
    frame_interval: float = 4.0,
    max_frames: int = 100,
    target_width: int = 512,
) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps else 0
    cap.release()

    desired = min(max(int(duration / frame_interval) or 1, 1), max_frames)

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()

    frames: List[Image.Image] = []
    if scenes:
        for i in np.linspace(0, len(scenes) - 1, min(len(scenes), desired), dtype=int):
            frames.append(_capture_screenshot(video_path, scenes[i][0].get_seconds()))

    while len(frames) < desired and duration:
        t = len(frames) * frame_interval
        try:
            frames.append(_capture_screenshot(video_path, t))
        except ffmpeg.Error:  # short file edge‑case
            break

    return [_normalize(f, target_width) for f in frames]


def _images_to_base64(imgs: List[Image.Image]) -> List[str]:
    out: List[str] = []
    for im in imgs:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


# --------------------------------------------------------------------------- #
#  Tools                                                                      #
# --------------------------------------------------------------------------- #

@mcp.tool()
async def download_video(url: str, download_directory: str | None = None) -> str:
    """
    Downloads a video from the given URL using yt_dlp and returns the local file path.

    Parameters:
    - url (str): The URL of the video to download. Supported platforms include YouTube and others compatible with yt_dlp.
    - download_directory (Optional[str]): Optional path to the directory where the video will be saved. If not specified, a temporary directory will be used.

    Returns:
    - str: The full file path of the downloaded video file.
    """
    download_directory = download_directory or tempfile.mkdtemp()
    Path(download_directory).mkdir(parents=True, exist_ok=True)
    template = str(Path(download_directory) / "%(title)s.%(ext)s")
    opts = {"format": "bestvideo+bestaudio/best", "outtmpl": template}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


@mcp.tool()
async def get_video_bytes(video_path: str) -> bytes:
    """
    Reads and returns the raw bytes of the video file located at *video_path*.

    Parameters:
    - video_path (str): The local path to the video file.

    Returns:
    - bytes: The binary content of the video file.
    """
    with open(video_path, "rb") as fh:
        return fh.read()


@mcp.tool()
async def get_video_screenshots(video_path: str, amount: int = 3) -> List[str]:
    """
    Captures uniformly distributed screenshots from the video file and returns them as base64-encoded JPEG strings.

    Parameters:
    - video_path (str): The local path to the video file.
    - amount (int): The number of screenshots to capture evenly spaced throughout the video. Default is 3.

    Returns:
    - List[str]: A list of base64-encoded JPEG images representing the captured frames.
    """

    probe = ffmpeg.probe(video_path)
    dur = float(probe["format"]["duration"])
    step = dur / (amount + 1)
    imgs = [_capture_screenshot(video_path, (i + 1) * step) for i in range(amount)]
    return _images_to_base64(imgs)


VIDEO_QA_PROMPT = """
Use the key‑frames and (optional) transcription to answer.

Transcription (may be empty):
{transcription}

Question:
{question}
""".strip()


@mcp.tool()
async def ask_question_about_video(
    video_path: str,
    question: str,
    use_audio_transcription: bool = False,
) -> str:
    """
    Answers a specific question about the content of the video file by analyzing keyframes and optionally its audio transcription, using multimodal GPT-4o.

    Parameters:
    - video_path (str): The local path to the video file.
    - question (str): The question to be answered based on the video content.
    - use_audio_transcription (bool): Whether to include audio transcription via Whisper model (AsyncOpenAI) to assist in answering the question. Default is False.

    Returns:
    - str: The AI-generated answer to the question based on visual and (optional) audio analysis of the video.
    """
    if not openai_client.api_key:
        return "OPENAI_API_KEY not set."

    frames = _extract_keyframes(video_path)
    images_b64 = _images_to_base64(frames)
    transcription = ""
    if use_audio_transcription:
        transcription = await _transcribe_audio_async(_extract_audio(video_path))

    user_message = [
        {
            "type": "text",
            "text": VIDEO_QA_PROMPT.format(transcription=transcription, question=question),
        },
        *(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
            for b in images_b64
        ),
    ]

    chat = await openai_client.chat.completions.create(
        model="gemini-2.5-pro-preview-05-06",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=512,
    )
    return chat.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
#  Entrypoint                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

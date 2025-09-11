import os
from fpdf import FPDF
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

mcp = FastMCP("pdf_generator")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

PDF_DIR = "generated_pdfs"
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def create_pdf_from_text(text: str, filename: str) -> str:
    """
    Creates a PDF file from a string of text.
    :param text: The text content to put in the PDF.
    :param filename: The name of the output PDF file (without .pdf extension).
    :return: The path to the created PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)

    filepath = os.path.join(PDF_DIR, f"{filename}.pdf")
    pdf.output(filepath)

    return filepath

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

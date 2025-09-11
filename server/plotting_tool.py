import os
import uuid
import matplotlib.pyplot as plt
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

mcp = FastMCP("plotting")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

PLOT_DIR = "generated_plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def save_plot(title: str) -> str:
    """Saves the current plot to a file with a unique name."""
    filename = f"{title.replace(' ', '_').lower()}_{uuid.uuid4()}.png"
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def create_bar_chart(data: dict, title: str, xlabel: str, ylabel: str) -> str:
    """
    Creates a bar chart from a dictionary of data and saves it as an image.
    :param data: A dictionary with labels as keys and values as values.
    :param title: The title of the chart.
    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    :return: The path to the saved plot image.
    """
    plt.figure()
    plt.bar(data.keys(), data.values())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return save_plot(title)


@mcp.tool()
async def create_line_chart(x: list, y: list, title: str, xlabel: str, ylabel: str) -> str:
    """
    Creates a line chart from lists of x and y values and saves it as an image.
    :param x: A list of x-axis values.
    :param y: A list of y-axis values.
    :param title: The title of the chart.
    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    :return: The path to the saved plot image.
    """
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    return save_plot(title)


@mcp.tool()
async def create_scatter_plot(x: list, y: list, title: str, xlabel: str, ylabel: str) -> str:
    """
    Creates a scatter plot from lists of x and y values and saves it as an image.
    :param x: A list of x-axis values.
    :param y: A list of y-axis values.
    :param title: The title of the chart.
    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    :return: The path to the saved plot image.
    """
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    return save_plot(title)

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

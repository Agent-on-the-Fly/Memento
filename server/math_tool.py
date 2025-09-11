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

mcp = FastMCP("math")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def add(a: float, b: float) -> float:
    """Return a + b."""
    logger.info(f"Adding {a} and {b}")
    return a + b


@mcp.tool()
async def sub(a: float, b: float) -> float:
    """Return a - b."""
    logger.info(f"Subtracting {b} from {a}")
    return a - b


@mcp.tool()
async def multiply(a: float, b: float, decimal_places: int = 2) -> float:
    """Return a * b, rounded to *decimal_places* (default 2)."""
    logger.info(f"Multiplying {a} and {b}")
    return round(a * b, decimal_places)


@mcp.tool()
async def divide(a: float, b: float, decimal_places: int = 2) -> float:
    """Return a / b, rounded to *decimal_places* (default 2)."""
    logger.info(f"Dividing {a} by {b}")
    if b == 0:
        logger.error("Division by zero")
        raise ValueError("division by zero")
    return round(a / b, decimal_places)


@mcp.tool()
async def round(a: float, decimal_places: int = 0) -> float:   # noqa: A001
    """Round *a* to *decimal_places* (default 0)."""
    logger.info(f"Rounding {a} to {decimal_places} decimal places")
    return round(a, decimal_places)

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
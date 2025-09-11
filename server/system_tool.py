import psutil
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

mcp = FastMCP("system")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def get_cpu_usage() -> float:
    """
    Returns the current system-wide CPU utilization as a percentage.
    This is a non-blocking call.
    :return: The CPU usage percentage.
    """
    return psutil.cpu_percent(interval=None)


@mcp.tool()
async def get_memory_info() -> dict:
    """
    Returns system memory usage statistics.
    :return: A dictionary with memory stats like total, available, used, and percentage.
    """
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "used": mem.used,
        "percent": mem.percent,
    }


@mcp.tool()
async def get_disk_usage(path: str = "/") -> dict:
    """
    Returns disk usage statistics for a given path.
    :param path: The path to get disk usage for (e.g., '/').
    :return: A dictionary with disk usage stats like total, used, free, and percentage.
    """
    disk = psutil.disk_usage(path)
    return {
        "total": disk.total,
        "used": disk.used,
        "free": disk.free,
        "percent": disk.percent,
    }

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

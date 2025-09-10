import psutil
from mcp.server.fastmcp import FastMCP

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
    mcp.run(transport="stdio")

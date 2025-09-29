import builtins
from mcp.server.fastmcp import FastMCP

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
    return a + b


@mcp.tool()
async def sub(a: float, b: float) -> float:
    """Return a - b."""
    return a - b


@mcp.tool()
async def multiply(a: float, b: float, decimal_places: int = 2) -> float:
    """Return a * b, rounded to *decimal_places* (default 2)."""
    return round(a * b, decimal_places)


@mcp.tool()
async def divide(a: float, b: float, decimal_places: int = 2) -> float:
    """Return a / b, rounded to *decimal_places* (default 2)."""
    if b == 0:
        raise ValueError("division by zero")
    return round(a / b, decimal_places)


@mcp.tool()
async def round(a: float, decimal_places: int = 0) -> float:   # noqa: A001
    """Round *a* to *decimal_places* (default 0)."""
    return builtins.round(a, decimal_places)

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
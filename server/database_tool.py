import sqlite3
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

mcp = FastMCP("database")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def execute_sql(database_path: str, query: str) -> list:
    """
    Executes a SQL query on a specified SQLite database and returns the results.
    This tool can execute any SQL query, including SELECT, INSERT, UPDATE, DELETE, etc.
    For SELECT queries, it returns a list of rows. For other queries, it returns an empty list.
    :param database_path: The path to the SQLite database file.
    :param query: The SQL query to execute.
    :return: A list of tuples, where each tuple is a row from the query results.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        cursor.execute(query)

        # If the query is a SELECT statement, fetch the results
        if query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
        else:
            conn.commit()
            results = []

    except sqlite3.Error as e:
        conn.rollback()
        raise ValueError(f"Database error: {e}")
    finally:
        conn.close()

    return results

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

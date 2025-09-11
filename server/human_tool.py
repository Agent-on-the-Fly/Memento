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

mcp = FastMCP("human")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def ask_question(question: str, choices: list = []) -> str:
    """
    Asks the user a question and returns their answer.
    This will pause the agent and wait for human input from the console.
    :param question: The question to ask the user.
    :param choices: A list of choices for the user to select from.
    :return: The user's answer as a string.
    """
    print(f"\n[Human Input Required] {question}")

    if choices:
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")

        while True:
            try:
                answer = input(f"Enter your choice (1-{len(choices)}): ")
                choice_index = int(answer) - 1
                if 0 <= choice_index < len(choices):
                    return choices[choice_index]
                else:
                    print("Invalid choice, please try again.")
            except ValueError:
                print("Invalid input, please enter a number.")
    else:
        answer = input("Enter your answer: ")
        return answer

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

from mcp.server.fastmcp import FastMCP

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
    mcp.run(transport="stdio")

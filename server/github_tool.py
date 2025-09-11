import os
from github import Github
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("github")

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #

ACCESS_TOKEN = os.environ.get("GITHUB_TOKEN")
g = Github(ACCESS_TOKEN) if ACCESS_TOKEN else None

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def get_repo_tree(owner: str, repo: str, path: str = "") -> list:
    """
    Get the file and directory tree of a GitHub repository at a specific path.
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param path: The path to get the tree from. Defaults to the root.
    :return: A list of files and directories in the specified path.
    """
    if not g:
        raise ValueError("GitHub API token not configured. Please set the GITHUB_TOKEN environment variable.")
    repository = g.get_repo(f"{owner}/{repo}")
    contents = repository.get_contents(path)
    tree = []
    for content in contents:
        tree.append({"type": content.type, "path": content.path, "name": content.name})
    return tree


@mcp.tool()
async def read_file(owner: str, repo: str, path: str) -> str:
    """
    Read the content of a file in a GitHub repository.
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param path: The path to the file.
    :return: The content of the file as a string.
    """
    if not g:
        raise ValueError("GitHub API token not configured. Please set the GITHUB_TOKEN environment variable.")
    repository = g.get_repo(f"{owner}/{repo}")
    file_content = repository.get_contents(path)
    return file_content.decoded_content.decode("utf-8")


@mcp.tool()
async def create_issue(owner: str, repo: str, title: str, body: str = "") -> dict:
    """
    Create a new issue in a GitHub repository.
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param title: The title of the issue.
    :param body: The body content of the issue.
    :return: A dictionary with the issue number and URL.
    """
    if not g:
        raise ValueError("GitHub API token not configured. Please set the GITHUB_TOKEN environment variable.")
    repository = g.get_repo(f"{owner}/{repo}")
    issue = repository.create_issue(title=title, body=body)
    return {"number": issue.number, "url": issue.html_url}

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

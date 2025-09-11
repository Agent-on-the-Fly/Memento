import ast
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("code_optimizer")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def analyze_code_complexity(code: str) -> dict:
    """
    Analyzes the complexity of a piece of Python code using cyclomatic complexity.
    :param code: The Python code to analyze.
    :return: A dictionary with the complexity score for each function.
    """
    tree = ast.parse(code)
    results = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            complexity = 1
            for sub_node in ast.walk(node):
                if isinstance(sub_node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.With, ast.AsyncWith, ast.ExceptHandler)):
                    complexity += 1
            results[node.name] = complexity

    return results


@mcp.tool()
async def find_performance_bottlenecks(code: str) -> list:
    """
    Analyzes Python code to find potential performance bottlenecks.
    This is a static analysis and may not find all runtime bottlenecks.
    :param code: The Python code to analyze.
    :return: A list of potential bottlenecks.
    """
    tree = ast.parse(code)
    bottlenecks = []

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Check for nested loops
            for sub_node in ast.walk(node.body[0]):
                if isinstance(sub_node, ast.For):
                    bottlenecks.append(f"Potential performance bottleneck: Nested loop found at line {node.lineno}.")

    return bottlenecks if bottlenecks else ["No obvious performance bottlenecks found."]


@mcp.tool()
async def suggest_security_improvements(code: str) -> list:
    """
    Scans Python code for common security vulnerabilities and suggests improvements.
    :param code: The Python code to scan.
    :return: A list of security suggestions.
    """
    suggestions = []
    if "pickle.load" in code or "pickle.loads" in code:
        suggestions.append("Security suggestion: The 'pickle' module is not secure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Consider using a safer serialization format like JSON.")
    if "subprocess.call" in code and "shell=True" in code:
        suggestions.append("Security suggestion: Using 'shell=True' with 'subprocess' can be a security hazard if combined with untrusted input. Consider using 'shell=False' and passing arguments as a sequence.")
    if "eval(" in code:
        suggestions.append("Security suggestion: The 'eval' function is a security risk as it can execute arbitrary code. Avoid using it with untrusted input.")

    return suggestions if suggestions else ["No obvious security vulnerabilities found."]

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

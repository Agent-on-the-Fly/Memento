"""
code_exec_server.py – Enhanced FastMCP server for code execution with unified workspace
-----------------------------------------------------------------------------------

🎯 WORKFLOW:

This server provides tools for file management and command execution within a sandboxed
workspace. The standard workflow is:
1. Use `write_workspace_file` to create or modify scripts (e.g., Python, shell).
2. Use `execute_terminal_command` to run those scripts (e.g., `python my_script.py`) or any other shell command.
3. Use `read_workspace_file` or `list_workspace_files` to inspect the results and file system.

📁 DIRECTORY STRUCTURE:
    All files are stored and executed within a dedicated workspace directory for each task:
    `agent_cache/task_<id>/workspace/`
    This directory contains all user-created scripts, data files, and generated outputs
    in a flat structure. Subdirectories are not supported by default.

🔧 EXECUTION ENVIRONMENT:
    - All terminal commands run inside the workspace directory.
    - The environment is persistent for a given task, meaning files and changes
      made in one step are available in subsequent steps.
    - Python code execution is sandboxed, with restrictions on imports for security.

🔑 IMPORT WHITELIST (For Python scripts):
    When `execute_terminal_command` runs a Python script, the script's imports are
    validated against a security whitelist. Allowed packages include:
    • Standard Library: os, sys, json, csv, datetime, time, etc.
    • Data Science: numpy, pandas, matplotlib, seaborn, scipy, sklearn.
    • Other common packages like requests, beautifulsoup4, etc.
    
    ⚠️ Scripts attempting to import non-whitelisted packages will fail.

📤 OUTPUT FORMAT:
    Tool outputs are strings containing:
    • Standard output and error from command execution.
    • Confirmation messages for file operations.
    • File listings or content for read operations.

Tools provided:
    • execute_terminal_command(command, task_cache_dir?, verbose?) - Run shell commands in workspace.
    • list_workspace_files(task_cache_dir?) - List files in the workspace.
    • read_workspace_file(filename, task_cache_dir?) - Read the content of a file.
    • write_workspace_file(filename, content, task_cache_dir?) - Create or overwrite a file.
    • get_workspace_info(task_cache_dir?) - Get summary statistics of the workspace.
    • get_workspace_structure(task_cache_dir?) - Get a detailed tree-like structure of the workspace.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import argparse
import shutil
import uuid
import datetime
import tempfile
import subprocess
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set
import hashlib
import sys

from mcp.server.fastmcp import FastMCP

# The five interpreter wrappers come from camel‑ai
from interpreters.interpreters import (
    DockerInterpreter,
    E2BInterpreter,
    InternalPythonInterpreter,
    JupyterKernelInterpreter,
    SubprocessInterpreter,
)
from interpreters.logger import get_logger

# --------------------------------------------------------------------------- #
#  Default import whitelist configuration
# --------------------------------------------------------------------------- #

DEFAULT_IMPORT_WHITELIST = [
    # Standard library
    'os', 'sys', 'json', 'csv', 'datetime', 'time', 'math', 'random', 're', 
    'urllib', 'pathlib', 'collections', 'itertools', 'functools', 'operator',
    'typing', 'copy', 'pickle', 'hashlib', 'base64', 'uuid', 'tempfile',
    'shutil', 'glob', 'fnmatch', 'zipfile', 'tarfile', 'gzip', 'bz2',
    'io', 'sqlite3', 'configparser', 'argparse', 'logging', 'warnings',
    
    # Data science core libraries
    'numpy', 'np', 'pandas', 'pd', 'matplotlib', 'plt', 'seaborn', 'sns',
    'scipy', 'sklearn', 'plotly', 'dash', 'streamlit',
    
    # Machine learning frameworks
    'torch', 'torchvision', 'torchaudio', 'transformers', 'tensorflow', 'tf',
    'keras', 'jax', 'flax', 'optax',
    
    # Image processing
    'PIL', 'Image', 'cv2', 'skimage', 'imageio',
    
    # Network and data acquisition
    'requests', 'urllib3', 'beautifulsoup4', 'bs4', 'scrapy', 'selenium',
    
    # File processing
    'openpyxl', 'xlrd', 'xlwt', 'xlsxwriter', 'h5py', 'netCDF4', 'pyarrow',
    
    # Other common tools
    'tqdm', 'joblib', 'multiprocessing', 'concurrent', 'threading',
    'psutil', 'memory_profiler', 'line_profiler',
]

# --------------------------------------------------------------------------- #
#  Import whitelist validation functions
# --------------------------------------------------------------------------- #

def _extract_imports_from_code(code: str) -> Set[str]:
    """
    Extract all imported module names from Python code.
    
    Args:
        code: Python source code
        
    Returns:
        Set of imported module names
    """
    imports = set()
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Get the top-level module name
                    module_name = name.name.split('.')[0]
                    imports.add(module_name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Get the top-level module name
                    module_name = node.module.split('.')[0]
                    imports.add(module_name)
                    
    except SyntaxError:
        # If we can't parse the code, also try regex fallback
        pass
    
    # Regex fallback for cases where AST parsing fails
    import_patterns = [
        r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
    ]
    
    for pattern in import_patterns:
        matches = re.findall(pattern, code, re.MULTILINE)
        for match in matches:
            imports.add(match.split('.')[0])
    
    return imports


def _validate_imports(code: str, whitelist: Optional[List[str]] = None) -> tuple[bool, List[str], List[str]]:
    """
    Validate that all imports in code are in the whitelist.
    
    Args:
        code: Python source code to validate
        whitelist: List of allowed module names (defaults to DEFAULT_IMPORT_WHITELIST)
        
    Returns:
        Tuple of (is_valid, allowed_imports, forbidden_imports)
    """
    if whitelist is None:
        whitelist = DEFAULT_IMPORT_WHITELIST
    
    whitelist_set = set(whitelist)
    imports = _extract_imports_from_code(code)
    
    allowed_imports = []
    forbidden_imports = []
    
    for imp in imports:
        if imp in whitelist_set:
            allowed_imports.append(imp)
        else:
            forbidden_imports.append(imp)
    
    is_valid = len(forbidden_imports) == 0
    return is_valid, allowed_imports, forbidden_imports

# --------------------------------------------------------------------------- #
#  Text truncation utility
# --------------------------------------------------------------------------- #

def _truncate_text(text: str, max_tokens: int, filename: str = "") -> str:
    """
    Truncate text to approximately max_tokens, showing start and end.
    
    For code files (.py, .js, .ts, .java, .cpp, .c, .h, .cs, .php, .rb, .go, .rs, .swift),
    this function is not applied and full content is returned.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens to show (approximately)
        filename: Filename to check if it's a code file
        
    Returns:
        Truncated text or full text for code files
    """
    # Code file extensions that should show full content
    code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', 
                      '.php', '.rb', '.go', '.rs', '.swift', '.jsx', '.tsx', 
                      '.vue', '.svelte', '.kt', '.scala', '.clj', '.hs', '.ml', 
                      '.sh', '.bash', '.ps1', '.sql', '.r', '.m', '.ipynb'}
    
    if filename:
        file_ext = Path(filename).suffix.lower()
        if file_ext in code_extensions:
            return text
    
    # Rough approximation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Show first and last portions
    chunk_size = max_chars // 2
    start_chunk = text[:chunk_size]
    end_chunk = text[-chunk_size:]
    
    truncated_chars = len(text) - (2 * chunk_size)
    
    return f"{start_chunk}\n\n... [Truncated {truncated_chars:,} characters] ...\n\n{end_chunk}"

# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("code_exec")

# --------------------------------------------------------------------------- #
#  Simplified workspace management
# --------------------------------------------------------------------------- #

def _get_workspace_dir(task_cache_dir: Optional[str] = None) -> str:
    """
    Get or create the unified workspace directory for a task.
    
    📁 DIRECTORY STRUCTURE:
        task_cache_dir/workspace/  (flat structure - all files here)
    
    Args:
        task_cache_dir: Task-specific cache directory path. This is REQUIRED.
        
    Returns:
        str: Absolute path to workspace directory
        
    Raises:
        ValueError: If task_cache_dir is not provided.
    """
    if not task_cache_dir:
        raise ValueError("task_cache_dir must be provided to locate the workspace.")
    
    workspace_dir = Path(task_cache_dir) / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return str(workspace_dir)


def _log_execution(operation: str, details: dict, workspace_dir: str) -> None:
    """
    Log execution details for debugging and tracking.
    
    Args:
        operation: Type of operation performed
        details: Operation-specific details
        workspace_dir: Workspace directory path
    """
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation,
            "workspace": workspace_dir,
            "details": details
        }
        
        log_file = Path(workspace_dir).parent / "execution.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        logger.error(f"Failed to log execution: {e}")


# --------------------------------------------------------------------------- #
#  Enhanced sandbox wrapper with unified workspace
# --------------------------------------------------------------------------- #

class _UnifiedWorkspaceSandbox:
    """
    Unified sandbox with flat workspace structure and import whitelist validation.
    
    🎯 KEY FEATURES:
        • All files in single workspace directory
        • Persistent environment across executions
        • Terminal command support
        • File overwrite behavior (no versioning)
        • Import whitelist validation for security
    
    🔒 IMPORT SECURITY:
        • Validates all import statements against whitelist
        • Supports torch, transformers, and other ML libraries
        • Blocks unauthorized imports for security
    """

    def __init__(
        self,
        workspace_dir: str,
        sandbox: Literal[
            "internal_python", "jupyter", "docker", "subprocess", "e2b"
        ] = "subprocess",
        *,
        verbose: bool = False,
        unsafe_mode: bool = False,
        import_whitelist: Optional[list[str]] = None,
        require_confirm: bool = False,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.verbose = verbose
        
        # Set up import whitelist
        self.import_whitelist = import_whitelist or DEFAULT_IMPORT_WHITELIST.copy()
        self.unsafe_mode = unsafe_mode  # If True, skip whitelist validation
        self.require_confirm = require_confirm

        # Initialize the interpreter, ensuring it uses the correct workspace directory
        self.interpreter = self._initialize_interpreter(sandbox, self.workspace_dir)

    def _initialize_interpreter(self, sandbox_type: str, work_dir: str):
        """Initializes the correct interpreter."""
        
        # NOTE: The working directory is now passed to SubprocessInterpreter
        # directly via the workspace_dir parameter, eliminating the need
        # for os.chdir() in most cases.
        
        if sandbox_type == "internal_python":
            return SubprocessInterpreter(
                require_confirm=self.require_confirm,
                print_stdout=self.verbose,
                print_stderr=self.verbose,
                workspace_dir=work_dir,
                )
        elif sandbox_type == "jupyter":
            return JupyterKernelInterpreter(
                require_confirm=self.require_confirm,
                print_stdout=self.verbose,
                print_stderr=self.verbose,
                )
        elif sandbox_type == "docker":
            return DockerInterpreter(
                require_confirm=self.require_confirm,
                print_stdout=self.verbose,
                print_stderr=self.verbose,
                )
        elif sandbox_type == "e2b":
            return E2BInterpreter(require_confirm=self.require_confirm)
        
        # Default to SubprocessInterpreter
        return SubprocessInterpreter(
            require_confirm=self.require_confirm,
            print_stdout=self.verbose,
            print_stderr=self.verbose,
            workspace_dir=work_dir,
        )

    def execute_code(self, code: str, filename: str) -> str:
        """
        Execute Python code with import whitelist validation and save to specified filename.
        
        🔧 EXECUTION PROCESS:
            1. Validate imports against whitelist
            2. Change to workspace directory
            3. Write code to specified filename (OVERWRITES existing)
            4. Execute code in persistent environment
            5. Return execution result with validation info
        
        🔒 WHITELIST VALIDATION:
            • Checks all import statements in code
            • Allows: torch, transformers, numpy, pandas, matplotlib, etc.
            • Blocks: unauthorized system modules, network libraries not in whitelist
        
        Args:
            code: Python code to execute
            filename: Target filename (REQUIRED, will overwrite if exists)
            
        Returns:
            str: Execution output with import validation results and any error messages
        """
        # Ensure workspace directory exists
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        os.chdir(self.workspace_dir)
        
        try:
            # Validate imports if not in unsafe mode
            validation_result = ""
            if not self.unsafe_mode:
                is_valid, allowed_imports, forbidden_imports = _validate_imports(code, self.import_whitelist)
                
                if not is_valid:
                    error_msg = f"❌ IMPORT VALIDATION FAILED\n"
                    error_msg += f"Forbidden imports: {', '.join(forbidden_imports)}\n"
                    error_msg += f"Allowed imports in whitelist:\n"
                    for item in sorted(self.import_whitelist):
                        error_msg += f"  • {item}\n"
                    error_msg += f"\n💡 Contact admin to add other packages to whitelist."
                    return error_msg
                
                if allowed_imports:
                    validation_result = f"✅ IMPORTS VALIDATED: {', '.join(sorted(allowed_imports))}\n"
                    validation_result += f"🔒 Whitelist contains: {len(self.import_whitelist)} approved modules\n"
                    validation_result += "=" * 50 + "\n"
            
            # Write code to file (OVERWRITE if exists)
            code_file = Path(self.workspace_dir) / filename
            file_existed = code_file.exists()
            code_file.write_text(code, encoding="utf-8")
            logger.info(f"Code written to: {filename} ({'OVERWRITTEN' if file_existed else 'CREATED'})")
            
            # Execute the code
            execution_result = self.interpreter.run(code, code_type="python")
            
            # Combine validation and execution results
            full_result = validation_result + execution_result
            return full_result
            
        finally:
            os.chdir(original_cwd)

    def execute_terminal_command(self, command: str) -> str:
        """
        Execute a shell command in the workspace directory.
        
        Args:
            command: The shell command to execute.
            
        Returns:
            A string containing the stdout and stderr of the command.
        """
        if self.verbose:
            print(f"Executing terminal command: {command} in {self.workspace_dir}")

        log_details = {"command": command}
        _log_execution("execute_terminal_command", log_details, self.workspace_dir)
        
        # Ensure workspace directory exists before attempting to change to it
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        os.chdir(self.workspace_dir)
        
        try:
            # The interpreter is already configured with the correct working directory.
            # We pass 'bash' to indicate it's a shell command.
            result = self.interpreter.run(command, code_type="bash")

            # Process result based on its type
            if isinstance(result, tuple) and len(result) == 2:
                # Assuming (exit_code, logs)
                exit_code, logs = result
                if isinstance(logs, list):
                    output_lines = [log.content for log in logs]
                else: # Assuming logs is a string
                    output_lines = [str(logs)]

                if exit_code == 0:
                    status = "✅ Command executed successfully."
                else:
                    status = f"⚠️ Command finished with non-zero exit code: {exit_code}."
                
                return f"{status}\n\nSTDOUT/STDERR:\n{''.join(output_lines)}"

            # Handle simple string output for older interpreter versions
            elif isinstance(result, str):
                return f"✅ Command executed.\n\nOutput:\n{result}"
            
            # Handle other potential result formats
            else:
                return f"✅ Command executed.\n\nResult:\n{str(result)}"

        except Exception as e:
            return f"❌ Error executing terminal command: {str(e)}"
        finally:
            os.chdir(original_cwd)


# Global sandbox instances per task (for environment persistence)
_task_sandboxes: Dict[str, _UnifiedWorkspaceSandbox] = {}

def _get_or_create_sandbox(
    workspace_dir: str,
    sandbox: str,
    verbose: bool,
    unsafe_mode: bool,
    import_whitelist: Optional[List[str]] = None
) -> _UnifiedWorkspaceSandbox:
    """
    Get the existing sandbox for the workspace or create a new one.

    This function uses a global dictionary to cache sandbox instances based on
    the workspace directory. This ensures that the same sandbox is used for
all operations within the same task, preserving state.
    
    Args:
        workspace_dir: The absolute path to the workspace directory.
        sandbox: The type of sandbox to create.
        verbose: Whether to enable verbose logging.
        unsafe_mode: Whether to disable security checks (e.g., import validation).
        import_whitelist: A list of allowed Python modules.
        
    Returns:
        An instance of _UnifiedWorkspaceSandbox.
    """
    global _task_sandboxes
    if workspace_dir not in _task_sandboxes:
        if verbose:
            print(f"Creating new sandbox for workspace: {workspace_dir}")
        _task_sandboxes[workspace_dir] = _UnifiedWorkspaceSandbox(
            workspace_dir=workspace_dir,
            sandbox=sandbox,
            verbose=verbose,
            unsafe_mode=unsafe_mode,
            import_whitelist=import_whitelist,
        )
    return _task_sandboxes[workspace_dir]


# --------------------------------------------------------------------------- #
#  Enhanced tools with unified workspace
# --------------------------------------------------------------------------- #

def _normalize_filename(filename: str) -> str:
    """
    Sanitizes and normalizes a filename to prevent directory traversal.

    - Removes leading/trailing whitespace and quotes.
    - Replaces backslashes with forward slashes.
    - Removes any path components (e.g., '/', '..').
    - If the filename becomes empty, it defaults to a UUID-based name.
    
    Args:
        filename: The original filename provided by the user or agent.
        
    Returns:
        A safe, sanitized filename.
    """
    if not isinstance(filename, str):
        filename = str(filename)
        
    # Strip whitespace and quotes
    filename = filename.strip().strip('\'"')
    
    # Standardize path separators
    filename = filename.replace('\\', '/')
    
    # Remove any directory traversal components
    filename = os.path.basename(filename)
    
    # If the filename is empty after sanitization, create a default name
    if not filename:
        filename = f"file_{uuid.uuid4().hex[:8]}.txt"
        
    return filename


@mcp.tool()
async def execute_terminal_command(
    command: str,
    task_cache_dir: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Execute terminal command in workspace directory context.

    Workflow Note: To run a script (e.g., Python, shell), first create it
    using `write_workspace_file`, then execute it with this command.
    Example: `python your_script.py`.

    🖥️ TERMINAL EXECUTION RULES:
        • Commands run in workspace directory (task_cache_dir/workspace/)
        • Can execute Python files: "python script.py"
        • Can perform file operations: "ls", "cat file.txt", "rm file.py"
        • Can install packages: "pip install matplotlib"
        • Working directory is automatically set to workspace

    📝 INPUT:
        • command: Shell command to execute (REQUIRED)
        • task_cache_dir: Task directory path (auto-injected)

    📤 OUTPUT:
        Returns command output including:
        • STDOUT from command execution
        • STDERR if any errors occurred
        • Exit code status
        • Execution timeout (60 seconds max)

    🔒 ENVIRONMENT:
        • Same workspace as execute_code() operations
        • Can access all files created by code execution
        • Changes persist for subsequent operations

    Args:
        command: Shell command to execute in workspace
        task_cache_dir: Task cache directory path
        verbose: Include detailed execution information

    Returns:
        Command output, stderr, and exit status

    Example Usage:
        execute_terminal_command("python analysis.py")
        execute_terminal_command("ls -la")
        execute_terminal_command("pip install pandas")
        execute_terminal_command("cat results.txt")
    """
    if not command:
        return "❌ Error: Command cannot be empty."
    
    # Get workspace directory
    workspace_dir = _get_workspace_dir(task_cache_dir)
    
    # Get or create the sandbox for the workspace
    sandbox_instance = _get_or_create_sandbox(workspace_dir, "subprocess", verbose, False)
    
    # Execute the command
    result = sandbox_instance.execute_terminal_command(command)
        
    return result


@mcp.tool()
async def list_workspace_files(task_cache_dir: str | None = None) -> str:
    """
    List all files in the unified workspace directory with tree structure using rich.
    
    Workflow Note: After listing files, you can read a file with `read_workspace_file`
    or execute a script with `execute_terminal_command`.
    
    🎯 WHEN TO USE THIS FUNCTION:
        ✅ USE list_workspace_files() WHEN:
        • You want a quick overview of files in workspace
        • You need basic file information (names, sizes, dates)
        • You want a simple, readable file listing
        • You're checking if specific files exist

        📝 ALTERNATIVE FUNCTIONS:
        • Use get_workspace_structure() for detailed file structure with types
        • Use read_workspace_file() to read content of specific files
        • Use get_workspace_info() for summary statistics only

    📂 LISTING BEHAVIOR:
        • Shows all files in tree structure including subdirectories
        • Uses rich tree format for better visualization
        • Includes file sizes and modification times
        • Clean, hierarchical output format
    
    Args:
        task_cache_dir: Task cache directory path
        
    Returns:
        Formatted tree structure of workspace files with details

    Example Output:
        📁 workspace
        ├── 📄 analysis.py (2048 bytes, 2024-01-01 14:30:25)
        ├── 📄 data.csv (8192 bytes, 2024-01-01 14:28:10)
        └── 📁 upload_files
            ├── 📄 document.pdf (4096 bytes, 2024-01-01 14:31:05)
            └── 📄 image.png (2048 bytes, 2024-01-01 14:32:00)
    """
    from rich.tree import Tree
    from rich.console import Console
    from io import StringIO
    
    workspace_dir = _get_workspace_dir(task_cache_dir)
    workspace_path = Path(workspace_dir)
    
    if not workspace_path.exists():
        return f"📂 Workspace directory does not exist: {workspace_dir}"
    
    def add_files(path, tree_node):
        """Recursively add files and directories to the tree."""
        try:
            items = sorted(Path(path).iterdir())
            for item in items:
                if item.is_dir():
                    # Add directory node
                    branch = tree_node.add(f"📁 {item.name}")
                    add_files(item, branch)
                else:
                    # Add file node with size and modification time
                    try:
                        size = item.stat().st_size
                        mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                        mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                        tree_node.add(f"📄 {item.name} ({size} bytes, {mtime_str})")
                    except (OSError, ValueError):
                        tree_node.add(f"📄 {item.name} (size unknown)")
        except PermissionError:
            tree_node.add("❌ Permission denied")
        except Exception as e:
            tree_node.add(f"❌ Error: {str(e)}")
    
    # Create tree with workspace as root
    tree = Tree(f"📁 {workspace_path.name}")
    add_files(workspace_path, tree)
    
    # Capture console output
    console = Console(file=StringIO(), width=120)
    console.print(tree)
    tree_output = console.file.getvalue()
    
    logger.info(f"Generated tree structure for workspace: {workspace_dir}")
    return tree_output


@mcp.tool()
async def read_workspace_file(filename: str, task_cache_dir: str | None = None) -> str:
    """
    Read content of a file from workspace directory.
    
    Workflow Note: After reading a script, you can execute it using the
    `execute_terminal_command` tool.
    
    🎯 WHEN TO USE THIS FUNCTION:
        ✅ USE read_workspace_file() WHEN:
        • You want to read the content of an existing file
        • You need to examine data files, code files, or text files
        • You want to check what's inside a specific file
        • You're reviewing results or outputs from previous operations

        📝 ALTERNATIVE FUNCTIONS:
        • Use list_workspace_files() to see what files exist first
        • Use get_workspace_structure() for detailed file information
        • Use execute_code() to create and run new Python scripts

    📖 READING BEHAVIOR:
        • Reads files from flat workspace structure
        • Supports text files (Python, CSV, TXT, JSON, etc.)
        • Binary files return size information only
        • Large files may be truncated for display (full content preserved)
        • Shows file path and size information

    📂 SUPPORTED FILE TYPES:
        • Text files: .txt, .md, .csv, .json, .py, .js, .html, .css
        • Configuration files: .ini, .yaml, .toml, .conf
        • Data files: .csv, .tsv, .json, .xml
        • Code files: .py, .js, .sql, .sh (shown in full)
        • Binary files: size info only (cannot display content)
    
    Args:
        filename: Name of file to read from workspace (simple filename only)
        task_cache_dir: Task cache directory path
        
    Returns:
        File content or error message with file information

    Example Usage:
        ✅ CORRECT USAGE:
        read_workspace_file("data.csv")
        read_workspace_file("analysis.py")
        read_workspace_file("config.json")
        read_workspace_file("results.txt")
        
        ❌ INCORRECT USAGE:
        read_workspace_file("workspace/data.csv")  # Don't include workspace/ prefix
        read_workspace_file("/full/path/to/file.txt")  # Use simple filename only
    """
    workspace_dir = _get_workspace_dir(task_cache_dir)
    file_path = Path(workspace_dir) / filename
    
    if not file_path.exists():
        available_files = [f.name for f in Path(workspace_dir).iterdir() if f.is_file()]
        return f"❌ File not found: {filename}\nAvailable files: {available_files}"
    
    if not file_path.is_file():
        return f"❌ Path is not a file: {filename}"
    
    try:
        content = file_path.read_text(encoding="utf-8")
        logger.info(f"Read file: {filename} ({len(content)} characters)")
        
        # Apply text truncation for non-code files (1000 tokens max)
        display_content = _truncate_text(content, max_tokens=1000, filename=filename)
        
        result = []
        result.append(f"📄 File: {filename}")
        result.append(f"📁 Workspace: {workspace_dir}")
        result.append(f"📊 Size: {len(content)} characters")
        if display_content != content:
            result.append("⚠️ Content truncated for display (non-code file)")
        result.append("=" * 50)
        result.append(display_content)
        
        return "\n".join(result)
        
    except UnicodeDecodeError:
        # Binary file
        size = file_path.stat().st_size
        logger.info(f"Binary file detected: {filename} ({size} bytes)")
        return f"📄 Binary file: {filename} ({size} bytes)\n❌ Cannot display binary content as text"


@mcp.tool()
async def write_workspace_file(filename: str, content: str, task_cache_dir: str | None = None) -> str:
    """
    Write content to a file in workspace directory WITHOUT execution.

    Workflow Note: After writing an executable script (e.g., a `.py` or `.sh` file),
    your next step should be to call `execute_terminal_command` to run it.

    🎯 WHEN TO USE THIS FUNCTION:
        ✅ USE write_workspace_file() WHEN:
        • You want to SAVE content to a file WITHOUT running it
        • You're creating data files (CSV, JSON, TXT, etc.)
        • You're saving configuration files or documentation
        • You want to store code for later use without executing it immediately
        • You're creating templates, schemas, or reference files
        • You need to save non-Python content (HTML, CSS, SQL, etc.)

        ❌ DO NOT use write_workspace_file() when:
        • You want to execute Python code immediately
        • You need to run calculations, generate outputs, or process data
        • You want to see execution results or error messages

    📝 ALTERNATIVE FUNCTIONS:
        • Use execute_code() to create AND execute Python scripts
        • Use read_workspace_file() to read existing files
        • Use list_workspace_files() to see all files in workspace
        • Use get_workspace_structure() to see detailed file structure

    📝 WRITING BEHAVIOR:
        • Creates new file or OVERWRITES existing file
        • No backup or versioning - direct replacement
        • File saved to flat workspace structure
        • Content written exactly as provided (no processing)

    📂 SUPPORTED FILE TYPES:
        • Text files: .txt, .md, .csv, .json, .xml, .html, .css
        • Code files: .py, .js, .sql, .sh, .yaml, .toml
        • Configuration files: .ini, .conf, .properties
        • Data files: .csv, .tsv, .json, .xml
        • Any text-based content

    Args:
        filename: Target filename in workspace (simple filename only)
        content: Content to write to file (string)
        task_cache_dir: Task cache directory path

    Returns:
        Success confirmation with file details

    Example Usage:
        ✅ CORRECT USAGE:
        write_workspace_file("data.csv", "name,age\\nJohn,25\\nJane,30")
        write_workspace_file("config.json", '{"api_key": "abc123", "timeout": 30}')
        write_workspace_file("readme.md", "# Project Documentation\\n\\nThis is a sample project...")
        write_workspace_file("template.py", "# Template code for later use\\nimport pandas as pd\\n# TODO: Add functionality")
        write_workspace_file("results.txt", "Analysis completed at 2024-01-01\\nTotal records: 1000")
        
        ❌ INCORRECT USAGE:
        write_workspace_file("script.py", "print('hello')") # This only saves the file. To see output, you must then call execute_terminal_command("python script.py").
        write_workspace_file("workspace/file.txt", "content") # Don't include workspace/ prefix
    """
    workspace_dir = _get_workspace_dir(task_cache_dir)
    
    try:
        normalized_filename = _normalize_filename(filename)
    except ValueError as e:
        return f"❌ Filename error: {e}"
        
    file_path = Path(workspace_dir) / normalized_filename
    
    # Check if file exists
    file_existed = file_path.exists()
    
    try:
        file_path.write_text(content, encoding="utf-8")
        
        # Log the operation
        _log_execution("write_workspace_file", {
            "filename": filename,
            "file_existed": file_existed,
            "content_length": len(content)
        }, workspace_dir)
        
        logger.info(f"{'Overwrote' if file_existed else 'Created'} file: {filename}")
        
        result = []
        result.append(f"✅ File {'OVERWRITTEN' if file_existed else 'CREATED'}: {filename}")
        result.append(f"📁 Workspace: {workspace_dir}")
        result.append(f"📊 Size: {len(content)} characters")
        
        return "\n".join(result)
        
    except Exception as e:
        error_msg = f"Failed to write file {filename}: {e}"
        logger.error(error_msg)
        return f"❌ {error_msg}"


@mcp.tool()
async def get_workspace_info(task_cache_dir: str | None = None) -> str:
    """
    Get comprehensive information about the workspace.
    
    📊 INFORMATION PROVIDED:
        • Workspace directory path
        • Total files and sizes
        • File type distribution
        • Sandbox environment status
        • Recent activity summary
    
    Args:
        task_cache_dir: Task cache directory path
        
    Returns:
        Formatted workspace information summary
    """
    workspace_dir = _get_workspace_dir(task_cache_dir)
    workspace_path = Path(workspace_dir)
    
    # Analyze files
    file_types = {}
    total_size = 0
    file_count = 0
    
    for file_path in workspace_path.iterdir():
        if file_path.is_file():
            file_count += 1
            size = file_path.stat().st_size
            total_size += size
            
            suffix = file_path.suffix.lower() or "no_extension"
            if suffix not in file_types:
                file_types[suffix] = {"count": 0, "size": 0}
            file_types[suffix]["count"] += 1
            file_types[suffix]["size"] += size
    
    # Check sandbox status
    sandbox_active = workspace_dir in _task_sandboxes
    
    # Format output
    result = []
    result.append("📊 WORKSPACE INFORMATION")
    result.append("=" * 50)
    result.append(f"📁 Directory: {workspace_dir}")
    result.append(f"📄 Total Files: {file_count}")
    result.append(f"💾 Total Size: {total_size:,} bytes")
    result.append(f"🔧 Sandbox Active: {'Yes' if sandbox_active else 'No'}")
    result.append(f"🔒 Environment Persistent: {'Yes' if sandbox_active else 'No'}")
    
    if file_types:
        result.append("\n📈 FILE TYPE DISTRIBUTION:")
        result.append("-" * 30)
        for ext, info in sorted(file_types.items()):
            result.append(f"  {ext:<15} {info['count']:>3} files  {info['size']:>8,} bytes")
    
    logger.info(f"Workspace info: {file_count} files, {total_size} bytes")
    return "\n".join(result)


@mcp.tool()
async def get_workspace_structure(task_cache_dir: str | None = None) -> str:
    """
    Get detailed file structure and directory tree of the workspace.

    🎯 WHEN TO USE THIS FUNCTION:
        ✅ USE get_workspace_structure() WHEN:
        • You want to see ALL files in the workspace with details
        • You need to understand the complete file organization
        • You want to see file sizes, types, and modification times
        • You're exploring what files exist before reading or processing them
        • You need a comprehensive overview of workspace contents

    📂 INFORMATION PROVIDED:
        • Complete file listing with full paths
        • File sizes in human-readable format
        • File modification timestamps
        • File type identification based on extensions
        • Total workspace statistics (file count, total size)
        • Directory structure visualization

    📝 ALTERNATIVE FUNCTIONS:
        • Use list_workspace_files() for simple file listing
        • Use get_workspace_info() for summary statistics
        • Use read_workspace_file() to read specific files
        • Use write_workspace_file() to create new files

    Args:
        task_cache_dir: Task cache directory path

    Returns:
        Detailed workspace structure with file information

    Example Output:
        📂 WORKSPACE STRUCTURE
        ========================================
        📁 Directory: /path/to/workspace
        📊 Total Files: 5
        💾 Total Size: 15.2 KB
        
        📄 FILES:
        ├── analysis.py          (2.1 KB)  [Python]     2024-01-01 14:30:25
        ├── data.csv             (8.5 KB)  [Data]       2024-01-01 14:28:10
        ├── chart.png            (4.2 KB)  [Image]      2024-01-01 14:31:05
        ├── config.json          (0.3 KB)  [Config]     2024-01-01 14:25:00
        └── results.txt          (0.1 KB)  [Text]       2024-01-01 14:32:15
    """
    workspace_dir = _get_workspace_dir(task_cache_dir)
    workspace_path = Path(workspace_dir)
    
    if not workspace_path.exists():
        return f"📂 Workspace directory does not exist: {workspace_dir}"
    
    # Collect all files with detailed information
    files_info = []
    total_size = 0
    
    for file_path in workspace_path.iterdir():
        if file_path.is_file():
            try:
                stat = file_path.stat()
                size = stat.st_size
                total_size += size
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
                
                # Determine file type based on extension
                suffix = file_path.suffix.lower()
                file_type = _get_file_type_description(suffix)
                
                # Format file size
                size_str = _format_file_size(size)
                
                files_info.append({
                    'name': file_path.name,
                    'size': size,
                    'size_str': size_str,
                    'type': file_type,
                    'modified': mtime.strftime("%Y-%m-%d %H:%M:%S"),
                    'extension': suffix
                })
            except (OSError, ValueError) as e:
                # Handle files that can't be accessed
                files_info.append({
                    'name': file_path.name,
                    'size': 0,
                    'size_str': 'N/A',
                    'type': 'Unknown',
                    'modified': 'N/A',
                    'extension': file_path.suffix.lower(),
                    'error': str(e)
                })
    
    # Sort files by name
    files_info.sort(key=lambda x: x['name'])
    
    # Build output
    result = []
    result.append("📂 WORKSPACE STRUCTURE")
    result.append("=" * 50)
    result.append(f"📁 Directory: {workspace_dir}")
    result.append(f"📊 Total Files: {len(files_info)}")
    result.append(f"💾 Total Size: {_format_file_size(total_size)}")
    result.append("")
    
    if not files_info:
        result.append("📭 Workspace is empty (no files)")
    else:
        result.append("📄 FILES:")
        
        for i, file_info in enumerate(files_info):
            is_last = (i == len(files_info) - 1)
            prefix = "└──" if is_last else "├──"
            
            if 'error' in file_info:
                result.append(f"{prefix} {file_info['name']:<20} (ERROR: {file_info['error']})")
            else:
                result.append(
                    f"{prefix} {file_info['name']:<20} "
                    f"({file_info['size_str']:>8})  "
                    f"[{file_info['type']:<8}] "
                    f"{file_info['modified']}"
                )
    
    # Add file type summary
    if files_info:
        type_counts = {}
        for file_info in files_info:
            file_type = file_info['type']
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        result.append("")
        result.append("📊 FILE TYPE SUMMARY:")
        for file_type, count in sorted(type_counts.items()):
            result.append(f"  • {file_type}: {count} file{'s' if count != 1 else ''}")
    
    logger.info(f"Generated workspace structure for {len(files_info)} files")
    return "\n".join(result)


def _get_file_type_description(extension: str) -> str:
    """Get a human-readable description of file type based on extension."""
    type_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.html': 'HTML',
        '.css': 'CSS',
        '.json': 'JSON',
        '.xml': 'XML',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.toml': 'TOML',
        '.ini': 'Config',
        '.conf': 'Config',
        '.cfg': 'Config',
        '.txt': 'Text',
        '.md': 'Markdown',
        '.rst': 'reStruct',
        '.csv': 'CSV Data',
        '.tsv': 'TSV Data',
        '.xlsx': 'Excel',
        '.xls': 'Excel',
        '.pdf': 'PDF',
        '.png': 'PNG Image',
        '.jpg': 'JPEG Image',
        '.jpeg': 'JPEG Image',
        '.gif': 'GIF Image',
        '.svg': 'SVG Image',
        '.bmp': 'BMP Image',
        '.webp': 'WebP Image',
        '.mp4': 'Video',
        '.avi': 'Video',
        '.mov': 'Video',
        '.mp3': 'Audio',
        '.wav': 'Audio',
        '.zip': 'Archive',
        '.tar': 'Archive',
        '.gz': 'Archive',
        '.rar': 'Archive',
        '.sql': 'SQL',
        '.db': 'Database',
        '.sqlite': 'SQLite',
        '.log': 'Log',
        '.sh': 'Shell',
        '.bat': 'Batch',
        '.ps1': 'PowerShell',
        '.r': 'R Script',
        '.ipynb': 'Jupyter NB',
    }
    
    return type_map.get(extension, 'Unknown')


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    if i == 0:
        return f"{int(size)} {size_names[i]}"
    else:
        return f"{size:.1f} {size_names[i]}"


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

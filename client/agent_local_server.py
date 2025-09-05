"""
AgentFly - A Hierarchical AI Agent System

This module implements a hierarchical AI system with two main components:
1. META-PLANNER: Breaks down high-level questions into executable tasks
2. EXECUTOR: Executes individual tasks using available tools

The system uses OpenAI models and MCP (Model Context Protocol) for tool integration.
"""

from __future__ import annotations
import asyncio
import argparse
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import colorlog
import json
import tiktoken

# ---------------------------------------------------------------------------
#   Logging setup
# ---------------------------------------------------------------------------
# Configure colored logging for better visibility of log levels
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   Constants & templates
# ---------------------------------------------------------------------------
# System prompt for the meta-planner agent that breaks down complex problems
META_SYSTEM_PROMPT = (
    "You are the META‑PLANNER in a hierarchical AI system. A user will ask a\n"
    "high‑level question. **First**: break the problem into a *minimal sequence*\n"
    "of executable tasks. Reply ONLY in JSON with the schema:\n"
    "{ \"plan\": [ {\"id\": INT, \"description\": STRING} … ] }\n\n"
    "After each task is executed by the EXECUTOR you will receive its result.\n"
    "Please carefully consider the descriptions of the time of web pages and events in the task, and take these factors into account when planning and giving the final answer.\n"
    "If the final answer is complete, output it with the template:\n"
    "FINAL ANSWER: <answer>\n\n" \
    " YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
    "Please ensure that the final answer strictly follows the question requirements, without any additional analysis.\n"
    "If the final ansert is not complete, emit a *new* JSON plan for the remaining work. Keep cycles as\n"
    "few as possible. Never call tools yourself — that's the EXECUTOR's job."\
    "⚠️  Reply with *pure JSON only*."
)

# System prompt for the executor agent that handles individual tasks
EXEC_SYSTEM_PROMPT = (
    "You are the EXECUTOR sub-agent. You receive one task description at a time\n"
    "from the meta-planner. Your job is to complete the task, using available\n"
    "tools via function calling if needed. Always think step by step but reply\n"
    "with the minimal content needed for the meta-planner. If you must call a\n"
    "tool, produce the appropriate function call instead of natural language.\n"
    "When done, output a concise result. Do NOT output FINAL ANSWER."
)

# Maximum context length for token management
MAX_CTX = 175000
# Default executor model
EXE_MODEL = "qwen3-8b"

# ---------------------------------------------------------------------------
#   OpenAI backend
# ---------------------------------------------------------------------------
class ChatBackend:
    """Abstract base class for chat backends."""
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIBackend(ChatBackend):
    """OpenAI API backend for chat completions with retry logic."""
    
    def __init__(self, model: str):
        """
        Initialize OpenAI backend with specified model.
        
        Args:
            model: The OpenAI model to use (e.g., 'gpt-4', 'o3')
        """
        self.model = model
        # Initialize OpenAI client with API key and base URL from environment
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
    ) -> Dict[str, Any]:
        """
        Send chat completion request to OpenAI with optional tool calling.
        
        Args:
            messages: List of message dictionaries with role and content
            tools: Optional list of available tools for function calling
            tool_choice: How to handle tool selection ('auto', 'none', or specific tool)
            max_tokens: Maximum tokens in the response
            
        Returns:
            Dictionary containing response content and tool calls if any
            
        Raises:
            Various OpenAI API errors (handled by retry decorator)
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Add tools to payload if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            
        # Make API call to OpenAI
        resp = await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        
        # Extract tool calls if present
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
            # Convert tool calls to standardized format
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}

class DirectModelBackend(ChatBackend):
    """
    Backend that talks to a local OpenAI-compatible server (e.g., your vLLM at
    http://localhost:port/v1) and returns tool_calls parsed from the stream.
    """

    def __init__(
        self,
        model: str = "qwen3-8b",
        server_url: str | None = None,
        api_key: str | None = None,
        default_search_tool: bool = True,
    ):
        self.model = model
        self.server_url = server_url or os.getenv("DIRECT_BASE_URL", "http://localhost:port/v1")
        self.api_key = api_key or os.getenv("DIRECT_API_KEY", "EMPTY")
        self.default_search_tool = default_search_tool

        self._fallback_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Get external knowledge using search engine",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 10240,
    ) -> Dict[str, Any]:
        
        client = AsyncOpenAI(base_url=self.server_url, api_key=self.api_key)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools and len(tools) > 0:
            payload["tools"] = tools
        elif self.default_search_tool:
            payload["tools"] = self._fallback_tools

        payload["tool_choice"] = tool_choice or "auto"

        stream = await client.chat.completions.create(**payload)

        aggregated_text = ""
        tool_call_buffers: Dict[int, Dict[str, Any]] = {}
        finish_reason = None

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            if getattr(delta, "content", None):
                aggregated_text += delta.content

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = getattr(tc, "index", 0) or 0
                    b = tool_call_buffers.setdefault(
                        idx,
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if getattr(tc, "id", None):
                        b["id"] = tc.id
                    func = getattr(tc, "function", None)
                    if func is not None:
                        if getattr(func, "name", None):
                            b["function"]["name"] = func.name
                        if getattr(func, "arguments", None):
                            b["function"]["arguments"] += func.arguments

        tool_calls = list(tool_call_buffers.values()) if tool_call_buffers else None

        content = None if tool_calls else (aggregated_text or None)
        return {"content": content, "tool_calls": tool_calls}

# ---------------------------------------------------------------------------
#   Hierarchical client (trimmed: only essentials kept)
# ---------------------------------------------------------------------------
# Maximum number of conversation turns to keep in memory
MAX_TURNS_MEMORY = 50

def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences and extract JSON content.
    
    Args:
        text: Text that may contain markdown fences or JSON
        
    Returns:
        Cleaned text with fences removed
    """
    import re
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    # Extract JSON content if wrapped in braces
    m = re.search(r"{[\\s\\S]*}", text)
    return m.group(0) if m else text

def _count_tokens(msg: Dict[str, str], enc) -> int:
    """
    Count tokens in a message for context management.
    
    Args:
        msg: Message dictionary with role and content
        enc: Tokenizer encoding object
        
    Returns:
        Number of tokens in the message
    """
    role_tokens = 4  # OpenAI adds 4 tokens for role
    content = msg.get("content") or ""
    return role_tokens + len(enc.encode(content))

def _get_tokenizer(model: str):
    """
    Return a tokenizer for the specified model.
    
    Args:
        model: Model name to get tokenizer for
        
    Returns:
        Tokenizer encoding object, falls back to cl100k_base if model unknown
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
    
def trim_messages(messages: List[Dict[str, str]], max_tokens: int, model="gpt-3.5-turbo"):
    """
    Trim message history to fit within token limit while preserving system message.
    
    Args:
        messages: List of message dictionaries
        max_tokens: Maximum allowed tokens
        model: Model name for token counting
        
    Returns:
        Trimmed list of messages that fit within token limit
    """
    enc = _get_tokenizer(model)
    total = sum(_count_tokens(m, enc) for m in messages) + 2
    
    # If already within limit, return as is
    if total <= max_tokens:
        return messages
        
    # Always keep system message (first message)
    system_msg = messages[0]
    kept: List[Dict[str, str]] = [system_msg]
    total = _count_tokens(system_msg, enc) + 2
    
    # Add messages from most recent to oldest until limit is reached
    for msg in reversed(messages[1:]):
        t = _count_tokens(msg, enc)
        if total + t > max_tokens:
            break
        kept.insert(1, msg)  # Insert after system message
        total += t
    return kept

class HierarchicalClient:
    """
    Main client class that orchestrates the hierarchical AI system.
    
    Manages communication between meta-planner and executor agents,
    handles tool connections, and processes user queries through
    multiple planning and execution cycles.
    """
    
    # Maximum number of planning cycles before giving up
    MAX_CYCLES = 3

    def __init__(self, meta_model: str, exec_model: str):
        """
        Initialize the hierarchical client.
        
        Args:
            meta_model: Model name for the meta-planner agent
            exec_model: Model name for the executor agent
        """
        self.meta_llm = OpenAIBackend(meta_model)

        if exec_model.lower().startswith("qwen3"):
            self.exec_llm = DirectModelBackend(model=exec_model)
        else:
            self.exec_llm = OpenAIBackend(exec_model)

        self.exec_model = exec_model  
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, str]] = []

    def _resolve_tool_name(self, requested: str) -> str:
        if requested in self.sessions:
            return requested

        aliases = {
            "search": ["serp_search"],
        }
        for cand in aliases.get(requested, []):
            if cand in self.sessions:
                return cand

        for name in self.sessions.keys():
            if requested in name or name in requested:
                return name

        raise KeyError(f"No matching tool for '{requested}'. Available: {list(self.sessions.keys())}")

    def _massage_args_for_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        patched = dict(args)
        ln = tool_name.lower()
        if ln == "serp_search":
            if "q" not in patched and "query" in patched:
                patched["q"] = patched["query"]
            if "num_results" not in patched and "top_k" in patched:
                patched["num_results"] = patched["top_k"]
        return patched

    # ---------- Tool management ----------
    async def connect_to_servers(self, scripts: List[str]):
        """
        Connect to MCP tool servers specified by script paths.
        
        Args:
            scripts: List of paths to tool server scripts
            
        Raises:
            RuntimeError: If duplicate tool names are found
        """
        from contextlib import AsyncExitStack
        self.exit_stack = AsyncExitStack()
        
        for script in scripts:
            path = Path(script)
            # Determine command based on file extension
            cmd = "python" if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            
            # Create stdio client and session
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            
            # Register tools from this session
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session
                
        print("Connected tools:", list(self.sessions.keys()))

    async def _tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema for all available tools in a format suitable for OpenAI.
        
        Returns:
            List of tool schemas in OpenAI function calling format
        """
        result, cached = [], {}
        for session in self.sessions.values():
            # Cache tool listings to avoid repeated calls
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            
            # Convert MCP tool format to OpenAI function calling format
            for tool in tools_resp.tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return result

    # ---------- Main processing ----------
    async def process_query(self, query: str, file: str, task_id: str = "interactive") -> str:
        """
        Process a user query through the hierarchical AI system.
        
        This is the main method that:
        1. Gets the meta-planner to break down the query into tasks
        2. Executes each task using the executor agent
        3. Continues planning cycles until a final answer is reached
        
        Args:
            query: User's question or request
            file: Optional file path context
            task_id: Unique identifier for this query session
            
        Returns:
            Final answer to the user's query
        """
        tools_schema = await self._tools_schema()
        self.shared_history = []
        
        # Initialize conversation with user query
        self.shared_history.append({
            "role": "user", 
            "content": f"{query}\ntask_id: {task_id}\nfile_path: {file}\n"
        })
        planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history

        # Main planning and execution loop
        for cycle in range(self.MAX_CYCLES):
            # Get plan from meta-planner
            meta_reply = await self.meta_llm.chat(planner_msgs)
            meta_content = meta_reply["content"] or ""
            self.shared_history.append({"role": "assistant", "content": meta_content})

            # Check if we have a final answer
            if meta_content.startswith("FINAL ANSWER:"):
                return meta_content[len("FINAL ANSWER:"):].strip()

            # Parse the plan from meta-planner's response
            try:
                tasks = json.loads(_strip_fences(meta_content))["plan"]
            except Exception as e:
                return f"[planner error] {e}: {meta_content}"

            # Execute each task in the plan
            for task in tasks:
                task_desc = f"Task {task['id']}: {task['description']}"
                exec_msgs = (
                    [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] +
                    self.shared_history +
                    [{"role": "user", "content": task_desc}]
                )
                
                # Execute task with potential tool calls
                while True:
                    # Trim messages to fit within token limit
                    exec_msgs = trim_messages(exec_msgs, MAX_CTX, model=EXE_MODEL)
                    exec_reply = await self.exec_llm.chat(exec_msgs, tools_schema)
                    
                    # If executor has a direct response, use it
                    if exec_reply["content"]:
                        result_text = str(exec_reply["content"])
                        self.shared_history.append({
                            "role": "assistant", 
                            "content": f"Task {task['id']} result: {result_text}"
                        })
                        break
                        
                    # Handle tool calls from executor
                    for call in exec_reply.get("tool_calls") or []:
                        t_name = call["function"]["name"]
                        t_args = json.loads(call["function"].get("arguments") or "{}")

                        try:
                            resolved = self._resolve_tool_name(t_name)
                        except KeyError as e:
                            error_msg = f"[tool resolution error] {e}"
                            exec_msgs.extend([
                                {"role": "assistant", "content": None, "tool_calls": [call]},
                                {"role": "tool", "tool_call_id": call.get("id", str(uuid.uuid4())), "name": t_name, "content": error_msg},
                            ])
                            continue

                        session = self.sessions[resolved]
                        patched_args = self._massage_args_for_tool(resolved, t_args)

                        result_msg = await session.call_tool(resolved, patched_args)
                        result_text = str(result_msg.content)

                        exec_msgs.extend([
                            {"role": "assistant", "content": None, "tool_calls": [call]},
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id", str(uuid.uuid4())),
                                "name": resolved,
                                "content": result_text
                            },
                        ])

            # Update planner messages with execution results for next cycle
            planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history
            
        # If we've exhausted cycles, return the last meta-planner response
        return meta_content.strip()

    async def cleanup(self):
        """Clean up resources and close tool server connections."""
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()

# ---------------------------------------------------------------------------
#   Command‑line & main routine
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command line arguments for the AgentFly client.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="AgentFly – interactive version")
    parser.add_argument("-q", "--question", type=str, help="Your question")
    parser.add_argument("-f", "--file", type=str, default="", help="Optional file path")
    parser.add_argument("-m", "--meta_model", type=str, default="gpt-4.1", help="Meta‑planner model")
    parser.add_argument("-e", "--exec_model", type=str, default="qwen3-8b", help="Executor model")
    parser.add_argument("-s", "--servers", type=str, nargs="*", default=[
        "../server/code_agent.py",
        "../server/documents_tool.py",
        "../server/image_tool.py",
        "../server/math_tool.py",
        "../server/ai_crawl.py",
        "../server/serp_search.py",
    ], help="Paths of tool server scripts")
    return parser.parse_args()

async def run_single_query(client: HierarchicalClient, question: str, file_path: str):
    """
    Execute a single query and display the result.
    
    Args:
        client: Initialized HierarchicalClient instance
        question: User's question
        file_path: Optional file path for context
    """
    answer = await client.process_query(question, file_path, str(uuid.uuid4()))
    print("\nFINAL ANSWER:", answer)

async def main_async(args):
    """
    Main async function that sets up and runs the AgentFly client.
    
    Args:
        args: Parsed command line arguments
    """
    # Load environment variables (API keys, etc.)
    load_dotenv()
    
    # Initialize the hierarchical client
    client = HierarchicalClient(args.meta_model, args.exec_model)
    
    # Connect to tool servers
    await client.connect_to_servers(args.servers)

    try:
        if args.question:
            # Run single query mode
            await run_single_query(client, args.question, args.file)
        else:
            # Interactive mode
            print("Enter 'exit' to quit.")
            while True:
                q = input("\nQuestion: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    break
                f = input("File path (optional): ").strip()
                await run_single_query(client, q, f)
    finally:
        # Ensure cleanup happens even if errors occur
        await client.cleanup()

if __name__ == "__main__":
    # Parse arguments and run the main async function
    arg_ns = parse_args()
    asyncio.run(main_async(arg_ns))

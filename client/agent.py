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
from openai import AsyncOpenAI, AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import colorlog
import json
import tiktoken

# ---------------------------------------------------------------------------
#   Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   Constants & templates (same as original)
# ---------------------------------------------------------------------------
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
    "If the final answer is not complete, emit a *new* JSON plan for the remaining work. Keep cycles as\n"
    "few as possible. Never call tools yourself — that's the EXECUTOR's job."\
    "⚠️  Reply with *pure JSON only*."
)

EXEC_SYSTEM_PROMPT = (
    "You are the EXECUTOR sub-agent. You receive one task description at a time\n"
    "from the meta-planner. Your job is to complete the task, using available\n"
    "tools via function calling if needed. Always think step by step but reply\n"
    "with the minimal content needed for the meta-planner. If you must call a\n"
    "tool, produce the appropriate function call instead of natural language.\n"
    "When done, output a concise result. Do NOT output FINAL ANSWER."
)

MAX_CTX = 175000
EXE_MODEL = "o3"

# ---------------------------------------------------------------------------
#   OpenAI backend
# ---------------------------------------------------------------------------
class ChatBackend:
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIBackend(ChatBackend):
    def __init__(self, model: str, is_azure: bool):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        ) if not is_azure else AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
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
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        resp = await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
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

# ---------------------------------------------------------------------------
#   Hierarchical client (trimmed: only essentials kept)
# ---------------------------------------------------------------------------
MAX_TURNS_MEMORY = 50

def _strip_fences(text: str) -> str:
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    m = re.search(r"{[\\s\\S]*}", text)
    return m.group(0) if m else text

def _count_tokens(msg: Dict[str, str], enc) -> int:
    role_tokens = 4
    content = msg.get("content") or ""
    return role_tokens + len(enc.encode(content))

def _get_tokenizer(model: str):
    """Return a tokenizer; fall back to cl100k_base if model is unknown."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def trim_messages(messages: List[Dict[str, str]], max_tokens: int, model="gpt-3.5-turbo"):

    enc = _get_tokenizer(model)
    total = sum(_count_tokens(m, enc) for m in messages) + 2
    if total <= max_tokens:
        return messages
    system_msg = messages[0]
    kept: List[Dict[str, str]] = [system_msg]
    total = _count_tokens(system_msg, enc) + 2
    for msg in reversed(messages[1:]):
        t = _count_tokens(msg, enc)
        if total + t > max_tokens:
            break
        kept.insert(1, msg)
        total += t
    return kept

class HierarchicalClient:
    MAX_CYCLES = 3

    def __init__(self, meta_model: str, exec_model: str, is_azure: bool = True):
        self.meta_llm = OpenAIBackend(meta_model, is_azure)
        self.exec_llm = OpenAIBackend(exec_model, is_azure)
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, str]] = []

    # ---------- Tool management ----------
    async def connect_to_servers(self, scripts: List[str]):
        from contextlib import AsyncExitStack
        self.exit_stack = AsyncExitStack()
        for script in scripts:
            path = Path(script)
            cmd = "python" if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session
        print("Connected tools:", list(self.sessions.keys()))

    async def _tools_schema(self) -> List[Dict[str, Any]]:
        result, cached = [], {}
        for session in self.sessions.values():
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
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
        tools_schema = await self._tools_schema()
        self.shared_history = []
        self.shared_history.append({"role": "user", "content": f"{query}\ntask_id: {task_id}\nfile_path: {file}\n"})
        planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history

        for cycle in range(self.MAX_CYCLES):
            meta_reply = await self.meta_llm.chat(planner_msgs)
            meta_content = meta_reply["content"] or ""
            self.shared_history.append({"role": "assistant", "content": meta_content})

            if meta_content.startswith("FINAL ANSWER:"):
                return meta_content[len("FINAL ANSWER:"):].strip()

            try:
                tasks = json.loads(_strip_fences(meta_content))["plan"]
            except Exception as e:
                return f"[planner error] {e}: {meta_content}"

            for task in tasks:
                task_desc = f"Task {task['id']}: {task['description']}"
                exec_msgs = (
                    [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] +
                    self.shared_history +
                    [{"role": "user", "content": task_desc}]
                )
                while True:
                    exec_msgs = trim_messages(exec_msgs, MAX_CTX, model=EXE_MODEL)
                    exec_reply = await self.exec_llm.chat(exec_msgs, tools_schema)
                    if exec_reply["content"]:
                        result_text = str(exec_reply["content"])
                        self.shared_history.append({"role": "assistant", "content": f"Task {task['id']} result: {result_text}"})
                        break
                    for call in exec_reply.get("tool_calls") or []:
                        t_name = call["function"]["name"]
                        t_args = json.loads(call["function"].get("arguments") or "{}")
                        session = self.sessions[t_name]
                        result_msg = await session.call_tool(t_name, t_args)
                        result_text = str(result_msg.content)
                        exec_msgs.extend([
                            {"role": "assistant", "content": None, "tool_calls": [call]},
                            {"role": "tool", "tool_call_id": call["id"], "name": t_name, "content": result_text},
                        ])
            planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history
        return meta_content.strip()

    async def cleanup(self):
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()

# ---------------------------------------------------------------------------
#   Command‑line & main routine
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AgentFly – interactive version")
    parser.add_argument("-q", "--question", type=str, help="Your question")
    parser.add_argument("-f", "--file", type=str, default="", help="Optional file path")
    parser.add_argument("-m", "--meta_model", type=str, default="gpt-4.1", help="Meta‑planner model")
    parser.add_argument("-e", "--exec_model", type=str, default="o3-2025-04-16", help="Executor model")
    parser.add_argument("-s", "--servers", type=str, nargs="*", default=[
        "../server/code_agent.py",
        "../server/craw_page.py",
        "../server/documents_tool.py",
        "../server/excel_tool.py",
        "../server/image_tool.py",
        "../server/math_tool.py",
        "../server/search_tool.py",
        "../server/video_tool.py",
    ], help="Paths of tool server scripts")
    return parser.parse_args()

async def run_single_query(client: HierarchicalClient, question: str, file_path: str):
    answer = await client.process_query(question, file_path, str(uuid.uuid4()))
    print("\nFINAL ANSWER:", answer)

async def main_async(args):
    load_dotenv()
    client = HierarchicalClient(args.meta_model, args.exec_model, os.getenv("USE_AZURE_OPENAI") == "True")
    await client.connect_to_servers(args.servers)

    try:
        if args.question:
            await run_single_query(client, args.question, args.file)
        else:
            print("Enter 'exit' to quit.")
            while True:
                q = input("\nQuestion: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    break
                f = input("File path (optional): ").strip()
                await run_single_query(client, q, f)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    arg_ns = parse_args()
    asyncio.run(main_async(arg_ns))

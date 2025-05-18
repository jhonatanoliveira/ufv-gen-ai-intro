#!/usr/bin/env python3
"""LLM-powered agent that can use tools to answer questions and solve problems."""
import argparse
import asyncio
import datetime
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from io import StringIO
from typing import Sequence

import chromadb
from chromadb.errors import NotFoundError
from config import get_settings
from litellm import acompletion
from pydantic import BaseModel, Field

settings = get_settings()

# Agent configuration and prompt tokens
FINAL_ANSWER_TOKEN = "Final Answer:"
OBSERVATION_TOKEN = "Observation:"
THOUGHT_TOKEN = "Thought:"
PROMPT_TEMPLATE = """Today is {today} and you can use tools to get new information.
Answer the question as best you can using the following tools:

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: comment on what you want to do next
Action: the action to take, exactly one element of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation repeats N times, use it until you are sure of the answer)
Thought: I now know the final answer
Final Answer: your final answer to the original input question

Begin!

Question: {question}
Thought: {previous_responses}
"""


class ToolInterface(BaseModel, ABC):
    """Abstract base class for agent tools."""

    name: str
    description: str

    @abstractmethod
    def use(self, input_text: str) -> str:
        """Execute the tool with the given input text and return the result."""


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL.
    If nothing is printed, the last expression is evaluated and returned.
    """

    globals: dict = Field(default_factory=dict)
    locals: dict = Field(default_factory=dict)

    def run(self, command: str) -> str:
        old, out_buf = sys.stdout, StringIO()
        try:
            sys.stdout = out_buf
            # 1) try exec first (catches statements & prints)
            exec(command, self.globals, self.locals)  # pylint: disable=exec-used
            result = out_buf.getvalue()
            # 2) if nothing printed, try evaluating last expr
            if not result.strip():
                try:
                    # Deliberately using eval in REPL context
                    last = command.strip().splitlines()[-1]
                    val = eval(  # pylint: disable=eval-used
                        last, self.globals, self.locals
                    )
                    return repr(val) + "\n"
                except Exception:  # pylint: disable=broad-except
                    pass
            return result
        except Exception as e:  # pylint: disable=broad-except
            return str(e)
        finally:
            sys.stdout = old


class PythonREPLTool(ToolInterface):
    """Tool for executing Python code."""

    name: str = "Python REPL"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Print outputs to see them."
    )
    python_repl: PythonREPL = PythonREPL()

    def use(self, input_text: str) -> str:
        """Execute Python code from input text, stripping markdown formatting."""
        code = input_text.strip().strip("```")
        return self.python_repl.run(code)


class VectorNewsSearchTool(ToolInterface):
    """Semantic search over news articles stored in ChromaDB."""

    name: str = "News Search"
    description: str = (
        "Perform a semantic search in the news articles. "
        "Input must be JSON with keys:\n"
        "  • query (str) – the search string;\n"
        "  • n_results (int, optional, default=5);\n"
        "  • filters (dict, optional) – metadata filters. Supported fields are:\n"
        "      • code (str): unique document ID\n"
        "      • category (str): news category\n"
        "      • title (str): article title\n"
        "      • date (str): publication date\n"
        "      • location (str): geographic location\n"
        "      • label (str): classification label\n"
        "      • embedded_at (str): embedding timestamp\n"
        "Returns a JSON array of {{id, document, metadata, distance}} hits."
    )

    # by default match your embedding code
    collection_name: str = Field(
        default_factory=lambda: getattr(
            settings, "chroma_collection_name", "news_articles"
        )
    )
    persist_directory: str = Field(
        default_factory=lambda: getattr(
            settings, "chroma_persist_directory", "chroma_db"
        )
    )

    def use(self, input_text: str) -> str:
        # 1) parse input
        try:
            params = json.loads(input_text)
            query = params["query"]
            n_results = int(params.get("n_results", 5))
            filters = params.get("filters", {}) or None
        except (json.JSONDecodeError, KeyError, TypeError):
            # fallback: treat entire input as the query
            query = input_text.strip()
            n_results = 5
            filters = None

        # 2) connect to ChromaDB
        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            coll = client.get_collection(name=self.collection_name)
        except NotFoundError:
            return json.dumps(
                {"error": f"Collection '{self.collection_name}' not found."},
                indent=2,
            )

        # 3) perform the query
        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        # 4) flatten and format
        ids = results.get("ids", [[]])
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        dists = results.get("distances", [[]])

        # Extract first items or use empty defaults
        ids = ids[0] if ids and len(ids) > 0 else []
        docs = docs[0] if docs and len(docs) > 0 else []
        metas = metas[0] if metas and len(metas) > 0 else []
        dists = dists[0] if dists and len(dists) > 0 else []

        output = []
        for i, doc in enumerate(docs):
            output.append({
                "id": ids[i] if i < len(ids) else None,
                "document": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })

        return json.dumps(output, indent=2)


class ChatLLM(BaseModel):
    """Client for communicating with LLM API."""

    model: str = settings.llm_model

    async def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        """Generate text completion from LLM given a prompt."""
        messages = [{"role": "user", "content": prompt}]
        logging.info("Calling LLM for prompt.")
        response = await acompletion(
            api_key=settings.openai_api_key.get_secret_value(),
            model=self.model,
            messages=messages,
            stop=stop,
        )
        content = response.choices[0].message.content  # type: ignore
        logging.info("Received response from LLM.")
        return content or ""


class Agent(BaseModel):
    """Orchestrates LLM interactions with tools to answer questions.
    The Agent orchestrates prompt -> LLM -> tool -> observation loops"""

    llm: ChatLLM
    tools: Sequence[ToolInterface]
    prompt_template: str = PROMPT_TEMPLATE
    max_loops: int = 15
    stop_pattern: list[str] = Field(
        default_factory=lambda: [f"\n{OBSERVATION_TOKEN}", f"\n\t{OBSERVATION_TOKEN}"]
    )

    @property
    def tool_description(self) -> str:
        """Generate formatted description of available tools."""
        return "\n".join(f"{t.name}: {t.description}" for t in self.tools)

    @property
    def tool_names(self) -> str:
        """Return comma-separated list of tool names."""
        return ",".join(t.name for t in self.tools)

    @property
    def tool_by_name(self) -> dict[str, ToolInterface]:
        """Create mapping of tool names to tool instances."""
        return {t.name: t for t in self.tools}

    async def run(self, question: str) -> str:
        """Run the agent to answer a question using available tools."""
        logging.info("Starting agent run.")
        previous: list[str] = []
        base_prompt = self.prompt_template.format(
            today=datetime.date.today(),
            tool_description=self.tool_description,
            tool_names=self.tool_names,
            question=question,
            previous_responses="{previous_responses}",
        )
        # Print initial prompt without previous steps
        print(base_prompt.format(previous_responses=""))
        for loop_idx in range(1, self.max_loops + 1):
            curr = base_prompt.format(previous_responses="\n".join(previous))
            logging.info("Loop %d: generating next action.", loop_idx)
            gen, tool_name, tool_input = await self.decide_next_action(curr)
            if tool_name == "Final Answer":
                logging.info("Final answer obtained.")
                return tool_input
            if tool_name not in self.tool_by_name:
                logging.error("Unknown tool: %s", tool_name)
                raise ValueError(f"Unknown tool: {tool_name}")
            logging.info("Using tool '%s' with input: %s", tool_name, tool_input)
            tool = self.tool_by_name[tool_name]
            # run synchronous tool in thread to keep async loop responsive
            observation = await asyncio.to_thread(tool.use, tool_input)
            logging.info("Observation: %s", observation.strip())
            step = f"{gen}\n{OBSERVATION_TOKEN} {observation.strip()}\n{THOUGHT_TOKEN}"
            print(step)
            previous.append(step)
        logging.error("Exceeded max loops without finding final answer.")
        raise RuntimeError("Agent failed to reach a final answer within loop limit")

    async def decide_next_action(self, prompt: str) -> tuple[str, str, str]:
        """Generate next action based on the current prompt."""
        content = await self.llm.generate(prompt, stop=self.stop_pattern)
        return content, *self.parse_response(content)

    def parse_response(self, generated: str) -> tuple[str, str]:
        """Extract tool name and input from generated text."""
        # Check for explicit Final Answer
        if FINAL_ANSWER_TOKEN in generated:
            return "Final Answer", generated.split(FINAL_ANSWER_TOKEN, 1)[1].strip()

        # Look for Action/Action Input format
        pattern = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(pattern, generated, re.DOTALL)
        if match:
            tool, inp = match.group(1).strip(), match.group(2).strip().strip('"')
            return tool, inp

        # If the model directly generates an answer without following the format,
        # treat it as a final answer
        logging.warning(
            "LLM generated direct response without using tools: %s",
            generated[:100] + "..." if len(generated) > 100 else generated,
        )
        return "Final Answer", generated.strip()


async def main():
    """Entry point for command-line agent execution."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run the async LLM agent")
    parser.add_argument("query", help="The question or prompt to send to the agent")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show detailed logs of what the agent is doing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.debug else logging.CRITICAL,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    llm = ChatLLM()
    tools = [PythonREPLTool(), VectorNewsSearchTool()]
    agent = Agent(llm=llm, tools=tools)
    answer = await agent.run(args.query)
    print(f"\nFinal answer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())

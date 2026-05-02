from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol

from .tools import Tool, run_tool

EventSink = Callable[[dict], None]


def emit(sink: EventSink | None, event: dict) -> None:
    if sink is not None:
        try:
            sink(event)
        except Exception:
            pass


class Provider(Protocol):
    name: str

    def run(
        self,
        system: str,
        user: str,
        tools: list[Tool],
        max_steps: int = 8,
        on_event: EventSink | None = None,
    ) -> dict: ...


# Used to extract the verdict from the response.
VERDICT_RE = re.compile(
    r'\{[^{}]*"verdict"\s*:\s*"(safe|unsafe)"[^{}]*\}', re.IGNORECASE
)


def extract_verdict(text: str) -> dict | None:
    if not text:
        return None
    # Try strict JSON anywhere in the response.
    for match in VERDICT_RE.finditer(text):
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        v = str(obj.get("verdict", "")).strip().lower()
        if v in ("safe", "unsafe"):
            return {"verdict": v, "rationale": str(obj.get("rationale", "")).strip()}
    # Fallback: any naked safe/unsafe token in the last 200 chars.
    tail = text[-200:].lower()
    if "unsafe" in tail and "safe" not in tail.replace("unsafe", ""):
        return {"verdict": "unsafe", "rationale": text.strip()[:400]}
    if " safe" in tail or tail.startswith("safe"):
        return {"verdict": "safe", "rationale": text.strip()[:400]}
    return None


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # local import so OpenAI-only users don't need it

        self._anthropic = anthropic
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        print(f"Using model: {self.model} for anthropic provider")
        self.client = anthropic.Anthropic()

    def tools_payload(self, tools: list[Tool]) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    def run(
        self,
        system: str,
        user: str,
        tools: list[Tool],
        max_steps: int = 8,
        on_event: EventSink | None = None,
    ) -> dict:
        trace: list[dict] = []
        messages: list[dict] = [{"role": "user", "content": user}]
        tool_payload = self.tools_payload(tools)

        for step_idx in range(max_steps):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=tool_payload,
                messages=messages,
            )
            text_chunks = []
            tool_uses = []
            for block in resp.content:
                if block.type == "text":
                    text_chunks.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            assistant_content = [b.model_dump() for b in resp.content]
            messages.append({"role": "assistant", "content": assistant_content})

            for chunk in text_chunks:
                if chunk.strip():
                    emit(on_event, {"step": "text", "text": chunk})

            if not tool_uses:
                final = "\n".join(text_chunks)
                trace.append({"step": "final", "text": final})
                verdict = extract_verdict(final)
                if verdict is None:
                    verdict = {
                        "verdict": "error",
                        "rationale": f"[parse-failure] {final[:300]}",
                    }
                emit(
                    on_event,
                    {
                        "step": "final",
                        "text": final,
                        "verdict": verdict,
                        "steps": step_idx + 1,
                    },
                )
                return {
                    **verdict,
                    "trace": trace,
                    "stop_reason": resp.stop_reason,
                    "steps": step_idx + 1,
                }

            tool_results = []
            for tu in tool_uses:
                emit(
                    on_event,
                    {"step": "tool_start", "name": tu.name, "input": dict(tu.input)},
                )
                result = run_tool(tools, tu.name, tu.input)
                trace.append(
                    {
                        "step": "tool",
                        "name": tu.name,
                        "input": tu.input,
                        "result": result,
                    }
                )
                emit(
                    on_event,
                    {
                        "step": "tool_end",
                        "name": tu.name,
                        "input": dict(tu.input),
                        "result": result,
                    },
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        trace.append({"step": "budget_exhausted"})
        emit(on_event, {"step": "budget_exhausted", "steps": max_steps})
        return {
            "verdict": "error",
            "rationale": "[budget-exhausted] agent did not return a verdict in time",
            "trace": trace,
            "stop_reason": "budget",
            "steps": max_steps,
        }


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.environ.get("OPENAI_MODEL", "minimax-m2.7")
        print(f"Using model for openai provider: {self.model}")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://opencode.ai/zen/v1")
        print(f"Using base URL for openai provider: {base_url}")
        self.client = OpenAI(base_url=base_url)

    def tools_payload(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def run(
        self,
        system: str,
        user: str,
        tools: list[Tool],
        max_steps: int = 8,
        on_event: EventSink | None = None,
    ) -> dict:
        trace: list[dict] = []
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tool_payload = self.tools_payload(tools)

        for step_idx in range(max_steps):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_payload,
                max_tokens=1024,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            tool_calls = msg.tool_calls or []

            if msg.content and msg.content.strip():
                emit(on_event, {"step": "text", "text": msg.content})

            if not tool_calls:
                final = msg.content or ""
                trace.append({"step": "final", "text": final})
                verdict = extract_verdict(final)
                if verdict is None:
                    verdict = {
                        "verdict": "error",
                        "rationale": f"[parse-failure] {final[:300]}",
                    }
                emit(
                    on_event,
                    {
                        "step": "final",
                        "text": final,
                        "verdict": verdict,
                        "steps": step_idx + 1,
                    },
                )
                return {
                    **verdict,
                    "trace": trace,
                    "stop_reason": resp.choices[0].finish_reason,
                    "steps": step_idx + 1,
                }

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                emit(
                    on_event,
                    {"step": "tool_start", "name": tc.function.name, "input": args},
                )
                result = run_tool(tools, tc.function.name, args)
                trace.append(
                    {
                        "step": "tool",
                        "name": tc.function.name,
                        "input": args,
                        "result": result,
                    }
                )
                emit(
                    on_event,
                    {
                        "step": "tool_end",
                        "name": tc.function.name,
                        "input": args,
                        "result": result,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        trace.append({"step": "budget_exhausted"})
        emit(on_event, {"step": "budget_exhausted", "steps": max_steps})
        return {
            "verdict": "error",
            "rationale": "[budget-exhausted] agent did not return a verdict in time",
            "trace": trace,
            "stop_reason": "budget",
            "steps": max_steps,
        }


def get_provider(name: str) -> Provider:
    n = name.lower()
    if n == "anthropic":
        return AnthropicProvider()
    if n == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name}")

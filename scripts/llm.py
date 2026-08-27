"""Thin JSON-only client for an OpenAI-compatible endpoint.

Points at whatever the named config section says -- LM Studio locally, or a
cloud endpoint such as OpenRouter. Nothing above this module knows or cares
which; `LLM(cfg, logger, section="openrouter")` is the whole difference.

Two quirks of LM Studio's reasoning models are handled here so callers do not
have to:

  * `response_format` must be `json_schema` or `text`. `json_object`, which is
    the usual OpenAI spelling, is rejected outright.
  * With a reasoning model the whole reply can land in `reasoning_content`
    while `content` comes back empty. The JSON is correct; it is just filed
    under the wrong key.

Neither quirk is universal, so both are configurable per section: a cloud
model rejects LM Studio's chat-template argument outright, and not every
model behind OpenRouter implements structured outputs at all.
"""
from __future__ import annotations

import json
import os
import re
import time


class LLMError(RuntimeError):
    pass


class LLM:
    """JSON-in, JSON-out. Never raises on a bad model reply -- returns None so
    a single bad window cannot kill a batch."""

    def __init__(self, cfg: dict, logger, section: str = "llm"):
        from openai import OpenAI

        if section not in cfg:
            raise LLMError(f"config has no `{section}:` section")
        c = cfg[section]
        self.cfg = c
        self.section = section
        self.logger = logger
        self.model = c["model"]
        self.temperature = float(c.get("temperature", 0.0))
        self.max_tokens = int(c.get("max_tokens", 4096))
        self.enable_thinking = bool(c.get("enable_thinking", False))
        # LM Studio wants the chat-template argument; a cloud endpoint rejects
        # the request outright when it arrives. Default on, so the local
        # `llm:` section keeps behaving exactly as it did.
        self.send_thinking_flag = bool(c.get("send_thinking_flag", True))
        self.structured_outputs = bool(c.get("structured_outputs", True))

        self.client = OpenAI(base_url=c["base_url"], api_key=self._api_key(c),
                             timeout=float(c.get("timeout_s", 300)),
                             default_headers=self._headers(c) or None)
        self.stats = {"calls": 0, "retries": 0, "failures": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def _api_key(self, c: dict) -> str:
        """Config first, then the environment variable it names.

        A section that names `api_key_env` is saying it talks to something
        that actually authenticates, so a missing key is a configuration error
        worth failing on rather than a 401 forty seconds into the run.
        """
        key = str(c.get("api_key") or "").strip()
        env = str(c.get("api_key_env") or "").strip()
        if not key and env:
            key = os.environ.get(env, "").strip()
        if not key and env:
            raise LLMError(
                f"no API key for the `{self.section}:` endpoint -- put one in "
                f"config.yaml under {self.section}.api_key, or set the {env} "
                f"environment variable")
        return key or "not-needed"

    @staticmethod
    def _headers(c: dict) -> dict:
        h = c.get("headers") or {}
        out = {}
        if str(h.get("referer") or "").strip():
            out["HTTP-Referer"] = str(h["referer"]).strip()
        if str(h.get("title") or "").strip():
            out["X-Title"] = str(h["title"]).strip()
        return out

    # ------------------------------------------------------------------ io --

    def _once(self, messages, schema, schema_name, max_tokens):
        kw = {}
        if schema and self.structured_outputs:
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        if not self.enable_thinking and self.send_thinking_flag:
            # Roughly halves latency on qwen3.5 and stops it narrating.
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        t0 = time.time()
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kw,
        )
        self.stats["seconds"] += time.time() - t0
        self.stats["calls"] += 1
        if getattr(r, "usage", None):
            self.stats["prompt_tokens"] += r.usage.prompt_tokens or 0
            self.stats["completion_tokens"] += r.usage.completion_tokens or 0

        msg = r.choices[0].message
        text = msg.content or ""
        if not text.strip():
            text = (getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None) or "")
        return text, r.choices[0].finish_reason

    # -------------------------------------------------------------- parsing --

    # JSON and LaTeX disagree about backslashes. `\b` and `\f` are valid JSON
    # escapes, so a model that writes "\boldsymbol" or "\frac" instead of
    # "\\boldsymbol" hands the parser a backspace or form feed and the command
    # name silently loses its first letter -- \boldsymbol{\epsilon} arrives as
    # <BS>oldsymbol{\epsilon}, which KaTeX cannot render and which is invisible
    # in a diff. Neither character has any business in lecture notes, so
    # putting the backslash back is safe.
    _MANGLED = {"\x08": r"\b", "\x0c": r"\f"}

    @classmethod
    def _unmangle(cls, obj, counter):
        if isinstance(obj, str):
            if any(ch in obj for ch in cls._MANGLED):
                for ch, repl in cls._MANGLED.items():
                    if ch in obj:
                        counter[0] += obj.count(ch)
                        obj = obj.replace(ch, repl)
            return obj
        if isinstance(obj, list):
            return [cls._unmangle(v, counter) for v in obj]
        if isinstance(obj, dict):
            return {k: cls._unmangle(v, counter) for k, v in obj.items()}
        return obj

    @staticmethod
    def extract_json(text: str):
        """Pull a JSON object out of a model reply, tolerating the usual mess:
        think blocks, ```json fences, and prose on either side."""
        if not text:
            return None
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^```(?:json)?|```$", " ", text.strip(),
                      flags=re.MULTILINE).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fall back to the first balanced {...}, ignoring braces inside strings.
        start = text.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
        return None

    # ----------------------------------------------------------------- api --

    def json_call(self, system: str, user: str, schema: dict = None,
                  schema_name: str = "result", max_tokens: int = None,
                  retries: int = 1, label: str = ""):
        """Returns parsed JSON, or None once retries are exhausted.

        The brief is explicit: retry once on malformed output, then log it and
        move on. A bad window must never take the batch down.
        """
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

        for attempt in range(retries + 1):
            try:
                text, finish = self._once(messages, schema, schema_name, max_tokens)
            except Exception as exc:
                self.logger.warning("%s LLM request failed (%s): %s",
                                    label, type(exc).__name__, str(exc)[:200])
                text, finish = "", "error"

            data = self.extract_json(text)
            if data is not None:
                fixed = [0]
                data = self._unmangle(data, fixed)
                if fixed[0]:
                    self.logger.info("%s repaired %d LaTeX backslash escape(s) "
                                     "eaten by JSON", label, fixed[0])
                if finish == "length":
                    self.logger.warning("%s reply hit the token cap -- it may be "
                                        "truncated; raising llm.max_tokens helps", label)
                return data

            if attempt < retries:
                self.stats["retries"] += 1
                self.logger.warning("%s malformed JSON, retrying once", label)
                messages = messages + [
                    {"role": "assistant", "content": (text or "")[:600]},
                    {"role": "user", "content":
                        "That was not valid JSON. Reply with the JSON object only -- "
                        "no prose, no code fences, no commentary."},
                ]

        self.stats["failures"] += 1
        self.logger.error("%s gave unusable output twice; skipping it", label)
        return None

    def log_stats(self):
        s = self.stats
        self.logger.info("LLM: %d calls, %d retries, %d failures, "
                         "%d prompt + %d completion tokens, %.0fs total",
                         s["calls"], s["retries"], s["failures"],
                         s["prompt_tokens"], s["completion_tokens"], s["seconds"])

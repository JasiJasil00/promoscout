from __future__ import annotations
import hashlib, json, time
from abc import ABC, abstractmethod
from pathlib import Path


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    """Model-independent interface. To plug in another provider, subclass and implement `_call`."""
    name = "base"

    def __init__(self, cache_dir: str | None = ".llm_cache", min_interval_s: float = 0.0, max_retries: int = 4):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_interval_s, self.max_retries = min_interval_s, max_retries
        self.raw_log: list[dict] = []        # every request/response, unedited, for export
        self._last = 0.0

    @abstractmethod
    def _call(self, system: str, user: str) -> str:
        """Return the model's raw text (expected to be JSON). Raise LLMError(retryable=...) via `retryable` attr."""

    def model_id(self) -> str:
        return self.name

    def generate_json(self, system: str, user: str, tag: str = "") -> dict:
        key = hashlib.sha256((self.model_id() + "\x00" + system + "\x00" + user).encode()).hexdigest()
        if self.cache_dir:
            f = self.cache_dir / f"{key}.json"
            if f.exists():
                text = f.read_text(encoding="utf-8")
                self.raw_log.append({"tag": tag, "model": self.model_id(), "cached": True, "response": text, "attempts": 0})
                return json.loads(text)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0: time.sleep(wait)
            self._last = time.time()
            text = ""
            try:
                text = self._call(system, user)
                data = json.loads(_strip_fences(text))
                self.raw_log.append({"tag": tag, "model": self.model_id(), "cached": False, "prompt_chars": len(user),
                                     "response": text, "attempts": attempt})
                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    (self.cache_dir / f"{key}.json").write_text(json.dumps(data), encoding="utf-8")
                return data
            except (LLMError, json.JSONDecodeError) as e:
                last_err = e
                self.raw_log.append({"tag": tag, "model": self.model_id(), "error": repr(e), "response": text, "attempts": attempt})
                if not getattr(e, "retryable", True): break
                time.sleep(min(2 ** attempt, 30) if self.min_interval_s else 0)
        raise LLMError(f"giving up after {self.max_retries} attempts: {last_err!r}")


def _strip_fences(t: str) -> str:
    t = t.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"): t = t.rstrip()[:-3]
    return t.strip()

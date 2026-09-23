"""Local answer generator running on-device conversational models."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.config.settings import get_settings
from app.generation.base import AnswerGenerator
from app.generation.exceptions import (
    EmptyResponseError,
    LocalGenerationError,
)
from app.memory.base import MemoryContext

logger = logging.getLogger("toy_agent.generation.local")


def format_memory_fact(key: str, value: Optional[str]) -> str:
    """Deterministically convert a memory key and optional value into a natural-language statement.

    Examples:
        ('child_name', 'Alex') -> "The child's name is Alex."
        ('name', 'Alex') -> "The child's name is Alex."
        ('favorite_animal', 'tiger') -> "The child's favorite animal is tiger."
        ('favorite_color', 'blue') -> "The child's favorite color is blue."
        ('favorite_song', None) -> "The child's favorite song is not known."
    """
    clean_key = key.strip().lower()

    if clean_key in {"child_name", "name"}:
        descriptor = "name"
    elif clean_key.startswith("favorite_"):
        descriptor = f"favorite {clean_key[9:].replace('_', ' ')}"
    elif clean_key.startswith("child_"):
        descriptor = clean_key[6:].replace('_', ' ')
    else:
        descriptor = clean_key.replace('_', ' ')

    if value is not None and str(value).strip():
        return f"The child's {descriptor} is {value}."
    else:
        return f"The child's {descriptor} is not known."


class LocalAnswerGenerator(AnswerGenerator):
    """Generates child-facing conversational answers using an on-device model.

    Designed for lightweight models (defaults to Qwen/Qwen2.5-1.5B-Instruct)
    running locally on CPU or Apple Silicon (MPS).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.local_answer_model
        self.max_tokens = max_tokens or settings.local_generation_max_tokens
        self.temperature = temperature if temperature is not None else settings.local_generation_temperature
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        return f"LocalAnswerGenerator({self.model_name})"

    def _ensure_loaded(self) -> None:
        """Synchronously load model and tokenizer if not already loaded."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._device is None:
                if torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

            logger.info("Loading local answer model '%s' on %s...", self.model_name, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=False,
                dtype=torch.float16 if self._device == "mps" else torch.float32,
            )
            self._model.to(self._device)
            self._model.eval()
            self._is_loaded = True
            logger.info("Local answer model '%s' successfully loaded.", self.model_name)
        except Exception as exc:
            logger.error("Failed to load local answer model '%s': %s", self.model_name, exc)
            raise LocalGenerationError(f"Failed to load local model '{self.model_name}': {exc}") from exc

    def _build_prompt_messages(
        self,
        query: str,
        memory_context: Optional[MemoryContext] = None,
    ) -> List[Dict[str, str]]:
        """Construct structured chat messages separating persona, memory, and query."""
        if memory_context is not None and (memory_context.hits or memory_context.misses):
            fact_lines = []
            for k, v in memory_context.hits.items():
                fact_lines.append(f"- {format_memory_fact(k, v)}")
            for k in memory_context.misses:
                fact_lines.append(f"- {format_memory_fact(k, None)}")

            known_info = "\n".join(fact_lines)
            system_content = (
                "You are a friendly conversational toy assistant.\n"
                "Answer the child's question clearly and concisely in 1-2 friendly sentences.\n"
                "Use the provided device memory when it is relevant.\n"
                "Never invent personal information.\n"
                "If requested personal information is not present in memory, say that you don't know.\n\n"
                f"Known device information:\n{known_info}"
            )
        else:
            system_content = (
                "You are a friendly conversational toy assistant.\n"
                "Answer the child's question clearly and concisely in 1-2 friendly sentences."
            )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query.strip()},
        ]

    def _generate_sync(
        self,
        query: str,
        memory_context: Optional[MemoryContext] = None,
    ) -> str:
        """Synchronous inference worker executed in background thread."""
        self._ensure_loaded()

        messages = self._build_prompt_messages(query, memory_context)

        try:
            import torch

            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

            inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
            prompt_len = inputs["input_ids"].shape[1]

            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.max_tokens,
                "pad_token_id": self._tokenizer.eos_token_id or self._tokenizer.pad_token_id,
            }
            if self.temperature > 0.0:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["do_sample"] = True
            else:
                generate_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **generate_kwargs)

            new_tokens = output_ids[0][prompt_len:]
            answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            if not answer:
                raise EmptyResponseError(f"Model '{self.model_name}' produced an empty answer")

            return answer

        except EmptyResponseError:
            raise
        except Exception as exc:
            logger.error("Local generation failed for model '%s': %s", self.model_name, exc)
            raise LocalGenerationError(f"Local generation failed: {exc}") from exc

    async def generate(
        self,
        query: str,
        memory_context: Optional[MemoryContext] = None,
    ) -> str:
        """Asynchronously generate answer text via background thread executor."""
        return await asyncio.to_thread(self._generate_sync, query, memory_context)

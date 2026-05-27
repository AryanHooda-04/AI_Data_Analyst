"""OpenAI-backed natural language analysis."""

from __future__ import annotations

import io
import os
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI, OpenAIError

from utils import clean_text, dataframe_context


load_dotenv()

DEFAULT_MODEL = "gpt-5.2"
INSECURE_SSL_MODE = "insecure"
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"

if not os.getenv("OPENAI_SSL"):
    os.environ["OPENAI_SSL"] = INSECURE_SSL_MODE


def openai_ssl_mode() -> str:
    """Return the configured OpenAI SSL mode for this app."""
    return (os.getenv("OPENAI_SSL") or INSECURE_SSL_MODE).strip().lower()


def openai_ssl_is_insecure() -> bool:
    """Whether OpenAI requests should bypass local certificate verification."""
    return openai_ssl_mode() == INSECURE_SSL_MODE


def _client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file, then rerun the app."
        )

    if openai_ssl_is_insecure():
        return OpenAI(http_client=DefaultHttpxClient(verify=False))

    return OpenAI()


def _uses_responses_api(model: str) -> bool:
    """Return whether a model should use the Responses API path."""
    normalized = model.lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def _message_input_for_responses(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert simple chat messages into Responses API input items."""
    items = []
    for message in messages:
        role = message["role"]
        if role == "system":
            role = "developer"
        items.append({"role": role, "content": message["content"]})
    return items


def complete_with_messages(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    temperature: float = 0.2,
    max_tokens: int = 1_200,
) -> str:
    """Call OpenAI and return text content."""
    try:
        client = _client()
        if _uses_responses_api(model):
            kwargs = {
                "model": model,
                "input": _message_input_for_responses(messages),
                "max_output_tokens": max_tokens,
            }
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(**kwargs)
            return (getattr(response, "output_text", "") or "").strip()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    content = response.choices[0].message.content or ""
    return content.strip()


def _format_history(history: Iterable[dict[str, str]], limit: int = 6) -> list[dict[str, str]]:
    allowed_roles = {"user", "assistant"}
    compact = []
    for message in list(history)[-limit:]:
        role = message.get("role")
        content = clean_text(message.get("content"))
        if role in allowed_roles and content:
            compact.append({"role": role, "content": content[:2_000]})
    return compact


def ask_ai(
    df: pd.DataFrame,
    question: str,
    history: Iterable[dict[str, str]] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    max_tokens: int = 1_200,
    context_max_chars: int = 16_000,
) -> str:
    """Answer a natural-language question about the DataFrame."""
    question = clean_text(question)
    if not question:
        raise ValueError("Please enter a question to analyze.")

    system_prompt = (
        "You are a senior data analyst. Answer clearly with reasoning. "
        "Use only the dataset context provided unless you explicitly label an assumption. "
        "Call out uncertainty, data quality issues, and practical next steps when relevant. "
        "Return the answer using exactly these Markdown section headings: "
        "Summary, Evidence, Caveats, Recommended Next Steps."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Dataset context for this analysis:\n\n" + dataframe_context(df, max_chars=context_max_chars),
        },
    ]
    if history:
        messages.extend(_format_history(history))
    messages.append({"role": "user", "content": question})

    return complete_with_messages(
        messages,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=0.2,
        max_tokens=max_tokens,
    )


def conversation_ai(
    df: pd.DataFrame,
    message: str,
    history: Iterable[dict[str, str]] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = "none",
    max_tokens: int = 1_200,
    context_max_chars: int = 16_000,
) -> str:
    """Continue a conversational analysis grounded in the active DataFrame."""
    message = clean_text(message)
    if not message:
        raise ValueError("Please enter a message before sending.")

    system_prompt = (
        "You are Conversation AI, a senior data analyst embedded in an analytics workspace. "
        "Use the dataset context and the recent chat history to answer naturally, including follow-up questions. "
        "Do not invent facts that are not supported by the dataset context; label assumptions clearly. "
        "When the user asks for analysis, structure the answer with these Markdown headings when useful: "
        "Summary, Evidence, Caveats, Recommended Next Steps. "
        "For quick clarifications, definitions, or conversational follow-ups, answer directly and concisely. "
        "If a chart, SQL query, or Pandas snippet would help, suggest it briefly."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Active dataset context for this conversation:\n\n" + dataframe_context(df, max_chars=context_max_chars),
        },
    ]
    if history:
        messages.extend(_format_history(history, limit=10))
    messages.append({"role": "user", "content": message})

    return complete_with_messages(
        messages,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=0.25,
        max_tokens=max_tokens,
    )


def transcribe_audio(
    audio_bytes: bytes,
    *,
    filename: str = "voice_input.webm",
    model: str = TRANSCRIPTION_MODEL,
) -> str:
    """Transcribe recorded or uploaded audio into text."""
    if not audio_bytes:
        raise ValueError("No audio was provided for transcription.")

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    try:
        result = _client().audio.transcriptions.create(model=model, file=audio_file)
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI transcription failed: {exc}") from exc

    text = getattr(result, "text", "") or str(result)
    return text.strip()


def text_to_speech(
    text: str,
    *,
    voice: str = "coral",
    model: str = TTS_MODEL,
    instructions: str = "Speak clearly in a professional data analyst tone.",
) -> bytes:
    """Generate spoken audio for an AI response."""
    text = clean_text(text)
    if not text:
        raise ValueError("No text was provided for speech generation.")

    try:
        response = _client().audio.speech.create(
            model=model,
            voice=voice,
            input=text[:12_000],
            instructions=instructions,
        )
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI speech generation failed: {exc}") from exc

    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content
    return bytes(response)

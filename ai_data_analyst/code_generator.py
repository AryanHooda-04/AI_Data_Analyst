"""AI-assisted SQL and Pandas code generation."""

from __future__ import annotations

import pandas as pd

from ai_engine import complete_with_messages
from utils import clean_text, dataframe_context, format_schema


def generate_code(
    df: pd.DataFrame,
    user_request: str,
    *,
    model: str = "gpt-5.2",
    reasoning_effort: str = "none",
    max_tokens: int = 1_400,
    context_max_chars: int = 16_000,
) -> str:
    """Generate SQL and Pandas code for an analytical request."""
    user_request = clean_text(user_request)
    if not user_request:
        raise ValueError("Please describe the SQL or Pandas code you want generated.")

    prompt = f"""
You are an expert analytics engineer.

The uploaded dataset is available in two ways:
- SQL table name: uploaded_data
- Pandas DataFrame variable: df

Schema:
{format_schema(df)}

Sample and summary:
{dataframe_context(df, max_chars=context_max_chars)}

User request:
{user_request}

Generate clean, executable code with explanation.
Return exactly these sections:
1. SQL query
2. Pandas code
3. Short explanation

Rules:
- Use only columns present in the schema.
- Quote SQL identifiers with double quotes when names contain spaces or special characters.
- Prefer readable Pandas method chains.
- Do not invent tables, columns, credentials, or external files.
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You generate accurate SQL and Pandas for data analysis. "
                "Be concise, executable, and explicit about assumptions."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return complete_with_messages(
        messages,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=0.1,
        max_tokens=max_tokens,
    )

"""
synthetic_labeling.py

Module for synthetic labeling of a database using an LLM.
Includes functions to label rows using the OpenAI API and to load and upsert labeled data
using the Peewee ORM.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from config import get_settings
from litellm import acompletion
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from logger import logger
from models import LabeledNewsModel, NewsLabel, NewsModel, db_client
from peewee import Database, Model
from pydantic import BaseModel

settings = get_settings()


async def synthetic_label(
    rows: List[Dict[str, Any]],
    batch_size: int,
    available_labels: List[NewsLabel],
) -> List[str]:
    """
    Asynchronously labels all rows using the LiteLLM API's structured output feature.
    Processes rows in batches to avoid overwhelming the LLM.

    Args:
        rows: List of dictionaries representing rows (each key is a column name).
        batch_size: Number of rows to label in one API call.
        available_labels: List of NewsLabel models containing label names and descriptions.
    Returns:
        A list of labels corresponding to each row in the input.
    """
    if not rows:
        return []

    all_labels = []
    total_batches = (len(rows) + batch_size - 1) // batch_size

    # Check if the model supports response_format
    model = settings.llm_model
    supported_params = get_supported_openai_params(model=model)
    if not supported_params or "response_format" not in supported_params:
        raise ValueError(f"Model {model} does not support response_format parameter")

    class LabelResponse(BaseModel):
        labels: List[str]

    # Process each batch
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(rows))
        batch_rows = rows[start:end]

        # Craft the prompt using best practices
        columns = list(batch_rows[0].keys())
        header = ", ".join(columns)

        # Create a structured, informative prompt
        prompt = f"""
## Task: News Article Classification

### Available Labels
{"\n".join([f"- **{label.label}**: {label.description}" for label in available_labels])}

### Data Format
Each news article has these fields: {header}

### Instructions
You are an expert news classifier with deep knowledge of content categorization.
Your task is to analyze each article and assign the most appropriate label from the options above.

Think step by step:
1. Read each article carefully
2. Consider the title, content, and any other relevant fields
3. Compare the content against each label definition
4. Choose the single most appropriate label that best captures the article's main category

### Articles to Classify
{"\n\n".join([f"**Article {idx}**:\n" + "\n".join([f"- {k}: {v}" for k, v in row.items()]) for idx, row in enumerate(batch_rows, start=1)])}

### Output Format
Respond with a JSON object containing a "labels" array with exactly {len(batch_rows)} label(s), one for each article in the order presented. Choose only from the valid label names (not descriptions).
"""

        # Call LiteLLM API
        response = await acompletion(
            api_key=settings.openai_api_key.get_secret_value(),
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise news classification assistant capable of"
                        " analyzing articles and assigning accurate labels based on"
                        " content."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format=LabelResponse,
        )

        # Parse the output
        try:
            content = response.choices[0].message.content  # type: ignore
            json_response = json.loads(str(content))
            labels_model = LabelResponse(**json_response)
            all_labels.extend(labels_model.labels)
        except Exception as e:
            logger.error("Error parsing LLM response: %s", response)
            raise ValueError("Error parsing LLM response") from e

    return all_labels


async def load_data(
    ModelClass: type[Model], limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Asynchronously loads rows from a table specified by table_name using Peewee.

    Args:
        table_name: The name of the table to load data from.
        limit: Optional maximum number of rows to fetch.
    Returns:
        A list of dictionaries representing rows.
    """

    def query_rows() -> List[Dict[str, Any]]:
        if limit:
            q = ModelClass.select().limit(limit).dicts()
        else:
            q = ModelClass.select().dicts()
        return q

    rows = await asyncio.to_thread(query_rows)
    final_rows = []
    for row in rows:
        for key, value in row.items():
            row[key] = str(value)
        final_rows.append(row)
    return final_rows


async def upsert_labeled_data(
    peewee_db_client: Database,
    ModelClass: type[Model],
    rows: List[Dict[str, Any]],
    labels: List[str],
    check_existing: bool = False,
    primary_key: Optional[str] = None,
) -> None:
    """
    Asynchronously upserts labeled rows into the specified table.

    For each row and its corresponding label, if check_existing is True and primary_key is provided,
    the function checks if an entry with the same primary key exists in the table and has a label.
    If it exists and has a label, the row is skipped. Otherwise, the row is inserted.

    Args:
        peewee_db_client: The database client to use.
        ModelClass: The Peewee model class to use for the table.
        rows: List of dictionaries representing the original rows.
        labels: List of labels corresponding to each row.
        check_existing: Whether to check for existing labeled entries. Defaults to False.
        primary_key: The name of the primary key column. Required if check_existing is True.
    """
    if len(rows) != len(labels):
        raise ValueError("The number of rows and labels must match.")

    if check_existing and primary_key is None:
        raise ValueError("primary_key must be provided if check_existing is True.")

    def process_rows() -> None:
        try:
            peewee_db_client.connect(reuse_if_open=True)
            peewee_db_client.create_tables([ModelClass], safe=True)
            with peewee_db_client.atomic():
                for row, label in zip(rows, labels):
                    if check_existing:
                        assert (
                            primary_key is not None
                        ), "primary_key must be provided if check_existing is True."
                        # Check if label exists
                        existing = ModelClass.get_or_none(
                            (getattr(ModelClass, primary_key) == row[primary_key])
                            & (getattr(ModelClass, "label").is_null(False))
                        )
                        if existing:
                            continue  # Skip if entry exists with a label

                    data_to_insert = {**row, "label": label}
                    # Assuming the primary key conflict should replace the existing entry
                    # Use insert().on_conflict_replace() if that's the desired behavior
                    # or handle conflicts appropriately based on requirements.
                    ModelClass.insert(**data_to_insert).execute()
        finally:
            if not peewee_db_client.is_closed():
                peewee_db_client.close()

    await asyncio.to_thread(process_rows)


if __name__ == "__main__":

    async def main():
        news_rows = await load_data(NewsModel, limit=10)
        logger.info("Loaded %d news rows", len(news_rows))
        labels = await synthetic_label(
            news_rows,
            batch_size=5,
            available_labels=[
                NewsLabel(
                    label="Freshman",
                    description="News relevant to first-year university students",
                ),
                NewsLabel(
                    label="Senior",
                    description="News relevant to final year university students",
                ),
            ],
        )
        logger.info("Labeled %d news rows", len(labels))
        await upsert_labeled_data(
            db_client,
            LabeledNewsModel,
            news_rows,
            labels,
            check_existing=True,
            primary_key="code",
        )
        logger.info("Upserted %d labeled news rows", len(labels))

    asyncio.run(main())

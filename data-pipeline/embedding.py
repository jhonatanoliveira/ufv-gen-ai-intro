"""
data_embedding.py

Module for vectorizing labeled news data and storing it in a ChromaDB vector database.
This allows for semantic search and retrieval of news articles.
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List

import chromadb
from chromadb.errors import NotFoundError
from logger import logger
from models import LabeledNewsModel


async def load_labeled_data() -> List[Dict[str, Any]]:
    """
    Asynchronously load all labeled news data from the database.

    Returns:
        A list of dictionaries representing labeled news rows.
    """

    def query_rows() -> List[Dict[str, Any]]:
        return LabeledNewsModel.select().dicts()

    rows = await asyncio.to_thread(query_rows)
    return rows


def create_vector_store(
    rows: List[Dict[str, Any]],
    collection_name: str = "news_articles",
    persist_directory: str = "chroma_db",
) -> chromadb.Collection:
    """
    Create a vector store from the labeled news data.

    Args:
        rows: List of dictionaries representing labeled news rows.
        collection_name: Name of the collection to create.
        persist_directory: Directory to persist the vector store.

    Returns:
        The created ChromaDB collection.
    """
    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Create persistent client
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        logger.info("Using existing collection: %s", collection_name)
    except NotFoundError:
        collection = client.create_collection(name=collection_name)
        logger.info("Created new collection: %s", collection_name)

    # Prepare data for vectorization
    documents = []
    ids = []
    metadatas = []

    for row in rows:
        # Use content as the document
        content = row.get("content", "")
        if not content:
            logger.warning("Skipping row with empty content: %s", row.get("code"))
            continue

        # Use code as the document ID
        doc_id = str(row.get("code", ""))
        if not doc_id:
            logger.warning("Skipping row with missing code")
            continue

        # Create metadata from other fields
        metadata = {
            "code": doc_id,
            "category": str(row.get("category", "")),
            "title": str(row.get("title", "")),
            "date": str(row.get("date", "")),
            "location": str(row.get("location", "")),
            "label": str(row.get("label", "")),
            "embedded_at": str(datetime.now()),
        }

        documents.append(content)
        ids.append(doc_id)
        metadatas.append(metadata)

    # Add documents in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            ids=ids[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )
        logger.info(
            "Added batch %d with %d documents", i // batch_size + 1, end_idx - i
        )

    return collection


def print_collection_stats(collection: chromadb.Collection) -> None:
    """
    Print statistics about the collection.

    Args:
        collection: ChromaDB collection to print stats for.
    """
    # Get collection data
    count = collection.count()

    # Sample a document to get embedding dimensions if available
    if count > 0:
        sample = collection.get(limit=1)

        # Get embedding dimensions if available
        dimensions = "unknown"
        # Explicitly request embeddings when retrieving the sample
        embedding_sample = collection.get(limit=1, include=["embeddings"])
        embeddings = embedding_sample.get("embeddings")
        if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
            dimensions = len(embeddings[0])

        # Print stats
        logger.info("Collection '%s' statistics:", collection.name)
        logger.info("  - Total documents: %d", count)
        logger.info("  - Embedding dimensions: %s", dimensions)

        # Print sample metadata fields
        metadatas = sample.get("metadatas")
        if metadatas and len(metadatas) > 0:
            metadata = metadatas[0]
            logger.info("  - Metadata fields: %s", ", ".join(metadata.keys()))

            # Count documents by label
            if "label" in metadata:
                # Use ChromaDB's query interface to get label stats
                try:
                    all_data = collection.get()
                    metadatas_list = all_data.get("metadatas") if all_data else None

                    if metadatas_list:
                        labels = {}
                        for meta in metadatas_list:
                            if meta is not None:
                                label = meta.get("label", "unknown")
                                labels[label] = labels.get(label, 0) + 1

                        logger.info("  - Documents by label:")
                        for label, label_count in labels.items():
                            logger.info("      %s: %d", label, label_count)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Error getting label statistics: %s", e)
    else:
        logger.info("Collection '%s' is empty", collection.name)


async def main():
    """Main function to load data, create vector store, and print stats."""
    # Load labeled data
    logger.info("Loading labeled news data from database...")
    rows = await load_labeled_data()
    logger.info("Loaded %d labeled news articles", len(rows))

    # Create vector store
    logger.info("Creating vector store...")
    collection = create_vector_store(rows)

    # Print stats
    print_collection_stats(collection)

    logger.info("Data embedding complete!")


if __name__ == "__main__":
    asyncio.run(main())

"""
Create an Atlas Vector Search index for Gemini-size embeddings (768d, cosine).

Requires MongoDB Atlas 7+ and PyMongo 4.5+ (create_search_index on the driver).
Set MONGODB_ATLAS_SRV in .env. Optional: MONGODB_DB_NAME, MONGODB_COLLECTION,
MONGODB_EMBEDDING_PATH (default path for the vector field: 'embedding').
"""

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGODB_ATLAS_SRV = (os.environ.get("MONGODB_ATLAS_SRV") or "").strip()
MONGODB_DB_NAME = (os.environ.get("MONGODB_DB_NAME") or "weld_inspection").strip()
MONGODB_COLLECTION = (os.environ.get("MONGODB_COLLECTION") or "documents").strip()
EMBEDDING_PATH = (os.environ.get("MONGODB_EMBEDDING_PATH") or "embedding").strip()

INDEX_NAME = "weld_inspection_index"
DIMENSIONS = 768


def main() -> int:
    if not MONGODB_ATLAS_SRV:
        print("MONGODB_ATLAS_SRV is not set. Add it to your .env file.", file=sys.stderr)
        return 1

    try:
        client = MongoClient(MONGODB_ATLAS_SRV, serverSelectionTimeoutMS=15_000)
        client.admin.command("ping")
    except PyMongoError as exc:
        print(f"Could not connect to MongoDB Atlas: {exc}", file=sys.stderr)
        return 1

    coll = client[MONGODB_DB_NAME][MONGODB_COLLECTION]

    if MONGODB_COLLECTION not in client[MONGODB_DB_NAME].list_collection_names():
        client[MONGODB_DB_NAME].create_collection(MONGODB_COLLECTION)

    existing = {doc.get("name") for doc in coll.list_search_indexes() if doc.get("name")}
    if INDEX_NAME in existing:
        print(
            f"Index '{INDEX_NAME}' already exists on "
            f"{MONGODB_DB_NAME}.{MONGODB_COLLECTION} — nothing to do."
        )
        client.close()
        return 0

    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": EMBEDDING_PATH,
                    "numDimensions": DIMENSIONS,
                    "similarity": "cosine",
                }
            ]
        },
        name=INDEX_NAME,
        type="vectorSearch",
    )

    try:
        created = coll.create_search_index(model)
    except PyMongoError as exc:
        print(f"create_search_index failed: {exc}", file=sys.stderr)
        client.close()
        return 1

    print(
        f"Created vector search index '{created}' on "
        f"{MONGODB_DB_NAME}.{MONGODB_COLLECTION} "
        f"(path={EMBEDDING_PATH!r}, dimensions={DIMENSIONS}, similarity=cosine)."
    )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

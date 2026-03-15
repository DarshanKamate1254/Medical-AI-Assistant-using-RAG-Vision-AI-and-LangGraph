"""
seed_pinecone.py - One-time script to seed medical knowledge into Pinecone.

Run:  python seed_pinecone.py
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s – %(message)s")

from config import validate_config
from rag import seed_knowledge_base

def main():
    missing = validate_config()
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    print("Seeding Pinecone with medical knowledge…")
    seed_knowledge_base(force=True)
    print("Done! Your Pinecone index is ready.")

if __name__ == "__main__":
    main()

import os
import json
from pathlib import Path

from whoosh import index
from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import And, Or, Term
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_FILE = os.path.join(BASE_DIR, "data", "classification", "labels.json")
INDEX_DIR = os.path.join(BASE_DIR, "data", "search_index")

def create_schema():
    return Schema(
        path=ID(stored=True, unique=True),
        pdf_path=ID(stored=True),
        filename=TEXT(stored=True),
        content=TEXT(analyzer=StemmingAnalyzer(), stored=False),
        preview=STORED,
        entities=KEYWORD(stored=True, commas=True),
        file_type=ID(stored=True),
        evidence_of=KEYWORD(stored=True, commas=True),
        cluster_id=ID(stored=True),
        has_hidden_content=ID(stored=True),
        recovered_text=TEXT(analyzer=StemmingAnalyzer(), stored=True)
    )

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def build_index():
    print("=" * 60)
    print("Building Search Index")
    print("=" * 60)

    os.makedirs(INDEX_DIR, exist_ok=True)

    schema = create_schema()
    ix = index.create_in(INDEX_DIR, schema)

    labels = load_labels()
    print(f"Found {len(labels)} documents to index")

    if not labels:
        print("No documents found. Run classification first.")
        return ix

    writer = ix.writer()

    for doc in tqdm(labels, desc="Indexing"):
        try:
            text = doc.get("text", "")
            if len(text) < 20:
                continue

            pdf_path = doc.get("pdf_path", "")
            filename = doc.get("filename", "")
            entities = doc.get("entities", [])
            file_type = doc.get("file_type", "unknown")
            evidence_of = doc.get("evidence_of", [])
            cluster_id = str(doc.get("cluster_id", ""))
            preview = doc.get("preview", text[:500])
            has_hidden = doc.get("has_hidden_content", False)
            recovered = doc.get("recovered_text", "")
            
            full_content = text
            if recovered:
                full_content = f"{text}\n\n[RECOVERED HIDDEN TEXT]\n{recovered}"

            writer.add_document(
                path=pdf_path,
                pdf_path=pdf_path,
                filename=filename,
                content=full_content,
                preview=preview,
                entities=",".join(entities) if entities else "",
                file_type=file_type,
                evidence_of=",".join(evidence_of) if evidence_of else "",
                cluster_id=cluster_id,
                has_hidden_content="yes" if has_hidden else "no",
                recovered_text=recovered
            )

        except Exception as e:
            print(f"Error indexing {doc.get('filename', 'unknown')}: {e}")

    writer.commit()
    print(f"\nIndex created at {INDEX_DIR}")
    print(f"Total documents indexed: {ix.doc_count()}")

    return ix

def search(query_str, entity_filter=None, file_type_filter=None, limit=20):
    if not os.path.exists(INDEX_DIR):
        print("Index not found. Run build_index() first.")
        return []

    ix = index.open_dir(INDEX_DIR)

    with ix.searcher() as searcher:
        parser = MultifieldParser(["content", "filename", "entities"], ix.schema)
        query = parser.parse(query_str)

        if entity_filter:
            entity_query = Term("entities", entity_filter)
            query = And([query, entity_query])
        
        if file_type_filter:
            ft_query = Term("file_type", file_type_filter)
            query = And([query, ft_query])

        results = searcher.search(query, limit=limit)

        output = []
        for hit in results:
            output.append({
                "path": hit["path"],
                "filename": hit["filename"],
                "preview": hit["preview"],
                "entities": hit["entities"].split(",") if hit["entities"] else [],
                "file_type": hit["file_type"],
                "evidence_of": hit["evidence_of"].split(",") if hit["evidence_of"] else [],
                "score": hit.score
            })

        return output

def interactive_search():
    print("\n" + "=" * 60)
    print("Interactive Search")
    print("=" * 60)
    print("Commands:")
    print("  search <query>     - Search for documents")
    print("  entity <name>      - Filter by entity")
    print("  entities           - List entities")
    print("  quit               - Exit")
    print("=" * 60)

    if not os.path.exists(INDEX_DIR):
        print("Index not found. Building index first...")
        build_index()

    ix = index.open_dir(INDEX_DIR)
    current_entity = None

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "quit" or cmd == "exit":
            break

        elif cmd == "entities":
            with ix.searcher() as searcher:
                all_entities = set()
                for doc in searcher.all_stored_fields():
                    entities = doc.get("entities", "")
                    if entities:
                        all_entities.update(entities.split(","))

                print("\nEntities found:")
                for entity in sorted(all_entities)[:50]:
                    print(f"  - {entity}")

        elif cmd == "entity":
            current_entity = arg if arg else None
            if current_entity:
                print(f"Filter set to entity: {current_entity}")
            else:
                print("Entity filter cleared")

        elif cmd == "search" or cmd not in ["entities", "entity", "quit", "exit"]:
            query = arg if cmd == "search" else user_input

            if not query:
                print("Please provide a search query")
                continue

            results = search(query, entity_filter=current_entity)

            if not results:
                print("No results found")
            else:
                print(f"\nFound {len(results)} results:")
                for i, r in enumerate(results, 1):
                    print(f"\n--- Result {i} (score: {r['score']:.2f}) ---")
                    print(f"File: {r['filename']}")
                    print(f"Type: {r['file_type']} | Entities: {', '.join(r['entities'][:5])}")
                    print(f"Preview: {r['preview'][:200]}...")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        results = search(query)
        for r in results:
            print(f"\n{r['filename']} (score: {r['score']:.2f})")
            print(f"Type: {r['file_type']} | Entities: {', '.join(r['entities'][:5])}")
            print(f"{r['preview'][:200]}...")
    else:
        interactive_search()

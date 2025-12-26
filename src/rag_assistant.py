import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_FILE = os.path.join(BASE_DIR, "data", "classification", "labels.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "classification", "embeddings.pkl")
RAG_INDEX_FILE = os.path.join(BASE_DIR, "data", "rag_index.json")
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "mistral"

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

class DocumentChunk:
    def __init__(self, text: str, filename: str, filepath: str, chunk_id: int):
        self.text = text
        self.filename = filename
        self.filepath = filepath
        self.chunk_id = chunk_id
        self.embedding: Optional[np.ndarray] = None

class RAGAssistant:

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None
        self.is_ready = False
        self.ollama_available = False

    def check_ollama(self) -> bool:
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                self.ollama_available = MODEL_NAME in model_names
                return self.ollama_available
        except:
            pass
        return False

    def initialize(self, progress_callback=None) -> bool:
        if not HAS_EMBEDDINGS:
            print("sentence-transformers not installed")
            return False

        if not self.check_ollama():
            print(f"Ollama not running or {MODEL_NAME} model not installed")
            return False

        if os.path.exists(RAG_INDEX_FILE):
            return self._load_index()
        else:
            return self._build_index(progress_callback)

    def _load_index(self) -> bool:
        try:
            with open(RAG_INDEX_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.chunks = []
            for chunk_data in data["chunks"]:
                chunk = DocumentChunk(
                    text=chunk_data["text"],
                    filename=chunk_data["filename"],
                    filepath=chunk_data["filepath"],
                    chunk_id=chunk_data["chunk_id"]
                )
                self.chunks.append(chunk)

            embeddings_file = RAG_INDEX_FILE.replace(".json", "_embeddings.npy")
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)

            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_ready = True
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def _build_index(self, progress_callback=None) -> bool:
        try:
            if progress_callback:
                progress_callback("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            if not os.path.exists(LABELS_FILE):
                print(f"Labels file not found: {LABELS_FILE}")
                return False
            
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            if not labels:
                print("No documents found in labels file")
                return False

            if progress_callback:
                progress_callback(f"Processing {len(labels)} documents...")

            self.chunks = []
            for i, doc in enumerate(labels):
                if progress_callback and i % 100 == 0:
                    progress_callback(f"Processing document {i+1}/{len(labels)}")

                try:
                    content = doc.get("text", "")
                    recovered = doc.get("recovered_text", "")
                    
                    if recovered:
                        content = f"{content}\n\n[RECOVERED HIDDEN TEXT]\n{recovered}"
                    
                    if len(content) < 50:
                        continue
                    
                    filename = doc.get("filename", "unknown")
                    filepath = doc.get("pdf_path", "")

                    chunk_size = 500
                    overlap = 100
                    text_chunks = self._chunk_text(content, chunk_size, overlap)

                    for chunk_id, chunk_text in enumerate(text_chunks):
                        if len(chunk_text.strip()) > 50:
                            self.chunks.append(DocumentChunk(
                                text=chunk_text,
                                filename=filename,
                                filepath=filepath,
                                chunk_id=chunk_id
                            ))
                except Exception as e:
                    continue

            if not self.chunks:
                print("No valid chunks created")
                return False

            if progress_callback:
                progress_callback(f"Creating embeddings for {len(self.chunks)} chunks...")

            texts = [c.text for c in self.chunks]
            self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

            if progress_callback:
                progress_callback("Saving RAG index...")

            self._save_index()
            self.is_ready = True
            return True

        except Exception as e:
            print(f"Error building index: {e}")
            return False

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk)
            start = end - overlap

        return chunks

    def _save_index(self):
        os.makedirs(os.path.dirname(RAG_INDEX_FILE), exist_ok=True)

        data = {
            "chunks": [
                {
                    "text": c.text,
                    "filename": c.filename,
                    "filepath": c.filepath,
                    "chunk_id": c.chunk_id
                }
                for c in self.chunks
            ]
        }

        with open(RAG_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        embeddings_file = RAG_INDEX_FILE.replace(".json", "_embeddings.npy")
        np.save(embeddings_file, self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.is_ready or self.embeddings is None:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "filename": chunk.filename,
                "filepath": chunk.filepath,
                "score": float(similarities[idx])
            })

        return results

    def query(self, question: str, top_k: int = 5) -> Dict:
        if not self.is_ready:
            return {
                "answer": "AI Assistant is not initialized. Please run setup first.",
                "sources": [],
                "error": True
            }

        relevant_chunks = self.retrieve(question, top_k)

        if not relevant_chunks:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "error": True
            }

        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"[Document: {chunk['filename']}]\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are an AI assistant analyzing the DOJ Epstein Files. Answer the user's question based ONLY on the provided document excerpts. Be factual and cite which document(s) your information comes from. If the documents don't contain enough information to answer, say so.

DOCUMENT EXCERPTS:
{context}

USER QUESTION: {question}

ANSWER (cite documents by filename):"""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1000
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                answer = response.json().get("response", "")
                return {
                    "answer": answer,
                    "sources": [
                        {"filename": c["filename"], "filepath": c["filepath"], "score": c["score"]}
                        for c in relevant_chunks
                    ],
                    "error": False
                }
            else:
                return {
                    "answer": f"Ollama error: {response.status_code}",
                    "sources": [],
                    "error": True
                }

        except requests.exceptions.Timeout:
            return {
                "answer": "Request timed out. The model may be loading.",
                "sources": [],
                "error": True
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "error": True
            }

    def stream_query(self, question: str, top_k: int = 5):
        if not self.is_ready:
            yield {"type": "error", "content": "AI Assistant is not initialized."}
            return

        relevant_chunks = self.retrieve(question, top_k)

        if not relevant_chunks:
            yield {"type": "error", "content": "No relevant documents found."}
            return

        yield {
            "type": "sources",
            "content": [
                {"filename": c["filename"], "filepath": c["filepath"], "score": c["score"]}
                for c in relevant_chunks
            ]
        }

        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"[Document: {chunk['filename']}]\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are an AI assistant analyzing the DOJ Epstein Files. Answer the user's question based ONLY on the provided document excerpts. Be factual and cite which document(s) your information comes from. If the documents don't contain enough information to answer, say so.

DOCUMENT EXCERPTS:
{context}

USER QUESTION: {question}

ANSWER (cite documents by filename):"""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1000
                    }
                },
                stream=True,
                timeout=120
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield {"type": "token", "content": data["response"]}
                    if data.get("done"):
                        break

            yield {"type": "done", "content": ""}

        except Exception as e:
            yield {"type": "error", "content": str(e)}

assistant = RAGAssistant()

def get_assistant() -> RAGAssistant:
    return assistant

if __name__ == "__main__":
    print("Initializing RAG Assistant...")

    def progress(msg):
        print(f"  {msg}")

    if assistant.initialize(progress):
        print("\nAssistant ready!")
        print(f"Loaded {len(assistant.chunks)} document chunks")

        while True:
            question = input("\nAsk a question (or 'quit'): ")
            if question.lower() == 'quit':
                break

            result = assistant.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources:")
            for src in result['sources']:
                print(f"  - {src['filename']} (score: {src['score']:.3f})")
    else:
        print("Failed to initialize assistant")

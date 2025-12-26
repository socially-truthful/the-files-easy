import os
import sys
import json
import pickle
import warnings
import hashlib
import threading
import multiprocessing
from functools import lru_cache
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from queue import Queue
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

sys.path.append(os.path.dirname(__file__))
from topic_classifier import (
    load_pdfs_by_page, extract_text_from_pdf, ensure_ollama_ready,
    check_ollama, OLLAMA_URL, MODEL_OPTIONS, select_model,
    load_known_labels, save_known_labels, update_known_labels,
    build_existing_labels_prompt, filter_garbage, PHOTO_KEYWORDS,
    AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, MEDIA_KEYWORDS, GARBAGE_TERMS,
    get_ollama_threads, detect_gpu_vram, get_optimal_workers,
    guess_file_type
)

DOWNLOADS_DIR = os.path.join(BASE_DIR, "data", "downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "classification")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "classification", "embeddings.pkl")
LABELS_FILE = os.path.join(BASE_DIR, "data", "classification", "labels.json")
KNOWN_LABELS_FILE = os.path.join(BASE_DIR, "data", "classification", "known_labels.json")
HYBRID_CACHE_FILE = os.path.join(BASE_DIR, "data", "classification", "hybrid_cache.pkl")
PAGE_EXTRACTION_CACHE = os.path.join(BASE_DIR, "data", "classification", "page_extraction_cache.pkl")
CLASSIFICATION_CACHE = os.path.join(BASE_DIR, "data", "classification", "classification_cache.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EMBEDDING_BATCH_SIZE = 64
LLM_BATCH_SIZE = 8
PDF_PROCESS_WORKERS = max(4, multiprocessing.cpu_count())

_page_cache = {}
_page_cache_lock = threading.Lock()
_classification_cache = {}
_classification_cache_lock = threading.Lock()
_embedding_cache = {}
_embedding_cache_lock = threading.Lock()


def get_text_hash(text):
    return hashlib.md5(text[:2000].encode('utf-8', errors='ignore')).hexdigest()


def load_page_extraction_cache():
    global _page_cache
    try:
        if os.path.exists(PAGE_EXTRACTION_CACHE):
            with open(PAGE_EXTRACTION_CACHE, 'rb') as f:
                _page_cache = pickle.load(f)
            print(f"[Cache] Loaded {len(_page_cache)} cached page extractions")
            return True
    except Exception as e:
        print(f"[Cache] Could not load page cache: {e}")
    return False


def save_page_extraction_cache():
    global _page_cache
    try:
        os.makedirs(os.path.dirname(PAGE_EXTRACTION_CACHE), exist_ok=True)
        with open(PAGE_EXTRACTION_CACHE, 'wb') as f:
            pickle.dump(_page_cache, f)
        print(f"[Cache] Saved {len(_page_cache)} page extractions to cache")
    except Exception as e:
        print(f"[Cache] Could not save page cache: {e}")


def load_classification_cache():
    global _classification_cache
    try:
        if os.path.exists(CLASSIFICATION_CACHE):
            with open(CLASSIFICATION_CACHE, 'rb') as f:
                _classification_cache = pickle.load(f)
            print(f"[Cache] Loaded {len(_classification_cache)} cached classifications")
            return True
    except Exception as e:
        print(f"[Cache] Could not load classification cache: {e}")
    return False


def save_classification_cache():
    global _classification_cache
    try:
        os.makedirs(os.path.dirname(CLASSIFICATION_CACHE), exist_ok=True)
        with open(CLASSIFICATION_CACHE, 'wb') as f:
            pickle.dump(_classification_cache, f)
        print(f"[Cache] Saved {len(_classification_cache)} classifications to cache")
    except Exception as e:
        print(f"[Cache] Could not save classification cache: {e}")

HIGH_IMPORTANCE_KEYWORDS = {
    'interview', 'deposition', 'testimony', 'witness', 'subpoena', 'affidavit',
    'confession', 'admission', 'allegation', 'accusation', 'investigation',
    'interrogation', 'statement', 'declaration', 'evidence', 'exhibit',
    'court', 'legal', 'attorney', 'prosecutor', 'defense', 'judge', 'magistrate'
}

INCriminating_KEYWORDS = {
    'sexual', 'assault', 'abuse', 'minor', 'underage', 'victim', 'trafficking',
    'prostitution', 'massage', 'therapist', 'naked', 'nude', 'inappropriate',
    'illegal', 'crime', 'criminal', 'felony', 'misdemeanor', 'charged',
    'arrested', 'convicted', 'indicted', 'pleaded', 'guilty', 'innocent'
}

NAME_PATTERNS = {
    r'[A-Z][a-z]+ [A-Z][a-z]+',
    r'[A-Z]\. [A-Z][a-z]+',
    r'[A-Z][a-z]+, [A-Z]\.',
}

STANDARD_CORRESPONDENCE_KEYWORDS = {
    'letter', 'memo', 'email', 'fax', 'cover sheet', 'transmittal',
    'correspondence', 'communication', 'message', 'note', 'reminder',
    'notification', 'announcement', 'invitation', 'greeting', 'regards'
}

class HybridClassifier:
    def __init__(self, ollama_model=None):
        self.ollama_model = ollama_model
        self.embedding_model = None
        self.knn_model = None
        self.tfidf_vectorizer = None
        self.document_embeddings = None
        self.document_texts = []
        self.knn_labels = []
        self.is_initialized = False
        
    def initialize(self, progress_callback=None):
        if progress_callback:
            progress_callback("Loading embedding model...")
            
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Warning: sentence-transformers not available")
            return False
            
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.document_embeddings = data["embeddings"]
                    self.document_texts = data["texts"]
                    self.knn_labels = data.get("labels", [])
                    
                if progress_callback:
                    progress_callback(f"Loaded {len(self.document_texts)} cached documents")
                    
                if len(self.document_texts) >= 10:
                    self._initialize_knn()
                    
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error loading cache: {e}")
                    
        self.is_initialized = True
        return True
        
    def _initialize_knn(self):
        if not HAS_SKLEARN or self.document_embeddings is None:
            return
            
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn_model.fit(self.document_embeddings)
        
    def _calculate_importance_score(self, text, filename):
        text_lower = text.lower()
        filename_lower = filename.lower()
        score = 0.0
        
        for keyword in HIGH_IMPORTANCE_KEYWORDS:
            if keyword in text_lower:
                score += 2.0
            if keyword in filename_lower:
                score += 1.0
                
        for keyword in INCriminating_KEYWORDS:
            if keyword in text_lower:
                score += 3.0
            if keyword in filename_lower:
                score += 1.5
                
        import re
        name_count = 0
        for pattern in NAME_PATTERNS:
            matches = re.findall(pattern, text)
            name_count += len(matches)
            
        if name_count > 2:
            score += name_count * 0.5
            
        if len(text) > 1000:
            score += 0.5
        elif len(text) > 5000:
            score += 1.0
            
        for keyword in STANDARD_CORRESPONDENCE_KEYWORDS:
            if keyword in filename_lower:
                score -= 1.0
            if keyword in text_lower and name_count < 2:
                score -= 0.5
                
        return max(0, score)
        
    def batch_encode_embeddings(self, texts, batch_size=EMBEDDING_BATCH_SIZE):
        if not self.embedding_model:
            return None
            
        results = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []
        with _embedding_cache_lock:
            for i, text in enumerate(texts):
                text_hash = get_text_hash(text[:1000])
                if text_hash in _embedding_cache:
                    results[i] = _embedding_cache[text_hash]
                else:
                    texts_to_encode.append(text[:1000])
                    indices_to_encode.append(i)
        if texts_to_encode:
            embeddings = self.embedding_model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            with _embedding_cache_lock:
                for idx, (orig_idx, text) in enumerate(zip(indices_to_encode, texts_to_encode)):
                    text_hash = get_text_hash(text)
                    _embedding_cache[text_hash] = embeddings[idx]
                    results[orig_idx] = embeddings[idx]
        
        return np.array(results)
    
    def batch_classify_with_knn(self, texts, k=3):
        if not self.knn_model or len(self.document_texts) == 0:
            return [None] * len(texts)
            
        try:
            text_embeddings = self.batch_encode_embeddings(texts)
            if text_embeddings is None:
                return [None] * len(texts)
            distances, indices = self.knn_model.kneighbors(text_embeddings)
            
            results = []
            for i in range(len(texts)):
                neighbor_labels = []
                for idx in indices[i]:
                    if idx < len(self.knn_labels):
                        neighbor_labels.append(self.knn_labels[idx])
                        
                if not neighbor_labels:
                    results.append(None)
                    continue
                    
                aggregated = {
                    "entities": [],
                    "file_type": "",
                    "evidence_of": [],
                    "confidence": 1.0 - np.mean(distances[i])
                }
                
                all_entities = []
                for label in neighbor_labels:
                    all_entities.extend(label.get("entities", []))
                if all_entities:
                    entity_counts = Counter(all_entities)
                    aggregated["entities"] = [ent for ent, count in entity_counts.most_common(5)]
                    
                file_types = [label.get("file_type", "document") for label in neighbor_labels]
                if file_types:
                    file_type_counts = Counter(file_types)
                    aggregated["file_type"] = file_type_counts.most_common(1)[0][0]
                    
                all_evidence = []
                for label in neighbor_labels:
                    all_evidence.extend(label.get("evidence_of", []))
                if all_evidence:
                    evidence_counts = Counter(all_evidence)
                    aggregated["evidence_of"] = [ev for ev, count in evidence_counts.most_common(3)]
                    
                results.append(aggregated)
            
            return results
            
        except Exception as e:
            print(f"Batch KNN classification error: {e}")
            return [None] * len(texts)
    
    def _classify_with_knn(self, text, k=3):
        if not self.knn_model or len(self.document_texts) == 0:
            return None
            
        try:
            text_hash = get_text_hash(text[:1000])
            with _embedding_cache_lock:
                if text_hash in _embedding_cache:
                    text_embedding = _embedding_cache[text_hash].reshape(1, -1)
                else:
                    text_embedding = self.embedding_model.encode([text[:1000]])
                    _embedding_cache[text_hash] = text_embedding[0]
            
            distances, indices = self.knn_model.kneighbors(text_embedding)
            
            neighbor_labels = []
            for idx in indices[0]:
                if idx < len(self.knn_labels):
                    neighbor_labels.append(self.knn_labels[idx])
                    
            if not neighbor_labels:
                return None
                
            aggregated = {
                "entities": [],
                "file_type": "",
                "evidence_of": [],
                "confidence": 1.0 - np.mean(distances[0])
            }
            
            all_entities = []
            for label in neighbor_labels:
                all_entities.extend(label.get("entities", []))
            if all_entities:
                entity_counts = Counter(all_entities)
                aggregated["entities"] = [ent for ent, count in entity_counts.most_common(5)]
                
            file_types = [label.get("file_type", "document") for label in neighbor_labels]
            if file_types:
                file_type_counts = Counter(file_types)
                aggregated["file_type"] = file_type_counts.most_common(1)[0][0]
                
            all_evidence = []
            for label in neighbor_labels:
                all_evidence.extend(label.get("evidence_of", []))
            if all_evidence:
                evidence_counts = Counter(all_evidence)
                aggregated["evidence_of"] = [ev for ev, count in evidence_counts.most_common(3)]
                
            return aggregated
            
        except Exception as e:
            print(f"KNN classification error: {e}")
            return None
            
    def _classify_with_llm(self, text, filename):
        if not HAS_REQUESTS or not check_ollama(self.ollama_model):
            return None
        cache_key = get_text_hash(text[:1200] + filename)
        with _classification_cache_lock:
            if cache_key in _classification_cache:
                return _classification_cache[cache_key]
            
        prompt = f"""Analyze this document from the Epstein case files.

{build_existing_labels_prompt()}
Filename: {filename}
Content: {text[:1200]}

Extract:
1. entities: Names of people mentioned (full names when possible)
2. evidence_of: What does this document provide evidence of? Focus on illegal activities, witness testimony, investigations, or important relationships.

Rules:
- Focus on extracting names and evidence of wrongdoing
- Look for witnesses, victims, perpetrators, investigators
- Identify evidence of crimes, cover-ups, or illegal activities
- Return ONLY valid JSON: {{"entities": [], "evidence_of": []}}"""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 150,
                        "num_gpu": 99,
                        "num_thread": get_ollama_threads()
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                start = result.find('{')
                end = result.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = result[start:end]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        classification = {
                            "entities": filter_garbage(parsed.get("entities", []) if isinstance(parsed.get("entities"), list) else []),
                            "evidence_of": filter_garbage(parsed.get("evidence_of", []) if isinstance(parsed.get("evidence_of"), list) else [])
                        }
                        update_known_labels(classification)
                        with _classification_cache_lock:
                            _classification_cache[cache_key] = classification
                        return classification
        except Exception as e:
            pass
            
        return None
    
    def batch_classify_with_llm(self, docs, max_concurrent=LLM_BATCH_SIZE):
        if not HAS_REQUESTS or not check_ollama(self.ollama_model):
            return [None] * len(docs)
        
        results = [None] * len(docs)
        docs_to_process = []
        indices_to_process = []
        with _classification_cache_lock:
            for i, doc in enumerate(docs):
                text = doc.get("text", "")
                filename = doc.get("filename", "")
                cache_key = get_text_hash(text[:1200] + filename)
                if cache_key in _classification_cache:
                    results[i] = _classification_cache[cache_key]
                else:
                    docs_to_process.append(doc)
                    indices_to_process.append(i)
        
        if not docs_to_process:
            return results
        def process_single_llm(doc):
            text = doc.get("text", "")
            filename = doc.get("filename", "")
            return self._classify_with_llm(text, filename)
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(process_single_llm, doc): i 
                      for i, doc in zip(indices_to_process, docs_to_process)}
            for future in as_completed(futures):
                orig_idx = futures[future]
                try:
                    results[orig_idx] = future.result(timeout=180)
                except Exception as e:
                    print(f"[LLM] Timeout/error on doc {orig_idx}: {e}")
                    results[orig_idx] = None
        
        return results
        
    def _classify_with_rules(self, text, filename):
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        entities = []
        evidence_of = []
        
        import re
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        potential_names = re.findall(name_pattern, text)
        entities = [name for name in potential_names if len(name.split()) == 2][:5]
        
        file_type = guess_file_type(filename_lower, text_lower)
        
        if any(kw in text_lower for kw in HIGH_IMPORTANCE_KEYWORDS):
            if 'interview' in text_lower or 'deposition' in text_lower:
                evidence_of.append("witness testimony")
            if 'court' in text_lower or 'legal' in text_lower:
                evidence_of.append("legal proceeding")
            if 'investigation' in text_lower:
                evidence_of.append("investigation")
                
        if any(kw in text_lower for kw in INCriminating_KEYWORDS):
            evidence_of.append("potential evidence of illegal activity")
            
        return {
            "entities": entities,
            "file_type": file_type,
            "evidence_of": evidence_of,
            "confidence": 0.5
        }
        
    def classify_document(self, text, filename, is_photo_page=False, has_hidden_content=False, recovered_text=""):
        text_lower = text.lower() if text else ""
        filename_lower = filename.lower() if filename else ""
        
        if any(filename_lower.endswith(ext) for ext in AUDIO_EXTENSIONS):
            return {
                "entities": [],
                "file_type": "audio",
                "evidence_of": ["audio recording"],
                "method": "rule",
                "confidence": 1.0
            }
        if any(filename_lower.endswith(ext) for ext in VIDEO_EXTENSIONS):
            return {
                "entities": [],
                "file_type": "video",
                "evidence_of": ["video recording"],
                "method": "rule",
                "confidence": 1.0
            }
        
        if is_photo_page or text.strip() == "[Image/Photo]":
            return {
                "entities": [],
                "file_type": "photograph",
                "evidence_of": ["photographic evidence"],
                "method": "rule",
                "confidence": 1.0
            }
        
        file_type = guess_file_type(filename_lower, text_lower)
            
        importance_score = self._calculate_importance_score(text, filename)
        
        if has_hidden_content and recovered_text:
            importance_score += 5.0
        
        if importance_score < 3.0:
            knn_result = self._classify_with_knn(text)
            if knn_result and knn_result.get("confidence", 0) > 0.6:
                knn_result["method"] = "knn"
                knn_result["file_type"] = file_type
                return knn_result
                
        if importance_score >= 3.0 and check_ollama(self.ollama_model):
            analysis_text = text
            if recovered_text:
                analysis_text = f"{text}\n\n[RECOVERED HIDDEN TEXT]\n{recovered_text}"
            
            llm_result = self._classify_with_llm(analysis_text, filename)
            if llm_result:
                llm_result["method"] = "llm"
                llm_result["importance_score"] = importance_score
                llm_result["file_type"] = file_type
                return llm_result
                
        rule_result = self._classify_with_rules(text, filename)
        rule_result["method"] = "rule"
        rule_result["importance_score"] = importance_score
        
        return rule_result
        
    def update_training_data(self, text, label):
        if not self.embedding_model:
            return
            
        try:
            embedding = self.embedding_model.encode([text[:1000]])[0]
            
            if self.document_embeddings is None:
                self.document_embeddings = np.array([embedding])
                self.document_texts = [text]
                self.knn_labels = [label]
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embedding])
                self.document_texts.append(text)
                self.knn_labels.append(label)
                
            if len(self.document_texts) >= 10:
                self._initialize_knn()
                
        except Exception as e:
            print(f"Error updating training data: {e}")
            
    def save_cache(self):
        try:
            cache_data = {
                "document_embeddings": self.document_embeddings,
                "document_texts": self.document_texts,
                "knn_labels": self.knn_labels,
                "embedding_model_name": EMBEDDING_MODEL
            }
            
            with open(HYBRID_CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            print(f"Error saving cache: {e}")
            
    def load_cache(self):
        try:
            if os.path.exists(HYBRID_CACHE_FILE):
                with open(HYBRID_CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                self.document_embeddings = cache_data.get("document_embeddings")
                self.document_texts = cache_data.get("document_texts", [])
                self.knn_labels = cache_data.get("knn_labels", [])
                
                if self.document_embeddings is not None and len(self.document_texts) > 0:
                    self._initialize_knn()
                    return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            
        return False

def create_hybrid_labels(documents, ollama_model=None, max_workers=12):
    from topic_classifier import load_classification_progress, clear_classification_progress, save_classification_progress
    
    classifier = HybridClassifier(ollama_model)
    
    print("Initializing hybrid classifier...")
    classifier.initialize()
    classifier.load_cache()
    load_classification_cache()
    load_page_extraction_cache()
    prev_results, completed_indices = load_classification_progress()
    results = [None] * len(documents)
    for i, res in enumerate(prev_results):
        if i < len(results) and res is not None:
            results[i] = res
    filename_to_idx = {doc.get("filename", ""): i for i, doc in enumerate(documents)}
    for res in prev_results:
        if res and res.get("filename"):
            idx = filename_to_idx.get(res["filename"])
            if idx is not None:
                results[idx] = res
                completed_indices.add(idx)
    
    if completed_indices:
        print(f"[Resume] Found {len(completed_indices)} previously classified documents")
        
    llm_count = 0
    knn_count = 0
    rule_count = 0
    cache_hits = 0
    
    print(f"Classifying {len(documents)} documents with optimized hybrid approach...")
    quick_classify_indices = []
    need_processing_indices = []
    
    for i, doc in enumerate(documents):
        if i in completed_indices:
            continue
            
        text = doc.get("text", "")
        filename = doc.get("filename", "").lower()
        is_image_page = doc.get("is_image_page", False)
        if any(filename.endswith(ext) for ext in AUDIO_EXTENSIONS):
            results[i] = {"entities": [], "file_type": "audio", "evidence_of": ["audio recording"], "method": "rule", "confidence": 1.0}
            rule_count += 1
        elif any(filename.endswith(ext) for ext in VIDEO_EXTENSIONS):
            results[i] = {"entities": [], "file_type": "video", "evidence_of": ["video recording"], "method": "rule", "confidence": 1.0}
            rule_count += 1
        elif is_image_page or text.strip() == "[Image/Photo]":
            results[i] = {"entities": [], "file_type": "photograph", "evidence_of": ["photographic evidence"], "method": "rule", "confidence": 1.0}
            rule_count += 1
        else:
            cache_key = get_text_hash(text[:1200] + filename)
            with _classification_cache_lock:
                if cache_key in _classification_cache:
                    results[i] = _classification_cache[cache_key].copy()
                    results[i]["method"] = "cached"
                    cache_hits += 1
                else:
                    need_processing_indices.append(i)
    
    print(f"  Quick classified: {len(documents) - len(need_processing_indices)} (cache hits: {cache_hits})")
    print(f"  Need processing: {len(need_processing_indices)}")
    
    if need_processing_indices:
        high_importance_docs = []
        low_importance_docs = []
        
        for i in need_processing_indices:
            doc = documents[i]
            importance = classifier._calculate_importance_score(doc.get("text", ""), doc.get("filename", ""))
            if doc.get("has_hidden_content"):
                importance += 5.0
            if importance >= 3.0:
                high_importance_docs.append((i, doc, importance))
            else:
                low_importance_docs.append((i, doc, importance))
        
        print(f"  High importance (LLM): {len(high_importance_docs)}")
        print(f"  Low importance (KNN/Rules): {len(low_importance_docs)}")
        if low_importance_docs and classifier.knn_model:
            print("  Running batch KNN classification...")
            low_texts = [doc.get("text", "") for _, doc, _ in low_importance_docs]
            knn_results = classifier.batch_classify_with_knn(low_texts)
            
            for j, (i, doc, importance) in enumerate(low_importance_docs):
                knn_result = knn_results[j]
                if knn_result and knn_result.get("confidence", 0) > 0.6:
                    knn_result["method"] = "knn"
                    knn_result["file_type"] = guess_file_type(doc.get("filename", "").lower(), doc.get("text", "").lower())
                    results[i] = knn_result
                    knn_count += 1
                else:
                    rule_result = classifier._classify_with_rules(doc.get("text", ""), doc.get("filename", ""))
                    rule_result["method"] = "rule"
                    rule_result["importance_score"] = importance
                    results[i] = rule_result
                    rule_count += 1
        elif low_importance_docs:
            for i, doc, importance in low_importance_docs:
                rule_result = classifier._classify_with_rules(doc.get("text", ""), doc.get("filename", ""))
                rule_result["method"] = "rule"
                rule_result["importance_score"] = importance
                results[i] = rule_result
                rule_count += 1
        if high_importance_docs and check_ollama(ollama_model):
            print(f"  Running batch LLM classification ({LLM_BATCH_SIZE} concurrent)...")
            from topic_classifier import save_classification_progress, clear_classification_progress
            batch_size = LLM_BATCH_SIZE * 4
            completed_indices = set(i for i, r in enumerate(results) if r is not None)
            batch_num = 0
            
            for batch_start in tqdm(range(0, len(high_importance_docs), batch_size), desc="LLM Batches"):
                batch = high_importance_docs[batch_start:batch_start + batch_size]
                batch_docs = []
                
                for i, doc, importance in batch:
                    analysis_text = doc.get("text", "")
                    recovered = doc.get("recovered_text", "")
                    if recovered:
                        analysis_text = f"{analysis_text}\n\n[RECOVERED HIDDEN TEXT]\n{recovered}"
                    batch_docs.append({"text": analysis_text, "filename": doc.get("filename", ""), "orig_idx": i, "importance": importance})
                
                llm_results = classifier.batch_classify_with_llm(batch_docs)
                
                for j, (i, doc, importance) in enumerate(batch):
                    llm_result = llm_results[j]
                    if llm_result:
                        llm_result["method"] = "llm"
                        llm_result["importance_score"] = importance
                        llm_result["file_type"] = guess_file_type(doc.get("filename", "").lower(), doc.get("text", "").lower())
                        results[i] = llm_result
                        llm_count += 1
                        classifier.update_training_data(doc.get("text", ""), llm_result)
                    else:
                        rule_result = classifier._classify_with_rules(doc.get("text", ""), doc.get("filename", ""))
                        rule_result["method"] = "rule"
                        rule_result["importance_score"] = importance
                        results[i] = rule_result
                        rule_count += 1
                    completed_indices.add(i)
                batch_num += 1
                save_classification_progress(results, list(completed_indices))
        elif high_importance_docs:
            for i, doc, importance in high_importance_docs:
                rule_result = classifier._classify_with_rules(doc.get("text", ""), doc.get("filename", ""))
                rule_result["method"] = "rule"
                rule_result["importance_score"] = importance
                results[i] = rule_result
                rule_count += 1
    for i, doc in enumerate(documents):
        if results[i] is None:
            results[i] = {"entities": [], "file_type": "document", "evidence_of": [], "method": "rule", "confidence": 0.5}
            rule_count += 1
            
        results[i].update({
            "pdf_path": doc.get("pdf_path", ""),
            "filename": doc.get("filename", ""),
            "page_num": doc.get("page_num", 0),
            "total_pages": doc.get("total_pages", 1),
            "is_image_page": doc.get("is_image_page", False),
            "preview": doc.get("text_preview", "")[:300],
            "text": doc.get("text", ""),
            "has_hidden_content": doc.get("has_hidden_content", False),
            "recovered_text": doc.get("recovered_text", ""),
            "ai_classified": results[i].get("method") == "llm",
            "hybrid_classified": True
        })
    classifier.save_cache()
    save_classification_cache()
    clear_classification_progress()
    
    print(f"\nClassification Summary:")
    print(f"  LLM (important docs): {llm_count}")
    print(f"  KNN (similar docs): {knn_count}")
    print(f"  Rules (fallback): {rule_count}")
    print(f"  Cache hits: {cache_hits}")
    
    return results

def classify_document_wrapper(classifier, doc):
    return classifier.classify_document(
        doc["text"], 
        doc.get("filename", ""),
        is_photo_page=doc.get("is_image_page", False),
        has_hidden_content=doc.get("has_hidden_content", False),
        recovered_text=doc.get("recovered_text", "")
    )

def main():
    print("=" * 60)
    print("DOJ Epstein Files - Hybrid Classification")
    print("=" * 60)
    ollama_model = select_model(interactive=True)
    print(f"\n[OK] Using model: {ollama_model}")
    if not ensure_ollama_ready(ollama_model):
        print("\n[!] Ollama not available - will use KNN and rule-based classification")
        ollama_model = None
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from topic_classifier import save_extracted_documents, load_extracted_documents
    documents = load_extracted_documents()
    
    if documents:
        print(f"[Resume] Found {len(documents)} previously extracted documents")
        print("         Skipping extraction, proceeding to classification...")
    else:
        print(f"\nLoading PDFs from {DOWNLOADS_DIR}...")
        try:
            documents = load_pdfs_by_page(DOWNLOADS_DIR)
        except Exception as e:
            print(f"\n[ERROR] Failed to load PDFs: {e}")
            import traceback
            traceback.print_exc()
            return
        
        if not documents:
            print("No PDF files found. Run download step first.")
            return
        print("\n[Save] Saving extracted text before classification...")
        save_extracted_documents(documents)
    print("\nRunning hybrid classification...")
    import multiprocessing
    optimal_workers = get_optimal_workers()
    
    labels = create_hybrid_labels(documents, ollama_model, optimal_workers)
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    print(f"\nLabels saved to {LABELS_FILE}")
    print("\nClassification Summary:")
    all_entities = []
    all_evidence = []
    all_methods = []
    
    for label in labels:
        all_entities.extend(label.get("entities", []))
        all_evidence.extend(label.get("evidence_of", []))
        all_methods.append(label.get("method", "rule"))
        
    print(f"\n  Top Entities:")
    for entity, count in Counter(all_entities).most_common(15):
        print(f"    {entity}: {count} documents")
        
    print(f"\n  Evidence Of:")
    for evidence, count in Counter(all_evidence).most_common(15):
        print(f"    {evidence}: {count} documents")
        
    print(f"\n  Classification Methods:")
    for method, count in Counter(all_methods).most_common():
        print(f"    {method}: {count} documents")
        
    print("\n" + "=" * 60)
    print("HYBRID CLASSIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

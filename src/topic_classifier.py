import os
import sys
import json
import pickle
import subprocess
import requests
import warnings
import hashlib
import threading
import multiprocessing
from functools import partial
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

import numpy as np
from tqdm import tqdm

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    import hdbscan
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

DOWNLOADS_DIR = os.path.join(BASE_DIR, "data", "downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "classification")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "classification", "embeddings.pkl")
LABELS_FILE = os.path.join(BASE_DIR, "data", "classification", "labels.json")
KNOWN_LABELS_FILE = os.path.join(BASE_DIR, "data", "classification", "known_labels.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

NUM_TOPICS = 20

PDF_EXTRACTION_CACHE = os.path.join(BASE_DIR, "data", "classification", "pdf_extraction_cache.pkl")
EXTRACTED_DOCS_FILE = os.path.join(BASE_DIR, "data", "classification", "extracted_documents.pkl")
CLASSIFICATION_PROGRESS_FILE = os.path.join(BASE_DIR, "data", "classification", "classification_progress.pkl")
PDF_BATCH_SIZE = 8
PDF_WORKERS = max(4, multiprocessing.cpu_count())

_pdf_page_cache = {}
_pdf_cache_lock = threading.Lock()


def get_pdf_cache_key(pdf_path, page_num):
    stat = os.stat(pdf_path)
    return f"{pdf_path}:{page_num}:{stat.st_size}:{stat.st_mtime}"


def load_pdf_extraction_cache():
    global _pdf_page_cache
    try:
        if os.path.exists(PDF_EXTRACTION_CACHE):
            with open(PDF_EXTRACTION_CACHE, 'rb') as f:
                _pdf_page_cache = pickle.load(f)
            print(f"[Cache] Loaded {len(_pdf_page_cache)} cached PDF page extractions")
            return True
    except Exception as e:
        print(f"[Cache] Could not load PDF cache: {e}")
    return False


def save_pdf_extraction_cache():
    global _pdf_page_cache
    try:
        os.makedirs(os.path.dirname(PDF_EXTRACTION_CACHE), exist_ok=True)
        with open(PDF_EXTRACTION_CACHE, 'wb') as f:
            pickle.dump(_pdf_page_cache, f)
        print(f"[Cache] Saved {len(_pdf_page_cache)} PDF page extractions to cache")
    except Exception as e:
        print(f"[Cache] Could not save PDF cache: {e}")


def save_extracted_documents(documents):
    try:
        os.makedirs(os.path.dirname(EXTRACTED_DOCS_FILE), exist_ok=True)
        tmp_file = EXTRACTED_DOCS_FILE + ".tmp"
        with open(tmp_file, 'wb') as f:
            pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_file, EXTRACTED_DOCS_FILE)
        print(f"[Cache] Saved {len(documents)} extracted documents for resumability")
        return True
    except Exception as e:
        print(f"[Cache] Could not save extracted documents: {e}")
        return False


def load_extracted_documents():
    try:
        if os.path.exists(EXTRACTED_DOCS_FILE):
            with open(EXTRACTED_DOCS_FILE, 'rb') as f:
                documents = pickle.load(f)
            print(f"[Cache] Loaded {len(documents)} previously extracted documents")
            return documents
    except Exception as e:
        print(f"[Cache] Could not load extracted documents: {e}")
    return None


def save_classification_progress(results, completed_indices):
    try:
        os.makedirs(os.path.dirname(CLASSIFICATION_PROGRESS_FILE), exist_ok=True)
        sparse_results = {i: results[i] for i in completed_indices if i < len(results) and results[i] is not None}
        tmp_file = CLASSIFICATION_PROGRESS_FILE + ".tmp"
        with open(tmp_file, 'wb') as f:
            pickle.dump({"sparse_results": sparse_results, "completed": list(completed_indices)}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_file, CLASSIFICATION_PROGRESS_FILE)
        return True
    except Exception as e:
        print(f"[Cache] Could not save classification progress: {e}")
        return False


def load_classification_progress():
    try:
        if os.path.exists(CLASSIFICATION_PROGRESS_FILE):
            with open(CLASSIFICATION_PROGRESS_FILE, 'rb') as f:
                data = pickle.load(f)
            completed = set(data.get("completed", []))
            if "sparse_results" in data:
                sparse = data["sparse_results"]
                if sparse:
                    max_idx = max(sparse.keys())
                    results = [sparse.get(i) for i in range(max_idx + 1)]
                else:
                    results = []
            else:
                results = data.get("results", [])
            print(f"[Cache] Loaded classification progress: {len(completed)} completed")
            return results, completed
    except Exception as e:
        print(f"[Cache] Could not load classification progress: {e}")
    return [], set()


def clear_classification_progress():
    try:
        if os.path.exists(CLASSIFICATION_PROGRESS_FILE):
            os.remove(CLASSIFICATION_PROGRESS_FILE)
    except:
        pass

OLLAMA_URL = "http://localhost:11434"

MODEL_OPTIONS = {
    "small": {"name": "dolphin-phi", "vram": 3, "desc": "Fast, 3GB VRAM"},
    "medium": {"name": "dolphin-mistral", "vram": 5, "desc": "Balanced, 5GB VRAM"},
    "large": {"name": "dolphin-llama3:8b", "vram": 6, "desc": "Greater quality, 6GB VRAM"},
    "xlarge": {"name": "dolphin-llama3:70b", "vram": 40, "desc": "Maximum quality, 40GB+ VRAM"},
}

def detect_gpu_vram():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            vram_mb = int(result.stdout.strip().split('\n')[0])
            return vram_mb / 1024
    except:
        pass
    return 0

def get_optimal_workers():
    import multiprocessing
    cpu_cores = multiprocessing.cpu_count()
    vram = detect_gpu_vram()
    
    base_workers = max(4, cpu_cores)
    
    if vram >= 16:
        multiplier = 2.0
    elif vram >= 8:
        multiplier = 1.5
    elif vram >= 4:
        multiplier = 1.25
    else:
        multiplier = 1.0
    
    optimal = int(base_workers * multiplier)
    return min(optimal, 32)

def get_recommended_model():
    vram = detect_gpu_vram()
    if vram >= 40:
        return "xlarge", vram
    elif vram >= 6:
        return "large", vram
    elif vram >= 5:
        return "medium", vram
    elif vram >= 3:
        return "small", vram
    else:
        return "medium", vram

def select_model(interactive=True):
    rec, vram = get_recommended_model()
    
    if not interactive:
        return MODEL_OPTIONS[rec]["name"]
    
    print(f"\n[GPU] Detected VRAM: {vram:.1f}GB" if vram > 0 else "\n[!] No NVIDIA GPU detected (will use CPU)")
    print("\nAvailable models:")
    for key, opt in MODEL_OPTIONS.items():
        marker = " <- recommended" if key == rec else ""
        print(f"  [{key[0]}] {opt['name']:20} - {opt['desc']}{marker}")
    
    print(f"\nPress Enter for recommended ({MODEL_OPTIONS[rec]['name']}) or type s/m/l/x: ", end="")
    choice = input().strip().lower()
    
    if choice == "" or choice == rec[0]:
        return MODEL_OPTIONS[rec]["name"]
    elif choice == "s":
        return MODEL_OPTIONS["small"]["name"]
    elif choice == "m":
        return MODEL_OPTIONS["medium"]["name"]
    elif choice == "l":
        return MODEL_OPTIONS["large"]["name"]
    elif choice == "x":
        return MODEL_OPTIONS["xlarge"]["name"]
    else:
        return MODEL_OPTIONS[rec]["name"]

OLLAMA_MODEL = None


def should_skip_pdf_path(pdf_path: Path):
    name = pdf_path.name
    if name.startswith("~$"):
        return True
    try:
        if pdf_path.stat().st_size == 0:
            return True
    except Exception:
        return True
    return False


def extract_hidden_text(page):
    standard_text = ""
    extracted_texts = []
    
    try:
        standard_text = page.get_text("text").strip()
        if standard_text:
            extracted_texts.append(("standard", standard_text))
    except:
        pass
    
    try:
        raw_text = page.get_text("rawdict")
        raw_parts = []
        if raw_text and "blocks" in raw_text:
            for block in raw_text["blocks"]:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            span_text = span.get("text", "").strip()
                            if span_text:
                                raw_parts.append(span_text)
        if raw_parts:
            extracted_texts.append(("rawdict", " ".join(raw_parts)))
    except:
        pass
    
    try:
        words = page.get_text("words")
        word_texts = [w[4] for w in words if len(w) > 4 and w[4].strip()]
        if word_texts:
            words_combined = " ".join(word_texts)
            extracted_texts.append(("words", words_combined))
    except:
        pass
    
    try:
        xml_text = page.get_text("xml")
        if xml_text:
            import re
            char_pattern = re.compile(r'c="([^"]+)"')
            chars = char_pattern.findall(xml_text)
            if chars:
                xml_extracted = "".join(chars)
                extracted_texts.append(("xml", xml_extracted))
    except:
        pass
    
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        dict_parts = []
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            dict_parts.append(text)
        if dict_parts:
            extracted_texts.append(("dict", " ".join(dict_parts)))
    except:
        pass
    
    best_text = standard_text
    best_method = "standard"
    for method, text in extracted_texts:
        if len(text) > len(best_text):
            best_text = text
            best_method = method
    
    has_hidden_content = False
    recovered_text = ""
    
    if best_method != "standard" and len(best_text) > len(standard_text) * 1.1 + 50:
        has_hidden_content = True
        standard_words = set(standard_text.lower().split())
        best_words = best_text.lower().split()
        recovered_words = [w for w in best_words if w not in standard_words]
        recovered_text = " ".join(recovered_words[:200])
    
    return best_text, standard_text, recovered_text, has_hidden_content


def extract_page_info(pdf_path, page_num, page):
    text = ""
    is_photo = False
    has_hidden_content = False
    recovered_text = ""
    
    try:
        image_list = page.get_images(full=True)
        has_images = len(image_list) > 0
        
        text, standard_text, recovered_text, has_hidden_content = extract_hidden_text(page)
        
        if not text:
            text = page.get_text("text").strip()
        
        if len(text) < 50 and HAS_OCR and has_images:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, config='--psm 6').strip()
                if len(ocr_text) > len(text):
                    text = ocr_text
            except:
                pass
        
        if has_images and len(text) < 150:
            is_photo = True
                
    except Exception as e:
        pass
    
    return text, is_photo, has_hidden_content, recovered_text


def extract_text_from_pdf(pdf_path):
    text_parts = []
    
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                text, _, _, _ = extract_page_info(pdf_path, page_num, page)
                if text.strip():
                    text_parts.append(text.strip())
            doc.close()
        except Exception as e:
            pass
    
    if not text_parts and HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    for table in page.extract_tables():
                        if table:
                            for row in table:
                                if row:
                                    text += "\n" + " | ".join(str(c) if c else "" for c in row)
                    if text.strip():
                        text_parts.append(text.strip())
        except:
            pass
    
    return "\n\n".join(text_parts)


def extract_single_pdf(pdf_path_str):
    pdf_path = Path(pdf_path_str)
    pages_data = []
    
    try:
        if not HAS_PYMUPDF:
            text = extract_text_from_pdf(pdf_path_str)
            if len(text) > 20:
                return [{
                    "pdf_path": pdf_path_str,
                    "filename": pdf_path.name,
                    "page_num": 0,
                    "is_image_page": False,
                    "text": text,
                    "text_preview": text[:500]
                }]
            return []
        
        doc = fitz.open(pdf_path_str)
        num_pages = len(doc)
        
        for page_num in range(num_pages):
            page = doc[page_num]
            text, is_image_page, has_hidden, recovered = extract_page_info(pdf_path_str, page_num, page)
            
            page_id = f"{pdf_path.name}#page{page_num + 1}"
            
            if is_image_page:
                page_data = {
                    "pdf_path": pdf_path_str,
                    "filename": page_id,
                    "page_num": page_num + 1,
                    "total_pages": num_pages,
                    "is_image_page": True,
                    "text": text if text else "[Image/Photo]",
                    "text_preview": text[:500] if text else "[Image/Photo]",
                    "has_hidden_content": has_hidden,
                    "recovered_text": recovered
                }
                pages_data.append(page_data)
            elif len(text) > 20:
                page_data = {
                    "pdf_path": pdf_path_str,
                    "filename": page_id,
                    "page_num": page_num + 1,
                    "total_pages": num_pages,
                    "is_image_page": False,
                    "text": text,
                    "text_preview": text[:500],
                    "has_hidden_content": has_hidden,
                    "recovered_text": recovered
                }
                pages_data.append(page_data)
        
        doc.close()
    except Exception as e:
        pass
    
    return pages_data


def load_pdfs_by_page(downloads_dir):
    downloads_path = Path(downloads_dir)
    documents = []
    
    pdf_files = [p for p in downloads_path.rglob("*.pdf") if not should_skip_pdf_path(p)]
    print(f"Found {len(pdf_files)} PDF files")
    
    photo_count = 0
    text_page_count = 0
    
    print(f"Processing PDFs with {PDF_WORKERS} workers...")
    
    pdf_paths = [str(p) for p in pdf_files]
    
    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as executor:
        futures = {executor.submit(extract_single_pdf, path): path for path in pdf_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting PDFs"):
            try:
                pages = future.result()
                for page in pages:
                    if page.get("is_image_page"):
                        photo_count += 1
                    else:
                        text_page_count += 1
                    documents.append(page)
            except Exception as e:
                pass
    
    print(f"Extracted {len(documents)} pages: {photo_count} image pages, {text_page_count} text pages")
    return documents


def load_pdfs(downloads_dir):
    downloads_path = Path(downloads_dir)
    documents = []
    
    pdf_files = [p for p in downloads_path.rglob("*.pdf") if not should_skip_pdf_path(p)]
    print(f"Found {len(pdf_files)} PDF files - the computer file type, not...y'know")
    
    for pdf_file in tqdm(pdf_files, desc="Extracting text from PDFs"):
        try:
            text = extract_text_from_pdf(str(pdf_file))
            
            if len(text) > 20:
                documents.append({
                    "pdf_path": str(pdf_file),
                    "filename": pdf_file.name,
                    "text": text,
                    "text_preview": text[:500]
                })
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    return documents


def is_ollama_installed():
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def start_ollama():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except:
        pass
    
    print("   Starting Ollama server...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        for _ in range(10):
            time.sleep(1)
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
                return True
            except:
                pass
    except Exception as e:
        print(f"   [!] Failed to start Ollama: {e}")
    return False

def pull_model(model):
    print(f"   Downloading model '{model}'...")
    print("   This may take 5-15 minutes depending on your connection.")
    try:
        result = subprocess.run(["ollama", "pull", model], timeout=1800)
        return result.returncode == 0
    except Exception as e:
        print(f"   [!] Failed to pull model: {e}")
        return False

def check_ollama(model=None):
    use_model = model or OLLAMA_MODEL or "dolphin-mistral"
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if use_model in model_names:
                return True
            base_name = use_model.split(":")[0]
            for name in model_names:
                if name.split(":")[0] == base_name:
                    return True
    except:
        pass
    return False

def ensure_ollama_ready(model):
    if not is_ollama_installed():
        print("\n[!] Ollama is not installed.")
        print("    Install from: https://ollama.com/download")
        print("    Or run: winget install Ollama.Ollama")
        return False
    
    if not start_ollama():
        print("\n[!] Could not start Ollama server.")
        return False
    
    if not check_ollama(model):
        print(f"\n[....] Model '{model}' not found. Pulling...")
        if not pull_model(model):
            print(f"\n[!] Failed to download model '{model}'")
            return False
        print(f"   [OK] Model '{model}' ready")
    
    return True

CLASSIFY_PROMPT = """Analyze this document page from the Epstein case files.

{existing_labels}
Filename: {filename}
Content: {text}

Extract:
1. entities: Names of people mentioned (full names when possible)
2. file_type: Be specific and descriptive. Examples: "photograph", "flight log", "email correspondence", "court document", "deposition transcript", "financial statement", "address book entry", "handwritten note", "fax", "phone record", "police report", "legal filing", "witness statement", "news clipping", "calendar entry", "receipt", "bank record", "property record", "travel itinerary", "message slip", "memo"
3. evidence_of: What does this document relate to or provide evidence of? Be specific.

Rules:
- If this is marked as [Image/Photo] or has minimal text with image indicators, file_type MUST be "photograph"
- NEVER use "unknown" or "other" - always pick the most appropriate specific type
- If unsure, make your best educated guess based on context clues
- Look at filename for hints (e.g. "flight" suggests flight log, "depo" suggests deposition)

Return ONLY valid JSON:
{{"entities": [], "file_type": "", "evidence_of": []}}"""

known_labels = {"entities": set(), "file_types": set(), "evidence_of": set()}

def load_known_labels():
    global known_labels
    if os.path.exists(KNOWN_LABELS_FILE):
        try:
            with open(KNOWN_LABELS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                known_labels["entities"] = set(data.get("entities", []))
                known_labels["file_types"] = set(data.get("file_types", []))
                known_labels["evidence_of"] = set(data.get("evidence_of", []))
        except:
            pass

def save_known_labels():
    global known_labels
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(KNOWN_LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "entities": sorted(known_labels["entities"]),
            "file_types": sorted(known_labels["file_types"]),
            "evidence_of": sorted(known_labels["evidence_of"])
        }, f, indent=2)

def update_known_labels(result):
    global known_labels
    if result.get("entities"):
        known_labels["entities"].update(result["entities"])
    if result.get("file_type"):
        known_labels["file_types"].add(result["file_type"])
    if result.get("evidence_of"):
        known_labels["evidence_of"].update(result["evidence_of"])

def build_existing_labels_prompt():
    parts = []
    if known_labels["entities"]:
        sample = sorted(known_labels["entities"])[:50]
        parts.append(f"Known entities (reuse if matching): {', '.join(sample)}")
    if known_labels["file_types"]:
        parts.append(f"Known file types (reuse if matching): {', '.join(sorted(known_labels['file_types']))}")
    if known_labels["evidence_of"]:
        sample = sorted(known_labels["evidence_of"])[:30]
        parts.append(f"Known evidence types (reuse if matching): {', '.join(sample)}")
    return '\n'.join(parts) + '\n' if parts else ''

PHOTO_KEYWORDS = {'photo', 'photograph', 'image', 'picture', 'img', 'pic_', '_pic', 'jpg', 'jpeg', 'png', 'screenshot'}

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.flac', '.aiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
MEDIA_KEYWORDS = {'audio', 'video', 'recording', 'voicemail', 'vm_', '_vm', 'call_', '_call'}

GARBAGE_TERMS = {
    'page', 'item', 'page item', 'document', 'file', 'text', 'content', 'data',
    'unknown', 'n/a', 'none', 'null', 'undefined', 'full name', 'name', 'type',
    'the', 'a', 'an', 'this', 'that', 'it', 'is', 'are', 'was', 'were',
    'redacted', '[redacted]', 'exhibit', 'attachment', 'appendix',
}

def filter_garbage(items):
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if len(item) < 2:
            continue
        if item.lower() in GARBAGE_TERMS:
            continue
        if item.lower().startswith('full ') or item.lower().endswith(' name'):
            continue
        cleaned.append(item)
    return cleaned

def get_ollama_threads():
    import multiprocessing
    return max(4, multiprocessing.cpu_count() // 2)

def ai_classify(text, filename="", model=None, is_photo_page=False):
    fname_lower = filename.lower()
    text_lower = text.lower()[:500] if text else ""
    
    if any(fname_lower.endswith(ext) for ext in AUDIO_EXTENSIONS) or \
       any(kw in fname_lower for kw in MEDIA_KEYWORDS if 'audio' in kw or 'voice' in kw or 'call' in kw):
        return {
            "entities": [],
            "file_type": "audio",
            "evidence_of": ["audio recording"]
        }
    
    if any(fname_lower.endswith(ext) for ext in VIDEO_EXTENSIONS) or \
       any(kw in fname_lower for kw in MEDIA_KEYWORDS if 'video' in kw):
        return {
            "entities": [],
            "file_type": "video",
            "evidence_of": ["video recording"]
        }
    
    is_photo = (
        is_photo_page or
        text.strip() == "[Image/Photo]" or
        '[photograph]' in text_lower or
        (any(kw in fname_lower for kw in PHOTO_KEYWORDS) and len(text.strip()) < 150)
    )
    
    if is_photo and len(text.strip()) < 150:
        return {
            "entities": [],
            "file_type": "photograph",
            "evidence_of": ["photographic evidence"]
        }
    
    text_sample = text[:1200].replace('\n', ' ').strip()
    existing_labels = build_existing_labels_prompt()

    prompt = CLASSIFY_PROMPT.format(filename=filename, text=text_sample, existing_labels=existing_labels)
    use_model = model or OLLAMA_MODEL or "dolphin-mistral"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": use_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200,
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
                    file_type = parsed.get("file_type", "") if isinstance(parsed.get("file_type"), str) else ""
                    
                    if not file_type or file_type.lower() in ['unknown', 'other', 'n/a', 'none']:
                        file_type = guess_file_type(fname_lower, text_lower)
                    
                    if is_image_page and file_type.lower() not in ['photograph', 'photo']:
                        file_type = "photograph"
                    
                    classification = {
                        "entities": filter_garbage(parsed.get("entities", []) if isinstance(parsed.get("entities"), list) else []),
                        "file_type": file_type,
                        "evidence_of": filter_garbage(parsed.get("evidence_of", []) if isinstance(parsed.get("evidence_of"), list) else [])
                    }
                    update_known_labels(classification)
                    return classification
    except Exception as e:
        pass

    return {
        "entities": [],
        "file_type": "photograph" if is_photo_page else guess_file_type(fname_lower, text_lower),
        "evidence_of": []
    }


def guess_file_type(filename_lower, text_lower):
    if 'deposition' in text_lower:
        return 'deposition transcript'
    if ('q.' in text_lower or 'q:' in text_lower) and ('a.' in text_lower or 'a:' in text_lower):
        return 'deposition transcript'
    if 'witness' in text_lower and ('sworn' in text_lower or 'testimony' in text_lower):
        return 'deposition transcript'
    
    if 'from:' in text_lower and 'to:' in text_lower and ('subject:' in text_lower or 're:' in text_lower):
        return 'email correspondence'
    if 'sent:' in text_lower and 'to:' in text_lower:
        return 'email correspondence'
    
    if 'flight' in text_lower and ('passenger' in text_lower or 'manifest' in text_lower or 'tail' in text_lower):
        return 'flight log'
    if 'aircraft' in text_lower and ('depart' in text_lower or 'arrival' in text_lower):
        return 'flight log'
    
    if 'facsimile' in text_lower or 'fax transmittal' in text_lower:
        return 'fax'
    if 'fax' in text_lower and ('to:' in text_lower or 'from:' in text_lower or 'pages:' in text_lower):
        return 'fax'
    
    if 'plaintiff' in text_lower and 'defendant' in text_lower:
        return 'court document'
    if 'case no' in text_lower or 'docket' in text_lower:
        return 'court document'
    if 'subpoena' in text_lower:
        return 'subpoena'
    if 'affidavit' in text_lower and 'sworn' in text_lower:
        return 'affidavit'
    
    if '$' in text_lower and ('total' in text_lower or 'amount' in text_lower or 'balance' in text_lower):
        return 'financial record'
    if 'invoice' in text_lower or 'receipt' in text_lower:
        return 'receipt'
    if 'bank' in text_lower and ('account' in text_lower or 'statement' in text_lower):
        return 'bank record'
    
    if 'dear' in text_lower and ('sincerely' in text_lower or 'regards' in text_lower or 'truly' in text_lower):
        return 'letter'
    
    if 'memo' in text_lower or 'memorandum' in text_lower:
        return 'memo'
    
    if 'message' in text_lower and ('call' in text_lower or 'phone' in text_lower):
        return 'phone message'
    
    if text_lower.count('@') > 2 or (text_lower.count('phone') > 1 and text_lower.count('address') > 0):
        return 'contact list'
    
    return 'document'

def generate_embeddings(documents, model):
    print("Generating embeddings...")

    texts = [doc["text"][:1000] for doc in documents]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    return embeddings

def cluster_documents(embeddings, method="hdbscan"):
    print(f"Clustering documents using {method}...")

    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(embeddings)
    else:
        clusterer = KMeans(n_clusters=NUM_TOPICS, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)

    return labels

def extract_cluster_keywords(documents, cluster_labels):
    clusters = {}

    for doc, label in zip(documents, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(doc["text"][:2000])

    cluster_keywords = {}

    for cluster_id, texts in clusters.items():
        if cluster_id == -1:
            cluster_keywords[cluster_id] = ["uncategorized"]
            continue

        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf = vectorizer.fit_transform(texts)

            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]

            cluster_keywords[cluster_id] = top_words
        except:
            cluster_keywords[cluster_id] = ["misc"]

    return cluster_keywords

def load_existing_labels():
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
                cache = {l.get("filename", ""): l for l in labels if l.get("ai_classified")}
                for l in labels:
                    if l.get("entities"):
                        known_labels["entities"].update(l["entities"])
                    if l.get("file_type") and l["file_type"] not in ['unknown', 'document']:
                        known_labels["file_types"].add(l["file_type"])
                    if l.get("evidence_of"):
                        known_labels["evidence_of"].update(l["evidence_of"])
                return cache
        except:
            pass
    return {}

def classify_single_doc(doc, cache):
    pdf_path = doc.get("pdf_path", "")
    filename = doc.get("filename", "")
    is_photo_page = doc.get("is_image_page", False)
    
    if filename in cache:
        cached = cache[filename].copy()
        cached["cached"] = True
        if "text" not in cached:
            cached["text"] = doc.get("text", "")
        return cached

    ai_result = ai_classify(doc["text"], filename, OLLAMA_MODEL, is_photo_page=is_photo_page)

    return {
        "pdf_path": pdf_path,
        "filename": filename,
        "page_num": doc.get("page_num", 0),
        "total_pages": doc.get("total_pages", 1),
        "is_image_page": is_photo_page,
        "entities": ai_result.get("entities", []),
        "file_type": ai_result.get("file_type", "document"),
        "evidence_of": ai_result.get("evidence_of", []),
        "ai_classified": True,
        "preview": doc["text_preview"][:300],
        "text": doc["text"],
        "cached": False,
        "has_hidden_content": doc.get("has_hidden_content", False),
        "recovered_text": doc.get("recovered_text", "")
    }

def create_topic_labels(documents, use_ai=True, max_workers=12):
    ai_available = use_ai and check_ollama()

    if use_ai and not ai_available:
        print(f"\n[!] Ollama not available. Run 'ollama pull {OLLAMA_MODEL or 'dolphin-mistral'}'")

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
    
    docs_to_process = [(i, doc) for i, doc in enumerate(documents) if i not in completed_indices]
    
    if completed_indices:
        print(f"[Resume] Found {len(completed_indices)} previously classified documents")
        print(f"         {len(docs_to_process)} remaining to classify")

    if ai_available:
        load_known_labels()
        cache = load_existing_labels()
        cached_count = sum(1 for i, d in docs_to_process if d.get("filename", "") in cache)
        
        image_pages = sum(1 for d in documents if d.get("is_image_page", False))

        print(f"\n[>] AI classification with {OLLAMA_MODEL}")
        print(f"   {len(documents)} pages ({image_pages} photos, {len(documents) - image_pages} text)")
        print(f"   {cached_count} in label cache, {len(docs_to_process)} to classify")
        print(f"   {max_workers} parallel workers")

        ai_success = 0
        photo_count = 0
        from_cache = 0
        save_interval = max(100, len(docs_to_process) // 10)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(classify_single_doc, doc, cache): (i, doc) for i, doc in docs_to_process}

            for future in tqdm(as_completed(futures), total=len(docs_to_process), desc="Classifying"):
                idx, doc = futures[future]
                result = future.result()
                results[idx] = result
                completed_indices.add(idx)

                if result.get("cached"):
                    from_cache += 1
                else:
                    ai_success += 1
                
                if result.get("file_type") == "photograph":
                    photo_count += 1
                
                if len(completed_indices) % save_interval == 0:
                    save_classification_progress(results, list(completed_indices))

        print(f"\n[OK] Done! Classified: {ai_success}, Cached: {from_cache}, Photos: {photo_count}")
        save_known_labels()
    else:
        for i, doc in tqdm(docs_to_process, desc="Processing"):
            is_image = doc.get("is_image_page", False)
            results[i] = {
                "pdf_path": doc.get("pdf_path", ""),
                "filename": doc["filename"],
                "page_num": doc.get("page_num", 0),
                "is_image_page": is_image,
                "entities": [],
                "file_type": "photograph" if is_image else "document",
                "evidence_of": [],
                "ai_classified": False,
                "preview": doc["text_preview"][:300],
                "text": doc["text"],
                "has_hidden_content": doc.get("has_hidden_content", False),
                "recovered_text": doc.get("recovered_text", "")
            }
            completed_indices.add(i)

    clear_classification_progress()
    
    return results

def main():
    global OLLAMA_MODEL
    
    print("=" * 60)
    print("DOJ Epstein Files - Topic Classification")
    print("=" * 60)

    OLLAMA_MODEL = select_model(interactive=True)
    print(f"\n[OK] Using model: {OLLAMA_MODEL}")
    
    if not ensure_ollama_ready(OLLAMA_MODEL):
        print("\n[!] Skipping AI classification - Ollama not available")
        print("    Documents will be indexed without AI labels.")
        print("    You can run classification later with: python src/topic_classifier.py")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    documents = load_extracted_documents()
    
    if documents:
        print(f"[Resume] Found {len(documents)} previously extracted documents")
        print("         Skipping extraction, proceeding to classification...")
    else:
        print(f"\nLoading PDFs from {DOWNLOADS_DIR} (page-by-page)...")
        documents = load_pdfs_by_page(DOWNLOADS_DIR)
        
        if not documents:
            print("No PDF files found. Run download step first.")
            return
        
        print("\n[Save] Saving extracted text before classification...")
        save_extracted_documents(documents)

    print("\nRunning topic classification (per page) - this will take a while...")
    import multiprocessing
    cpu_cores = multiprocessing.cpu_count()
    vram = detect_gpu_vram()
    optimal_workers = get_optimal_workers()
    print(f"[HW] CPU: {cpu_cores} cores, VRAM: {vram:.1f}GB -> {optimal_workers} workers, {get_ollama_threads()} Ollama threads")
    labels = create_topic_labels(documents, use_ai=True, max_workers=optimal_workers)

    if HAS_SENTENCE_TRANSFORMERS and HAS_CLUSTERING:
        print(f"\nLoading embedding model ({EMBEDDING_MODEL})...")
        model = SentenceTransformer(EMBEDDING_MODEL)

        embeddings = generate_embeddings(documents, model)

        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({
                "embeddings": embeddings,
                "filenames": [d["filename"] for d in documents]
            }, f)
        print(f"Embeddings saved to {EMBEDDINGS_FILE}")

        cluster_labels = cluster_documents(embeddings, method="kmeans")

        cluster_keywords = extract_cluster_keywords(documents, cluster_labels)

        for i, label_info in enumerate(labels):
            cluster_id = int(cluster_labels[i])
            label_info["cluster_id"] = cluster_id
            label_info["cluster_keywords"] = cluster_keywords.get(cluster_id, [])

        print("\nCluster Summary:")
        cluster_counts = Counter(cluster_labels)
        for cluster_id, count in sorted(cluster_counts.items()):
            keywords = cluster_keywords.get(cluster_id, ["unknown"])[:5]
            print(f"  Cluster {cluster_id}: {count} docs - {', '.join(keywords)}")

    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    print(f"\nLabels saved to {LABELS_FILE}")

    print("\nClassification Summary:")
    all_entities = []
    all_file_types = []
    all_evidence = []
    for label in labels:
        all_entities.extend(label.get("entities", []))
        if label.get("file_type"):
            all_file_types.append(label["file_type"])
        all_evidence.extend(label.get("evidence_of", []))

    print("\n  Top Entities:")
    for entity, count in Counter(all_entities).most_common(15):
        print(f"    {entity}: {count} documents")

    print("\n  File Types:")
    for ftype, count in Counter(all_file_types).most_common(15):
        print(f"    {ftype}: {count} documents")

    print("\n  Evidence Of:")
    for evidence, count in Counter(all_evidence).most_common(15):
        print(f"    {evidence}: {count} documents")

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

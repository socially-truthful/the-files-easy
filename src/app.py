import os
import json
import webbrowser
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from whoosh import index
    from whoosh.qparser import MultifieldParser
    from whoosh.query import And, Term
    HAS_WHOOSH = True
except ImportError:
    HAS_WHOOSH = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_TEXT_DIR = os.path.join(BASE_DIR, "data", "extracted_text")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "data", "downloads")
LABELS_FILE = os.path.join(BASE_DIR, "data", "classification", "labels.json")
INDEX_DIR = os.path.join(BASE_DIR, "data", "search_index")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
CORS(app)

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def get_all_filters():
    labels = load_labels()
    entities = set()
    file_types = set()
    evidence_of = set()
    
    for label in labels:
        entities.update(label.get("entities", []))
        if label.get("file_type"):
            file_types.add(label["file_type"])
        evidence_of.update(label.get("evidence_of", []))
    
    return {
        "entities": sorted(entities),
        "file_types": sorted(file_types),
        "evidence_of": sorted(evidence_of)
    }

def search_documents(query, topic=None, page=1, per_page=20):
    labels = load_labels()
    query_lower = query.lower().strip() if query else ""
    
    pdf_results = {}
    
    for label in labels:
        pdf_path = label.get("pdf_path", "")
        if not pdf_path:
            continue
            
        if pdf_path in pdf_results:
            if label.get("has_hidden_content"):
                pdf_results[pdf_path]["has_hidden_content"] = True
            continue
        
        match = False
        if not query_lower or query_lower == "*":
            match = True
        else:
            searchable = " ".join([
                label.get("filename", ""),
                label.get("text", ""),
                " ".join(label.get("entities", [])),
                " ".join(label.get("evidence_of", [])),
                label.get("file_type", "")
            ]).lower()
            match = query_lower in searchable
        
        if match:
            pdf_results[pdf_path] = {
                "path": pdf_path,
                "filename": Path(pdf_path).name,
                "topics": [label.get("file_type", "document")],
                "has_hidden_content": label.get("has_hidden_content", False)
            }
    
    results = list(pdf_results.values())
    total = len(results)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    return {
        "results": results[start:end],
        "total": total,
        "page": page,
        "per_page": per_page
    }

def browse_by_filter(filter_type, filter_value):
    labels = load_labels()
    results = []

    for label in labels:
        match = False
        if filter_type == "entity" and filter_value in label.get("entities", []):
            match = True
        elif filter_type == "file_type" and label.get("file_type") == filter_value:
            match = True
        elif filter_type == "evidence" and filter_value in label.get("evidence_of", []):
            match = True
        
        if match:
            results.append({
                "path": label.get("pdf_path", label.get("path", "")),
                "pdf_path": label.get("pdf_path", ""),
                "filename": label.get("filename", ""),
                "entities": label.get("entities", []),
                "file_type": label.get("file_type", "unknown"),
                "evidence_of": label.get("evidence_of", []),
                "preview": label.get("preview", "")[:300]
            })

    return results

def get_document_content(pdf_path):
    labels = load_labels()
    
    pages = []
    for label in labels:
        label_pdf = label.get("pdf_path") or label.get("path", "")
        if label_pdf == pdf_path:
            pages.append(label)
    
    if not pages:
        return {"error": "Document not found"}
    
    pages.sort(key=lambda x: x.get("page_num", 0))
    
    all_text = []
    all_entities = set()
    all_evidence = set()
    file_types = set()
    has_hidden_content = False
    all_recovered_text = []
    
    for page in pages:
        text = page.get("text", "")
        if text and text != "[Image/Photo]":
            page_num = page.get("page_num", 0)
            if page_num > 0:
                all_text.append(f"--- Page {page_num} ---\n{text}")
            else:
                all_text.append(text)
        all_entities.update(page.get("entities", []))
        all_evidence.update(page.get("evidence_of", []))
        ft = page.get("file_type", "")
        if ft and ft not in ["unknown", "document"]:
            file_types.add(ft)
        if page.get("has_hidden_content"):
            has_hidden_content = True
            recovered = page.get("recovered_text", "")
            if recovered:
                all_recovered_text.append(f"Page {page.get('page_num', '?')}: {recovered}")
    
    file_type = list(file_types)[0] if file_types else pages[0].get("file_type", "document")
    
    return {
        "path": pdf_path,
        "pdf_path": pdf_path,
        "filename": Path(pdf_path).name,
        "content": "\n\n".join(all_text),
        "preview": pages[0].get("preview", "") if pages else "",
        "entities": sorted(all_entities),
        "file_type": file_type,
        "evidence_of": sorted(all_evidence),
        "page_count": len(pages),
        "has_hidden_content": has_hidden_content,
        "recovered_text": "\n".join(all_recovered_text),
        "pages": [{
            "page_num": p.get("page_num", 0),
            "is_image": p.get("is_image_page", False),
            "file_type": p.get("file_type", ""),
            "entities": p.get("entities", []),
            "has_hidden_content": p.get("has_hidden_content", False),
            "recovered_text": p.get("recovered_text", "")
        } for p in pages]
    }

def get_stats():
    labels = load_labels()

    entity_counts = {}
    file_type_counts = {}
    evidence_counts = {}
    unique_pdfs = set()
    photo_pages = 0
    hidden_content_pages = 0
    pdfs_with_hidden = set()
    
    for label in labels:
        for entity in label.get("entities", []):
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        ft = label.get("file_type", "document")
        if ft and ft.strip():
            file_type_counts[ft] = file_type_counts.get(ft, 0) + 1
        for ev in label.get("evidence_of", []):
            evidence_counts[ev] = evidence_counts.get(ev, 0) + 1
        
        pdf_path = label.get("pdf_path", "")
        if pdf_path:
            unique_pdfs.add(pdf_path)
        if label.get("is_image_page") or ft == "photograph":
            photo_pages += 1
        if label.get("has_hidden_content"):
            hidden_content_pages += 1
            if pdf_path:
                pdfs_with_hidden.add(pdf_path)

    return {
        "total_documents": len(labels),
        "total_pages": len(labels),
        "unique_pdfs": len(unique_pdfs),
        "photo_pages": photo_pages,
        "hidden_content_pages": hidden_content_pages,
        "pdfs_with_hidden_content": len(pdfs_with_hidden),
        "entities": dict(sorted(entity_counts.items(), key=lambda x: -x[1])[:50]),
        "file_types": file_type_counts,
        "topics": file_type_counts,
        "evidence_of": dict(sorted(evidence_counts.items(), key=lambda x: -x[1])[:30]),
        "index_available": HAS_WHOOSH and os.path.exists(INDEX_DIR)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    topic = request.args.get('topic', None)
    page = int(request.args.get('page', 1))

    results = search_documents(query, topic, page)
    return jsonify(results)

@app.route('/api/filters')
def api_filters():
    return jsonify(get_all_filters())

@app.route('/api/browse/<filter_type>/<path:filter_value>')
def api_browse(filter_type, filter_value):
    results = browse_by_filter(filter_type, filter_value)
    return jsonify({"results": results, "filter_type": filter_type, "filter_value": filter_value, "count": len(results)})

@app.route('/api/document')
def api_document():
    path = request.args.get('path', '')
    if not path:
        return jsonify({"error": "No path provided"}), 400

    result = get_document_content(path)
    return jsonify(result)

@app.route('/api/pdf')
def api_pdf():
    path = request.args.get('path', '')
    if not path or not os.path.exists(path):
        return jsonify({"error": "PDF not found"}), 404
    
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(os.path.abspath(DOWNLOADS_DIR)):
        return jsonify({"error": "Access denied"}), 403
    
    directory = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    return send_from_directory(directory, filename, mimetype='application/pdf')

@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

@app.route('/api/topics')
def api_topics():
    labels = load_labels()
    file_types = set()
    file_type_counts = {}
    
    for label in labels:
        ft = label.get("file_type", "unknown")
        if ft and ft.strip():
            file_types.add(ft)
            file_type_counts[ft] = file_type_counts.get(ft, 0) + 1
    
    return jsonify({
        "topics": sorted(file_types),
        "counts": file_type_counts
    })

@app.route('/api/browse/<topic>')
def api_browse_topic(topic):
    labels = load_labels()
    pdf_results = {}
    
    for label in labels:
        if label.get("file_type") == topic:
            pdf_path = label.get("pdf_path", label.get("path", ""))
            if pdf_path:
                if pdf_path not in pdf_results:
                    pdf_results[pdf_path] = {
                        "path": pdf_path,
                        "filename": Path(pdf_path).name,
                        "topics": [topic],
                        "has_hidden_content": label.get("has_hidden_content", False)
                    }
                elif label.get("has_hidden_content"):
                    pdf_results[pdf_path]["has_hidden_content"] = True
    
    results = list(pdf_results.values())
    return jsonify({"results": results, "topic": topic, "count": len(results)})

@app.route('/gallery')
def gallery_page():
    return render_template('gallery.html')

@app.route('/api/gallery')
def api_gallery():
    labels = load_labels()
    media_items = []
    
    for label in labels:
        file_type = label.get("file_type", "")
        is_image = label.get("is_image_page", False)
        
        if file_type == "photograph" or is_image or file_type == "video" or file_type == "audio":
            pdf_path = label.get("pdf_path", label.get("path", ""))
            if pdf_path:
                path_obj = Path(pdf_path)
                parent_parts = path_obj.parts
                source = "Unknown"
                for i, part in enumerate(parent_parts):
                    if part.lower() == "downloads" and i + 1 < len(parent_parts):
                        source = parent_parts[i + 1]
                        break
                if source == "Unknown" and len(parent_parts) > 1:
                    source = parent_parts[-2]
                
                media_items.append({
                    "pdf_path": pdf_path,
                    "filename": Path(pdf_path).name,
                    "page": label.get("page_num", 1),
                    "type": "audio" if file_type == "audio" else ("video" if file_type == "video" else "photograph"),
                    "source": source,
                    "entities": label.get("entities", []),
                    "preview": label.get("preview", "")[:200]
                })
    
    sources = {}
    for item in media_items:
        src = item.get("source", "Unknown")
        if src not in sources:
            sources[src] = []
        sources[src].append(item)
    
    return jsonify({
        "items": media_items,
        "sources": sources,
        "source_list": sorted(sources.keys()),
        "total": len(media_items)
    })

@app.route('/graph')
def graph_page():
    return render_template('graph.html')

@app.route('/api/graph')
def api_graph():
    labels = load_labels()

    entity_docs = {}
    for label in labels:
        for entity in label.get("entities", []):
            if entity not in entity_docs:
                entity_docs[entity] = set()
            entity_docs[entity].add(label.get("pdf_path", label.get("path", "")))

    entity_docs = {k: v for k, v in entity_docs.items() if len(v) >= 2}

    nodes = []
    for entity, docs in entity_docs.items():
        nodes.append({
            "id": entity,
            "label": entity,
            "type": "entity",
            "count": len(docs),
            "size": min(50, 10 + len(docs) * 2)
        })

    edges = []
    entities_list = list(entity_docs.keys())
    for i, e1 in enumerate(entities_list):
        for e2 in entities_list[i+1:]:
            shared = len(entity_docs[e1] & entity_docs[e2])
            if shared > 0:
                edges.append({
                    "source": e1,
                    "target": e2,
                    "weight": shared,
                    "strength": min(1.0, shared / 5)
                })

    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "total_docs": len(labels)
    })

try:
    from rag_assistant import get_assistant, RAGAssistant, MODEL_NAME
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    MODEL_NAME = "mistral"

try:
    from hybrid_classifier import HybridClassifier, create_hybrid_labels
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/api/ai/status')
def api_ai_status():
    if not HAS_RAG:
        return jsonify({
            "available": False,
            "ollama_available": False,
            "rag_ready": False,
            "chunk_count": 0,
            "error": "RAG module not available"
        })

    assistant = get_assistant()
    return jsonify({
        "available": True,
        "ollama_available": assistant.check_ollama(),
        "rag_ready": assistant.is_ready,
        "chunk_count": len(assistant.chunks) if assistant.is_ready else 0
    })

@app.route('/api/ai/initialize', methods=['POST'])
def api_ai_initialize():
    if not HAS_RAG:
        return jsonify({"success": False, "error": "RAG module not available"})

    assistant = get_assistant()
    try:
        success = assistant.initialize()
        if success:
            return jsonify({
                "success": True,
                "chunk_count": len(assistant.chunks)
            })
        else:
            if not assistant.ollama_available:
                error = f"Ollama not running or '{MODEL_NAME}' model not installed. Run: ollama pull {MODEL_NAME}"
            else:
                error = "Failed to build RAG index. Run 'python main.py classify' first to create labels.json"
            return jsonify({"success": False, "error": error})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/ai/query', methods=['POST'])
def api_ai_query():
    if not HAS_RAG:
        return jsonify({
            "answer": "AI Assistant not available. Please run setup with AI option.",
            "sources": [],
            "error": True
        })

    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({
            "answer": "Please provide a question.",
            "sources": [],
            "error": True
        })

    assistant = get_assistant()

    if not assistant.is_ready:
        assistant.initialize()

    result = assistant.query(question)
    return jsonify(result)

@app.route('/api/hybrid/classify', methods=['POST'])
def api_hybrid_classify():
    if not HAS_HYBRID:
        return jsonify({"success": False, "error": "Hybrid classifier not available"})
    
    data = request.get_json()
    model_name = data.get('model', 'dolphin-mistral')
    max_workers = data.get('max_workers', 12)
    
    try:
        from topic_classifier import load_pdfs_by_page, ensure_ollama_ready, select_model, get_optimal_workers
        import multiprocessing
        
        documents = load_pdfs_by_page(DOWNLOADS_DIR)
        if not documents:
            return jsonify({"success": False, "error": "No documents found"})
        
        ollama_ready = ensure_ollama_ready(model_name)
        
        optimal_workers = get_optimal_workers()
        labels = create_hybrid_labels(documents, model_name if ollama_ready else None, optimal_workers)
        
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2)
        
        llm_count = sum(1 for label in labels if label.get("method") == "llm")
        knn_count = sum(1 for label in labels if label.get("method") == "knn")
        rule_count = sum(1 for label in labels if label.get("method") == "rule")
        
        return jsonify({
            "success": True,
            "total_documents": len(labels),
            "llm_classified": llm_count,
            "knn_classified": knn_count,
            "rule_classified": rule_count,
            "ollama_available": ollama_ready
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/hybrid/status')
def api_hybrid_status():
    if not HAS_HYBRID:
        return jsonify({
            "available": False,
            "error": "Hybrid classifier module not available"
        })
    
    labels = load_labels()
    hybrid_labels = [label for label in labels if label.get("hybrid_classified")]
    
    methods = {"llm": 0, "knn": 0, "rule": 0}
    for label in hybrid_labels:
        method = label.get("method", "rule")
        methods[method] = methods.get(method, 0) + 1
    
    return jsonify({
        "available": True,
        "has_hybrid_labels": len(hybrid_labels) > 0,
        "total_hybrid_labels": len(hybrid_labels),
        "methods": methods,
        "labels_file_exists": os.path.exists(LABELS_FILE)
    })

def open_browser():
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()

if __name__ == '__main__':
    print("=" * 60)
    print("DOJ Epstein Files Explorer")
    print("=" * 60)
    print(f"Search Index: {'Available' if HAS_WHOOSH and os.path.exists(INDEX_DIR) else 'Not Available'}")
    print(f"Starting server at http://localhost:5000")
    print("=" * 60)
    
    open_browser()
    app.run(debug=True, host='0.0.0.0', port=5000)

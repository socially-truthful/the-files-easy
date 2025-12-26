import os
import sys
import json
import re
from pathlib import Path
from tqdm import tqdm

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    print("PyMuPDF (fitz) required. Install with: pip install pymupdf")
    HAS_PYMUPDF = False
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "data", "downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "classification")


def find_redaction_rectangles(page):
    redaction_rects = []
    
    for annot in page.annots() or []:
        if annot.type[0] in [8, 12]:
            redaction_rects.append(annot.rect)
    
    try:
        drawings = page.get_drawings()
        for d in drawings:
            if d.get("fill") == (0, 0, 0) or d.get("fill") == 0:
                if d.get("rect"):
                    rect = d["rect"]
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    if width > 20 and height < 50:
                        redaction_rects.append(fitz.Rect(rect))
    except:
        pass
    
    return redaction_rects


def extract_text_methods(page):
    results = {}
    
    try:
        results["standard"] = page.get_text("text").strip()
    except:
        results["standard"] = ""
    
    try:
        raw = page.get_text("rawdict")
        texts = []
        if raw and "blocks" in raw:
            for block in raw["blocks"]:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            t = span.get("text", "").strip()
                            if t:
                                texts.append(t)
        results["rawdict"] = " ".join(texts)
    except:
        results["rawdict"] = ""
    
    try:
        words = page.get_text("words")
        results["words"] = " ".join([w[4] for w in words if len(w) > 4])
    except:
        results["words"] = ""
    
    try:
        xml = page.get_text("xml")
        char_pattern = re.compile(r'c="([^"]+)"')
        chars = char_pattern.findall(xml)
        results["xml"] = "".join(chars)
    except:
        results["xml"] = ""
    
    try:
        d = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        texts = []
        for block in d.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            texts.append(t)
        results["dict_preserved"] = " ".join(texts)
    except:
        results["dict_preserved"] = ""
    
    return results


def analyze_page(page, page_num):
    analysis = {
        "page": page_num,
        "has_redaction_rects": False,
        "text_lengths": {},
        "potential_hidden_text": False,
        "extra_text_found": "",
        "redaction_count": 0
    }
    
    rects = find_redaction_rectangles(page)
    analysis["redaction_count"] = len(rects)
    analysis["has_redaction_rects"] = len(rects) > 0
    
    texts = extract_text_methods(page)
    analysis["text_lengths"] = {k: len(v) for k, v in texts.items()}
    
    standard_len = len(texts["standard"])
    max_other_len = max(len(texts[k]) for k in texts if k != "standard")
    
    if max_other_len > standard_len * 1.1 + 50:
        analysis["potential_hidden_text"] = True
        
        longest_key = max(texts, key=lambda k: len(texts[k]) if k != "standard" else 0)
        standard_words = set(texts["standard"].lower().split())
        other_words = set(texts[longest_key].lower().split())
        extra_words = other_words - standard_words
        if extra_words:
            analysis["extra_text_found"] = " ".join(list(extra_words)[:50])
    
    return analysis


def analyze_pdf(pdf_path):
    results = {
        "file": str(pdf_path),
        "filename": os.path.basename(pdf_path),
        "total_pages": 0,
        "pages_with_redactions": 0,
        "pages_with_hidden_text": 0,
        "page_details": []
    }
    
    try:
        doc = fitz.open(pdf_path)
        results["total_pages"] = len(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            analysis = analyze_page(page, page_num + 1)
            
            if analysis["has_redaction_rects"]:
                results["pages_with_redactions"] += 1
            if analysis["potential_hidden_text"]:
                results["pages_with_hidden_text"] += 1
                results["page_details"].append(analysis)
        
        doc.close()
    except Exception as e:
        results["error"] = str(e)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze PDFs for hidden text under redactions")
    parser.add_argument("--sample", type=int, help="Only analyze N random files")
    parser.add_argument("--output", type=str, default="redaction_report.json", help="Output report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Redaction Analyzer - Finding Hidden Text")
    print("=" * 60)
    
    downloads_path = Path(DOWNLOADS_DIR)
    pdf_files = list(downloads_path.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DOWNLOADS_DIR}")
        print("Run download step first: python main.py download")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files")
    
    if args.sample:
        import random
        pdf_files = random.sample(pdf_files, min(args.sample, len(pdf_files)))
        print(f"Sampling {len(pdf_files)} files")
    
    findings = []
    total_hidden = 0
    total_redacted = 0
    
    for pdf_file in tqdm(pdf_files, desc="Analyzing PDFs"):
        result = analyze_pdf(str(pdf_file))
        
        if result["pages_with_hidden_text"] > 0:
            findings.append(result)
            total_hidden += result["pages_with_hidden_text"]
            
            if args.verbose:
                print(f"\n[!] {result['filename']}: {result['pages_with_hidden_text']} pages with hidden text")
        
        total_redacted += result["pages_with_redactions"]
    
    findings.sort(key=lambda x: x["pages_with_hidden_text"], reverse=True)
    
    report_path = os.path.join(OUTPUT_DIR, args.output)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    report = {
        "summary": {
            "total_files_analyzed": len(pdf_files),
            "files_with_hidden_text": len(findings),
            "total_pages_with_redactions": total_redacted,
            "total_pages_with_hidden_text": total_hidden
        },
        "files": findings
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Files analyzed:           {len(pdf_files)}")
    print(f"Pages with redaction bars: {total_redacted}")
    print(f"Pages with hidden text:    {total_hidden}")
    print(f"Files with hidden text:    {len(findings)}")
    print(f"\nReport saved to: {report_path}")
    
    if findings:
        print("\nTop files with potential hidden text:")
        for f in findings[:10]:
            print(f"  {f['filename']}: {f['pages_with_hidden_text']} pages")


if __name__ == "__main__":
    main()

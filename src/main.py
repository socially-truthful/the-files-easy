import sys
import os
import webbrowser
import threading

def print_banner():
    print("""
============================================================
       DOJ EPSTEIN FILES - Document Explorer
============================================================
    """)

def print_help():
    print("""
Commands:
    download   - Download all files from justice.gov
    classify   - Extract text and classify documents (LLM for all)
    hybrid     - Hybrid classification (LLM only for important docs)
    index      - Build full-text search index
    serve      - Start the web interface
    live       - Go live! Share publicly so others can browse
    all        - Run complete pipeline (download -> hybrid -> index -> serve)
    analyze    - Find hidden text under improper redactions

Examples:
    python main.py all
    python main.py serve
    python main.py live
    python main.py download
    python main.py hybrid
    python main.py analyze
    """)

def run_download():
    print("\n[1/4] DOWNLOADING FILES")
    print("=" * 60)
    from download_epstein_files import main as download_main
    download_main()

def run_classify():
    print("\n[2/4] EXTRACTING & CLASSIFYING (Full LLM)")
    print("=" * 60)
    from topic_classifier import main as classify_main
    classify_main()

def run_hybrid():
    print("\n[2/4] EXTRACTING & CLASSIFYING (Hybrid)")
    print("=" * 60)
    print("Using hybrid approach: LLM for important docs, KNN/rules for routine docs")
    from hybrid_classifier import main as hybrid_main
    hybrid_main()

def run_index():
    print("\n[3/4] BUILDING SEARCH INDEX")
    print("=" * 60)
    from search_index import build_index
    build_index()

def run_serve():
    print("\n[4/4] STARTING WEB INTERFACE")
    print("=" * 60)
    from app import app
    print("\nOpening http://localhost:5000 in your browser...")
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
    app.run(debug=False, host='0.0.0.0', port=5000)

def run_live():
    print("\nSTARTING LIVE SERVER")
    print("=" * 60)
    from tunnel import run_live_server
    run_live_server()

def run_analyze():
    print("\nANALYZING REDACTIONS")
    print("=" * 60)
    from redaction_analyzer import main as analyze_main
    analyze_main()

def run_all():
    run_download()
    run_hybrid()
    run_index()
    run_serve()

def main():
    print_banner()

    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    commands = {
        'download': run_download,
        'classify': run_classify,
        'hybrid': run_hybrid,
        'index': run_index,
        'serve': run_serve,
        'live': run_live,
        'analyze': run_analyze,
        'all': run_all,
        'help': print_help,
        '-h': print_help,
        '--help': print_help,
    }

    if command in commands:
        try:
            commands[command]()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        except Exception as e:
            print(f"\nError: {e}")
            raise
    else:
        print(f"Unknown command: {command}")
        print_help()

if __name__ == "__main__":
    main()

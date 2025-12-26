import subprocess
import sys
import time
import threading
import webbrowser
import re

tunnel_process = None

def start_tunnel(port=5000):
    global tunnel_process
    
    print("\n" + "=" * 60)
    print("GOING LIVE (no account needed)")
    print("=" * 60)
    print("\nCreating public tunnel...")
    
    try:
        tunnel_process = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-R", f"80:localhost:{port}", "localhost.run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        public_url = None
        for line in tunnel_process.stdout:
            match = re.search(r'(https://[a-zA-Z0-9]+\.lhr\.life[^\s]*)', line)
            if not match:
                match = re.search(r'(https://[a-zA-Z0-9]+\.localhost\.run[^\s]*)', line)
            if match:
                public_url = match.group(1)
                break
            if "permission denied" in line.lower() or "error" in line.lower():
                print(f"   {line.strip()}")
        
        if public_url:
            print(f"\n  LOCAL:  http://localhost:{port}")
            print(f"  PUBLIC: {public_url}")
            print("\n  Share the PUBLIC link with others!")
            print("  They can browse and search without any setup.")
            print("\n  Press Ctrl+C to stop sharing.")
            print("=" * 60)
            return public_url
        else:
            print("\n[!] Could not get public URL.")
            print("    Make sure SSH is available (Windows 10+ has it built-in)")
            print("    Try: ssh localhost.run")
            return None
            
    except FileNotFoundError:
        print("\n[!] SSH not found.")
        print("    Windows 10+: Enable OpenSSH in Settings > Apps > Optional Features")
        print("    Or install Git for Windows (includes SSH)")
        return None
    except Exception as e:
        print(f"\n[!] Failed to create tunnel: {e}")
        return None

def stop_tunnel():
    global tunnel_process
    if tunnel_process:
        tunnel_process.terminate()
        tunnel_process = None

def run_live_server():
    from app import app
    
    server_thread = threading.Thread(
        target=lambda: app.run(debug=False, host='0.0.0.0', port=5000, threaded=True),
        daemon=True
    )
    server_thread.start()
    
    time.sleep(1)
    
    public_url = start_tunnel(5000)
    if not public_url:
        return
    
    webbrowser.open(public_url)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_tunnel()
        print("\nStopped sharing.")

if __name__ == "__main__":
    run_live_server()

import os
import sys
import requests
from urllib.parse import unquote
import time
import zipfile
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "downloads")
BASE_URL = "https://www.justice.gov"

DATASET_ZIPS = [
    ("https://www.justice.gov/epstein/files/DataSet%201.zip", "DataSet_1"),
    ("https://www.justice.gov/epstein/files/DataSet%202.zip", "DataSet_2"),
    ("https://www.justice.gov/epstein/files/DataSet%203.zip", "DataSet_3"),
    ("https://www.justice.gov/epstein/files/DataSet%204.zip", "DataSet_4"),
    ("https://www.justice.gov/epstein/files/DataSet%205.zip", "DataSet_5"),
    ("https://www.justice.gov/epstein/files/DataSet%206.zip", "DataSet_6"),
    ("https://www.justice.gov/epstein/files/DataSet%207.zip", "DataSet_7"),
    ("https://www.justice.gov/epstein/files/DataSet%208.zip", "DataSet_8"),
]

FIRST_PHASE_FILES = [
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/First%20Phase%20of%20Declassified%20Epstein%20Files/A.%20Evidence%20List%20from%20US%20v.%20Maxwell,%201.20-cr-00330%20(SDNY%202020).pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/First%20Phase%20of%20Declassified%20Epstein%20Files/B.%20Flight%20Log%20Released%20in%20US%20v.%20Maxwell,%201.20-cr-00330%20(SDNY%202020).pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/First%20Phase%20of%20Declassified%20Epstein%20Files/C.%20Contact%20Book%20(Redacted).pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/First%20Phase%20of%20Declassified%20Epstein%20Files/D.%20Masseuse%20List%20(Redacted).pdf",
]

MAXWELL_PROFFER_FILES = [
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Interview%20Transcript%20-%20Maxwell%202025.07.24%20%28Redacted%29.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Interview%20Transcript%20-%20Maxwell%202025.07.24-cft%20%28Redacted%29.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Interview%20Transcript%20-%20Maxwell%202025.07.25-cft%20%28Redacted%29.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Interview%20Transcript%20-%20Maxwell%202025.07.25%20%28Redacted%29.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Signed%20Maxwell%20Proffer%20Agreement%20%28Redacted%29.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Part%201%20-%207_24_25_Tallahassee.003.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Part%202%20-%207_24_25_Tallahassee.004.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Part%203%20-%207_24_25_Tallahassee.005%20%28R%29.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Part%204%20-%207_24_25_Tallahassee.007.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Part%205%20-%207_24_25_Tallahassee.008%20%28R%29.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Part%206%20-%207_24_25_Tallahassee.009%20%28R%29.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Part%207%20-%207_24_25_Tallahassee.010.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Test%20-%207_24_25_Tallahassee.001.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Test%20-%207_24_25_Tallahassee.002.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%201%20-%20Test%20-%207_24_25_Tallahassee.006.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%202%20-%20Part%201%20-%202025.07.25%20-%20xxx7_25.003%20%28R%29.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%202%20-%20Part%202%20-%202025.07.25%20-%20xxx7_25.004%20%28R%29.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%202%20-%20Part%203%20-%202025.07.25%20-%20xxx7_25.005.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%202%20-%20Part%204%20-%202025.07.25%20-%20xxx7_25.006%20%28R%29.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%202%20-%20Test%20-%20xxx7_25.001.wav",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Maxwell%20Proffer/Audio/Day%202%20-%20Test%20-%20xxx7_25.002.wav",
]

MEMOS_FILES = [
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Memos.%20&%20Correspondence/2020.11%20DOJ%20Office%20of%20Professional%20Responsibility%20Report%20Executive%20Summary.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Memos.%20&%20Correspondence/2020.11%20DOJ%20Office%20of%20Professional%20Responsibility%20Report.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Memos.%20&%20Correspondence/2023.06%20OIG%20Memorandum%2023-085.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Memos.%20&%20Correspondence/2023.06.27%20OIG%20Press%20Release.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Memos.%20&%20Correspondence/2023.06.27%20OIG%20Statement.pdf",
    "https://www.justice.gov/multimedia/DOJ%20Disclosures/Memos.%20&%20Correspondence/2025.02.27%20Letter%20from%20Attorney%20General%20Bondi%20to%20FBI%20Director%20Patel.pdf",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def download_and_extract_zip(url, dataset_name, output_dir, session):
    dataset_dir = os.path.join(output_dir, dataset_name)
    marker_file = os.path.join(dataset_dir, ".download_complete")
    
    if os.path.exists(marker_file):
        file_count = sum(1 for f in os.listdir(dataset_dir) if not f.startswith('.'))
        print(f"   Already downloaded ({file_count} files)")
        return file_count
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"   Downloading zip... (attempt {attempt + 1})")
            response = session.get(url, headers=HEADERS, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunks = []
            
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"   Downloading: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({pct:.1f}%)", end="\r")
            
            print(f"   Download complete, extracting..." + " " * 30)
            
            zip_data = b''.join(chunks)
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                file_count = len(zf.namelist())
                zf.extractall(dataset_dir)
            
            with open(marker_file, 'w') as f:
                f.write(f"Downloaded: {file_count} files")
            
            print(f"   Extracted {file_count} files")
            return file_count
            
        except Exception as e:
            print(f"\n   Error: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"   Failed after {max_retries} attempts")
                return 0
    
    return 0

def download_file(url, output_dir, session):
    filename = unquote(url.split("/")[-1])
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return True, filename, "skip"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=HEADERS, stream=True, timeout=120)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return True, filename, "ok"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, filename, str(e)

    return False, filename, "max retries"

def download_additional_files(files, folder_name, output_dir, session):
    folder_dir = os.path.join(output_dir, folder_name)
    os.makedirs(folder_dir, exist_ok=True)
    
    success = 0
    skipped = 0
    failed = 0
    
    for i, url in enumerate(files, 1):
        result, filename, status = download_file(url, folder_dir, session)
        
        if status == "skip":
            skipped += 1
            marker = "SKIP"
        elif status == "ok":
            success += 1
            marker = "OK"
        else:
            failed += 1
            marker = "FAIL"
        
        print(f"   [{i}/{len(files)}] [{marker}] {filename[:50]}")
    
    return success, skipped, failed


def main():
    print("=" * 60)
    print("DOJ Epstein Files Downloader")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()
    
    total_files = 0
    
    print("\n[1/4] DOWNLOADING MAIN DATASETS (ZIP FILES)")
    print("-" * 60)
    print("These contain the bulk of ~20,000 files.")
    print("-" * 60)
    
    for url, dataset_name in DATASET_ZIPS:
        print(f"\n{dataset_name}:")
        count = download_and_extract_zip(url, dataset_name, OUTPUT_DIR, session)
        total_files += count
    
    print("\n[2/4] FIRST PHASE DECLASSIFIED FILES")
    print("-" * 60)
    s, sk, f = download_additional_files(FIRST_PHASE_FILES, "First_Phase_Declassified", OUTPUT_DIR, session)
    total_files += s + sk
    
    print("\n[3/4] MAXWELL PROFFER (Transcripts + Audio)")
    print("-" * 60)
    s, sk, f = download_additional_files(MAXWELL_PROFFER_FILES, "Maxwell_Proffer", OUTPUT_DIR, session)
    total_files += s + sk
    
    print("\n[4/4] MEMORANDA AND CORRESPONDENCE")
    print("-" * 60)
    s, sk, f = download_additional_files(MEMOS_FILES, "Memos_Correspondence", OUTPUT_DIR, session)
    total_files += s + sk
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total files: ~{total_files}")
    print(f"Location: {OUTPUT_DIR}")
    print("\nFolders:")
    print("  - DataSet_1 through DataSet_8 (main EFTA files)")
    print("  - First_Phase_Declassified (flight logs, contacts, etc.)")
    print("  - Maxwell_Proffer (interview transcripts + audio)")
    print("  - Memos_Correspondence (OIG reports, letters)")

if __name__ == "__main__":
    main()

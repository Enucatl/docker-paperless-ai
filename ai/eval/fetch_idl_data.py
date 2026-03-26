
import os
import json
from pathlib import Path
import io
from datasets import load_dataset
from PIL import Image

# Configuration
DATASET_NAME = "pixparse/idl-wds"
NUM_SAMPLES = 50
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "idl"
GOLDEN_DATASET_PATH = SCRIPT_DIR / "golden_dataset.json"

def main():
    print(f"Loading samples from {DATASET_NAME}...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset in streaming mode
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    entries = []
    
    # Load existing golden dataset if it exists
    if GOLDEN_DATASET_PATH.exists():
        with open(GOLDEN_DATASET_PATH, "r") as f:
            data = json.load(f)
            # Filter out previous IDL entries if we want to refresh
            entries = [e for e in data.get("entries", []) if "idl" not in e["file_path"]]
            description = data.get("description", "")
    else:
        description = "Ground-truth dataset for offline agent evaluation."

    count = 0
    max_search = 1000 
    
    print(f"Searching for documents with 1-5 pages (target: {NUM_SAMPLES})...")
    
    for i, example in enumerate(dataset):
        if count >= NUM_SAMPLES or i >= max_search:
            break
            
        key = example.get("__key__", f"sample_{i}")
        metadata = example.get("json", {})
        
        # Check page count
        pages = metadata.get("pages", [])
        page_count = len(pages)
        
        if not (1 <= page_count <= 5):
            continue

        # Find best image/document source
        doc_data = None
        doc_ext = None
        
        if example.get("pdf"):
            doc_data = example["pdf"]
            doc_ext = "pdf"
        elif example.get("tif"):
            doc_data = example["tif"]
            doc_ext = "tif"
        elif example.get("jpg"):
            doc_data = example["jpg"]
            doc_ext = "jpg"
        elif example.get("png"):
            doc_data = example["png"]
            doc_ext = "png"
            
        if doc_data is None:
            continue

        pdf_path = OUTPUT_DIR / f"{key}.pdf"
        
        try:
            # 1. Save PDF
            if doc_ext == "pdf":
                if isinstance(doc_data, bytes):
                    with open(pdf_path, "wb") as f:
                        f.write(doc_data)
                else:
                    with open(pdf_path, "wb") as f:
                        f.write(doc_data.read())
            else:
                pil_images = []
                if isinstance(doc_data, list):
                    for img_item in doc_data:
                        if isinstance(img_item, Image.Image):
                            pil_images.append(img_item.convert("RGB"))
                        else:
                            pil_images.append(Image.open(io.BytesIO(img_item)).convert("RGB"))
                else:
                    if isinstance(doc_data, Image.Image):
                        pil_images.append(doc_data.convert("RGB"))
                    else:
                        pil_images.append(Image.open(io.BytesIO(doc_data)).convert("RGB"))
                
                if pil_images:
                    pil_images[0].save(pdf_path, "PDF", save_all=True, append_images=pil_images[1:])
            
            # 2. Extract full OCR transcript
            full_ocr = ""
            if 'ocr' in example:
                ocr_bytes = example['ocr']
                if isinstance(ocr_bytes, bytes):
                    try:
                        full_ocr = ocr_bytes.decode('utf-8')
                    except:
                        full_ocr = ocr_bytes.decode('latin-1')
                else:
                    full_ocr = str(ocr_bytes)
            
            if not full_ocr.strip() and pages:
                # Fallback to JSON metadata text
                full_ocr = "\n".join([" ".join(p.get('text', [])) for p in pages])
            
            entries.append({
                "file_path": str(pdf_path.absolute()),
                "expected_correspondent": "Unknown (IDL Sample)",
                "expected_date": "2000-01-01",
                "expected_title_contains": "",
                "original_key": key,
                "page_count": page_count,
                "expected_ocr_transcript": full_ocr
            })
            
            count += 1
            if count % 5 == 0:
                print(f"Progress: {count}/{NUM_SAMPLES} (checked {i+1} samples)")
                
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue

    with open(GOLDEN_DATASET_PATH, "w") as f:
        json.dump({
            "description": description,
            "entries": entries
        }, f, indent=2)
        
    print(f"Successfully added {count} entries with full OCR to {GOLDEN_DATASET_PATH}")

if __name__ == "__main__":
    main()

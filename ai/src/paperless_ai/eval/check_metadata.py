import os
import json
import io
from datasets import load_dataset
from PIL import Image


def main():
    print("Loading first sample from pixparse/idl-wds...")
    dataset = load_dataset("pixparse/idl-wds", split="train", streaming=True)

    for i, example in enumerate(dataset):
        print(f"\n--- Sample {i} ---")
        print(f"Key: {example.get('__key__')}")

        if "ocr" in example:
            ocr_data = example["ocr"]
            print(f"\nOCR data type: {type(ocr_data)}")
            if isinstance(ocr_data, bytes):
                print(f"OCR data (first 500 bytes): {ocr_data[:500]}")
                try:
                    # Try to decode if it's text/json
                    print(
                        f"Decoded OCR (first 500 chars): {ocr_data.decode('utf-8')[:500]}"
                    )
                except:
                    print("Could not decode OCR as utf-8")
            else:
                print(f"OCR data: {str(ocr_data)[:500]}")

        metadata = example.get("json", {})
        if "pages" in metadata:
            full_text = ""
            for p in metadata["pages"]:
                full_text += " ".join(p.get("text", [])) + "\n"
            print(
                f"\nFull text from JSON metadata (first 500 chars):\n{full_text[:500]}"
            )
            print(f"Total length: {len(full_text)}")

        break


if __name__ == "__main__":
    main()

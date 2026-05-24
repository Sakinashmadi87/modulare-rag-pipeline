# parser.py
import os
import json
import sys
from pathlib import Path

# --- Importiere config.py ---
sys.path.append(os.getcwd())
try:
    import config
except ImportError as e:
    raise ImportError(f"config.py nicht gefunden: {e}")

# --- Lade Pfad aus config ---
PDF_DIR = config.PATHS["pdfs_active"]  # ✅ Jetzt korrekt: /kaggle/input/.../pdfs_active
OUTPUT_DIR = config.PATHS["markdown"]   # ✅ /kaggle/working/papers/extracted_markdown
CHECKPOINT_FILE = "checkpoint.json"     # ✅ Speichert im aktuellen Verzeichnis

# --- Checkpoint-Manager ---
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(done_files):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done_files), f, ensure_ascii=False, indent=2)

# --- Parser-Klasse ---
class ArxivParser:
    def __init__(self, method="PyMuPDF4LLM"):
        if method not in ["PyMuPDF4LLM", "Docling"]:
            raise ValueError("Methode muss 'PyMuPDF4LLM' oder 'Docling' sein.")
        self.method = method

    def parse(self, pdf_path):
        if self.method == "PyMuPDF4LLM":
            try:
                import pymupdf4llm
                return pymupdf4llm.to_markdown(pdf_path)
            except Exception as e:
                raise RuntimeError(f"PyMuPDF4LLM fehlgeschlagen: {e}")

        elif self.method == "Docling":
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(pdf_path)
                markdown_text = result.document.export_to_markdown()
                return markdown_text
            except Exception as e:
                raise RuntimeError(f"Docling fehlgeschlagen: {e}")

        else:
            raise ValueError(f"Unbekannte Methode: {self.method}")

# --- Haupt-Logik ---
def main():
    # Sicherstellen, dass Ausgabeverzeichnis existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = ArxivParser(method="Docling")  # Ändere zu "PyMuPDF4LLM", wenn gewünscht
    done_files = load_checkpoint()

    # Liste aller PDFs
    all_pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    total = len(all_pdfs)
    print(f"Starte Parsing von {total} PDFs...")

    for i, filename in enumerate(all_pdfs):
        if filename in done_files:
            continue  # Überspringe bereits verarbeitete Dateien

        pdf_path = os.path.join(PDF_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".md"))

        try:
            print(f"[{i+1}/{total}] Verarbeite: {filename}")
            content = parser.parse(pdf_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Füge zur Checkpoint-Liste hinzu
            done_files.add(filename)
            save_checkpoint(done_files)

            # Fortschrittsanzeige alle 100 Dateien
            if (i + 1) % 100 == 0:
                print(f"✅ Fortschritt: {i+1}/{total} abgeschlossen.")

        except Exception as e:
            print(f"❌ Fehler bei {filename}: {e}")

    print("✅ Alle PDFs verarbeitet oder bereits abgeschlossen.")
    print("Checkpoint aktualisiert. Du kannst das Skript jederzeit neu starten.")

if __name__ == "__main__":
    main()
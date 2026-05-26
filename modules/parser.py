# parser.py
import os
import json
import sys

# --- 1. Importiere config.py ---
sys.path.append(os.getcwd())
try:
    import config
except ImportError as e:
    raise ImportError(f"config.py konnte nicht geladen werden. Liegt sie im selben Ordner? Fehler: {e}")

# --- 2. Pfade aus der config.py laden ---
PDF_DIR = config.PATHS["pdfs_active"]       # Wo liegen die PDFs?
OUTPUT_DIR = config.PATHS["markdown"]       # Wo sollen die MD-Dateien hin?
# Checkpoint wird im Ausgabeverzeichnis der jeweiligen Cloud gespeichert
CHECKPOINT_FILE = os.path.join(config.PATHS["output_root"], "checkpoint_parsing.json")

# --- 3. Checkpoint-Manager (Fortschritt sichern) ---
def load_checkpoint():
    """Lädt die Liste der bereits erfolgreich verarbeiteten PDFs."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(done_files):
    """Speichert den aktuellen Fortschritt, damit man nach Abbruch weitermachen kann."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done_files), f, ensure_ascii=False, indent=2)

# --- 4. Parser-Klasse (Modular für AutoML) ---
class ArxivParser:
    def __init__(self, method="Docling"):
        if method not in ["PyMuPDF4LLM", "Docling"]:
            raise ValueError("Methode muss 'PyMuPDF4LLM' oder 'Docling' sein.")
        self.method = method
        self.converter = None
        
        # WICHTIG: Docling nur EINMAL starten, damit die Modelle auf der GPU bleiben
        if self.method == "Docling":
            print("🚀 Initialisiere Docling (Modelle werden in den Grafikspeicher geladen)...")
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()

    def parse(self, pdf_path):
        if self.method == "PyMuPDF4LLM":
            try:
                import pymupdf4llm
                return pymupdf4llm.to_markdown(pdf_path)
            except Exception as e:
                raise RuntimeError(f"PyMuPDF4LLM-Fehler: {e}")

        elif self.method == "Docling":
            try:
                # Nutzt den bereits gestarteten Converter auf der GPU
                result = self.converter.convert(pdf_path)
                return result.document.export_to_markdown()
            except Exception as e:
                raise RuntimeError(f"Docling-Fehler: {e}")

# --- 5. Hauptprogramm ---
def main():
    # Sicherstellen, dass der Ausgabe-Ordner existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parser starten (Hier kannst du zwischen "Docling" und "PyMuPDF4LLM" wählen)
    parser = ArxivParser(method="Docling") 
    done_files = load_checkpoint()

    # Alle PDFs einlesen und STRIKT sortieren (Wichtig für identische Viertel auf beiden Laptops!)
    all_pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    total_all = len(all_pdfs)
    
    if total_all == 0:
        print(f"❌ Keine PDFs im Ordner gefunden! Pfad prüfen: {PDF_DIR}")
        return

    # ====================================================
    # 🔥 HIER DEIN PAKET WÄHLEN (Ändere diesen Wert je nach Durchlauf)
    # Erlaubte Werte: "V1", "V2", "V3", "V4" oder "ALL"
    RUN_MODE = "V1"  
    # ====================================================

    # Berechne die Grenzen für die 4 Viertel
    q1 = total_all // 4
    q2 = q1 * 2
    q3 = q1 * 3

    if RUN_MODE == "V1":
        active_pdfs = all_pdfs[:q1]
        print(f"\n📦 MODUS VIERTEL 1: Verarbeite Dokument 1 bis {q1} ({len(active_pdfs)} Dateien)")
    elif RUN_MODE == "V2":
        active_pdfs = all_pdfs[q1:q2]
        print(f"\n📦 MODUS VIERTEL 2: Verarbeite Dokument {q1+1} bis {q2} ({len(active_pdfs)} Dateien)")
    elif RUN_MODE == "V3":
        active_pdfs = all_pdfs[q2:q3]
        print(f"\n📦 MODUS VIERTEL 3: Verarbeite Dokument {q2+1} bis {q3} ({len(active_pdfs)} Dateien)")
    elif RUN_MODE == "V4":
        active_pdfs = all_pdfs[q3:]
        print(f"\n📦 MODUS VIERTEL 4: Verarbeite Dokument {q3+1} bis {total_all} ({len(active_pdfs)} Dateien)")
    else:
        active_pdfs = all_pdfs
        print(f"\n📦 MODUS ALL: Verarbeite alle {total_all} Dateien")

    total_active = len(active_pdfs)
    print(f"Bisher bereits verarbeitet (laut Checkpoint): {len(done_files & set(active_pdfs))}/{total_active}\n")

    # Der eigentliche Verarbeitungs-Loop
    for i, filename in enumerate(active_pdfs):
        if filename in done_files:
            continue  # Bereits fertig geparst -> Überspringen

        pdf_path = os.path.join(PDF_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".md"))

        try:
            print(f"[{i+1}/{total_active}] Verarbeite: {filename}")
            
            # Text extrahieren
            content = parser.parse(pdf_path)
            
            # Als Markdown speichern
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Fortschritt im Checkpoint sichern
            done_files.add(filename)
            save_checkpoint(done_files)

            # Alle 10 Dateien eine kurze Erfolgsmeldung in der Konsole
            if (i + 1) % 10 == 0:
                print(f"✨ Zwischenstand: {i+1}/{total_active} Dateien in diesem Block fertig.")

        except Exception as e:
            # Wenn ein Paper fehlerhaft ist, loggen wir es und machen mit dem nächsten weiter
            print(f"⚠️ Fehler bei Datei {filename}: {e}")
            continue

    print(f"\n✅ Durchlauf für Modus '{RUN_MODE}' erfolgreich beendet!")
    print(f"Die Markdown-Dateien liegen bereit in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
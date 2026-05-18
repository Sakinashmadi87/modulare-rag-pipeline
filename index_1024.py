import os
import json
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
import time
from qdrant_client.http.exceptions import ResponseHandlingException

# Modules
from modules.cleaner import clean_scientific_markdown
from modules.chunker import ArxivChunker
from config import PATHS, IS_COLAB, IS_KAGGLE

# --- CONFIG DYNAMISCH FÜR LOKAL ODER CLOUD ---
if IS_COLAB:
    # 1. GOOGLE COLAB MODUS
    BASE_DATA_PATH = "/content/drive/MyDrive/rag_ml_data"
    MD_DIR = os.path.join(BASE_DATA_PATH, "papers", "extracted_markdown")
    CHECKPOINT = os.path.join(BASE_DATA_PATH, "papers", "checkpoint_1024.txt")
    device = "cuda"
    print("🤖 [INDEXER] Google Colab erkannt! Nutze GPU und Google Drive.")
    
elif IS_KAGGLE:
    # 2. KAGGLE CLOUD MODUS (NEU)
    # Kaggle Datasets liegen immer unter /kaggle/input/[dataset-name]
    # Arbeitsdaten (Checkpoints) müssen in den beschreibbaren /kaggle/working Ordner
    MD_DIR = "/kaggle/input/rag-ml-data/papers/extracted_markdown"
    CHECKPOINT = "/kaggle/working/checkpoint_1024.txt"
    device = "cuda"  # Zündet die kostenlose T4 GPU auf Kaggle!
    print("🦅 [INDEXER] Kaggle Umgebung erkannt! Nutze T4 GPU.")
    
else:
    # 3. LOKALER SURFACE LAPTOP MODUS
    MD_DIR = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\extracted_markdown"
    DB_PATH = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024"
    CHECKPOINT = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\checkpoint_1024.txt"
    device = "cpu"
    print("🏠 [INDEXER] Lokale Umgebung erkannt! Nutze CPU.")

# --- INITIALISIERE QDRANT CLIENT ---
if IS_COLAB or IS_KAGGLE:
    print("🌐 Verbinde mit Qdrant CLOUD Cluster...")
    client = qdrant_client.QdrantClient(
        url=os.getenv('QDRANT_URL'), 
        api_key=os.getenv('QDRANT_API_KEY'),
        check_compatibility=False,
        timeout=60.0
    )
    device = "cuda"  # Zündet die T4-GPU in Colab
else:
    # Unverändert für Ihren Surface Laptop
    MD_DIR = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\extracted_markdown"
    DB_PATH = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024"
    CHECKPOINT = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\checkpoint_1024.txt"
    
    print("🏠 Initialisiere lokalen Qdrant Client...")
    client = qdrant_client.QdrantClient(path=DB_PATH)
    device = "cpu"
   
NEW_CHUNK_SIZE = 1024
NEW_OVERLAP = 100
collection_name = "stage2_chunks_1024"

# --- SETUP MODEL & COLLECTION ---
print(f"🖥️ Lade BGE-M3 Modell auf: {device.upper()}...")
model = SentenceTransformer('BAAI/bge-m3', device=device)

# Erstelle Collection falls nicht vorhanden
if not client.collection_exists(collection_name):
    print(f"✨ Erstelle neue Collection in der Cloud: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

chunker = ArxivChunker(chunk_size=NEW_CHUNK_SIZE, chunk_overlap=NEW_OVERLAP)

# --- RESUME LOGIC ---
done_files = []
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, "r") as f:
        done_files = f.read().splitlines()

# Sicherstellen, dass der Ordner existiert
if os.path.exists(MD_DIR):
    all_files = [f for f in os.listdir(MD_DIR) if f.endswith(".md")]
else:
    print(f"❌ Fehler: Ordner nicht gefunden unter {MD_DIR}")
    all_files = []

files_to_process = [f for f in all_files if f not in done_files]

# --- LOOP (Läuft jetzt blitzschnell auf GPU!) ---
print(f"Processing {len(files_to_process)} papers in the Cloud.")

# Maximale Anzahl an Chunks, die wir auf einmal hochladen (schützt vor 400 Bad Request)
UPLOAD_BATCH_SIZE = 32

for filename in tqdm(files_to_process, desc="Total Progress"):
    file_path = os.path.join(MD_DIR, filename)
    paper_id = filename.replace(".md", "").replace("_", "/")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
    
    cleaned_md = clean_scientific_markdown(raw_md, replace_math=True)
    chunks = chunker.splitter.split_text(cleaned_md)
    
    if chunks:
        # 1. Berechne alle Embeddings für dieses Paper auf der GPU
        embeddings = model.encode(chunks, show_progress_bar=False)
        
        # 2. Erzeuge alle Point-Strukturen
        all_points = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(f"{paper_id}_{i}_1024".encode()).hexdigest()
            all_points.append(PointStruct(
                id=chunk_id,
                vector=vector.tolist(),
                payload={
                    "paper_id": paper_id,
                    "text_llm": chunk_text,
                    "chunk_size": 1024
                }
            ))
        
       
        # 3. Sicherheits-Schleife: Lade die Chunks in kleinen Portionen mit Retry-Logik hoch
        for k in range(0, len(all_points), UPLOAD_BATCH_SIZE):
            chunk_slice = all_points[k:k + UPLOAD_BATCH_SIZE]
            
            # Versuche den Upload bis zu 3-mal, falls das Internet laggt
            for attempt in range(3):
                try:
                    client.upsert(collection_name=collection_name, points=chunk_slice)
                    break  # Erfolgreich! Schleife verlassen und mit nächstem Slice weitermachen
                except Exception as e:
                    if attempt < 2:
                        print(f"\n⚠️ Netzwerk-Verzögerung bei {filename}. Starte Versuch {attempt + 2}/3 in 5 Sekunden...")
                        time.sleep(5)
                    else:
                        raise e  # Wenn es nach 3 Versuchen nicht klappt, wirf den Fehler


    # 4. Save Checkpoint (Direkt nach jedem erfolgreichen Paper)
    with open(CHECKPOINT, "a") as f:
        f.write(filename + "\n")

print("🚀 1024-Chunk Indexing Complete!")

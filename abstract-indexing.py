import json
import os
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import PATHS, IS_COLAB

# --- SELECTION OF TARGETS DEPENDING ON PLATFORM ---
if IS_COLAB:
    # Google Drive Pfade (Stellen Sie sicher, dass diese Dateien in Drive liegen!)
    BASE_DATA_PATH = "/content/drive/MyDrive/rag_ml_data"
    input_file = os.path.join(BASE_DATA_PATH, "modern_golden_set.jsonl")
    checkpoint_file = os.path.join(BASE_DATA_PATH, "indexing_checkpoint_only_pdfs.txt")
    pdf_active_dir = os.path.join(BASE_DATA_PATH, "data", "papers", "pdfs_active")
    
    # Qdrant Cloud Client
    print("🌐 Initialisiere Qdrant CLOUD Client...")
    client = qdrant_client.QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
    device = "cuda"  # Schaltet die T4-GPU ein!
else:
    # Lokale Pfade für Ihr Surface Laptop
    base_path = r"C:\Users\ahmad\Desktop\rag_ml\data\papers"
    input_file = os.path.join(base_path, "modern_golden_set.jsonl")
    checkpoint_file = os.path.join(base_path, "indexing_checkpoint_only_pdfs.txt")
    pdf_active_dir = os.path.join(base_path, "pdfs_active")
    db_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
    
    print("🏠 Initialisiere lokalen Qdrant Client...")
    client = qdrant_client.QdrantClient(path=db_path)
    device = "cpu"

collection_name = "stage1_abstracts_only_active_pdfs"

# --- SETUP MODEL ---
print(f"🖥️ Lade BGE-M3 Modell auf: {device.upper()}...")
model = SentenceTransformer('BAAI/bge-m3', device=device)

# Scanne verfügbare PDFs im jeweiligen Zielordner
print(f"Scanne verfügbare PDFs in: {pdf_active_dir}")
if os.path.exists(pdf_active_dir):
    available_pdfs = {f.replace(".pdf", "") for f in os.listdir(pdf_active_dir) if f.endswith(".pdf")}
else:
    print(f"⚠️ Warnung: Ordner {pdf_active_dir} nicht gefunden. Nutze leere Liste.")
    available_pdfs = set()
print(f"Gefunden: {len(available_pdfs)} aktive PDFs. Nur diese werden indexiert.")

# Collection erstellen falls nicht vorhanden
if not client.collection_exists(collection_name):
    print(f"✨ Erstelle neue Collection in der Cloud: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

# Checkpoint laden
start_line = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        start_line = int(f.read().strip())
    print(f"🔄 Fortsetzen ab Quellzeile {start_line}...")

# --- INDEXIERUNG LOOP ---
batch_size = 32  # Höhere Batch-Größe für die Cloud-GPU (beschleunigt den Prozess)
batch_papers = []
indexed_count = 0

print(f"🚀 Starte Indexierung (Filter: NUR Paper mit vorhandenem PDF)...")

if os.path.exists(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
                
            paper = json.loads(line)
            safe_id = paper['id'].replace('/', '_')
            
            # FILTER: Nur verarbeiten, wenn das PDF hochgeladen wurde
            if safe_id in available_pdfs:
                batch_papers.append(paper)
            
            if len(batch_papers) == batch_size:
                # Metadata Fusing
                texts_to_embed = []
                for p in batch_papers:
                    kws = ", ".join(p.get('keywords', []))
                    combined = f"Title: {p['title_clean']} | Keywords: {kws} | Abstract: {p['abstract_clean']}"
                    texts_to_embed.append(combined)
                
                # Embeddings berechnen (blitzschnell auf GPU)
                embeddings = model.encode(texts_to_embed, show_progress_bar=False)
                
                points = []
                for j in range(len(batch_papers)):
                    p_data = batch_papers[j]
                    s_id = p_data['id'].replace('/', '_')
                    
                    points.append(PointStruct(
                        id=indexed_count + j + 1,  # Fortlaufende ID
                        vector=embeddings[j].tolist(),
                        payload={
                            "paper_id": p_data['id'],
                            "file_id": s_id,
                            "title": p_data['title_clean'],
                            "has_full_text": True
                        }
                    ))
                
                # Upload zur Qdrant Cloud
                client.upsert(collection_name=collection_name, points=points)
                indexed_count += len(batch_papers)
                
                # Checkpoint speichern (direkt im Google Drive)
                with open(checkpoint_file, "w") as cf:
                    cf.write(str(i + 1))
                
                print(f"Line {i+1} verarbeitet. Indexierte Paper in Cloud: {indexed_count}")
                batch_papers = []

    # Letzten Batch verarbeiten (Restliche Dateien)
    if batch_papers:
        texts_to_embed = [f"Title: {p['title_clean']} | Keywords: {', '.join(p.get('keywords', []))} | Abstract: {p['abstract_clean']}" for p in batch_papers]
        embeddings = model.encode(texts_to_embed, show_progress_bar=False)
        points = [
            PointStruct(
                id=indexed_count + j + 1, 
                vector=embeddings[j].tolist(), 
                payload={"paper_id": batch_papers[j]['id'], "file_id": batch_papers[j]['id'].replace('/', '_'), "title": batch_papers[j]['title_clean'], "has_full_text": True}
            ) for j in range(len(batch_papers))
        ]
        client.client.upsert(collection_name=collection_name, points=points)
        indexed_count += len(batch_papers)
        
    print(f"🏁 Fertig! Insgesamt {indexed_count} Abstracts erfolgreich in die Qdrant Cloud geladen.")
else:
    print(f"❌ Fehler: {input_file} wurde nicht im Drive gefunden!")

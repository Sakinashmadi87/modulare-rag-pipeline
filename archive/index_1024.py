import os
import json
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# Modules
from modules.cleaner import clean_scientific_markdown
from modules.chunker import ArxivChunker # Make sure this is updated to 1024

# --- CONFIG ---
MD_DIR = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\extracted_markdown"
DB_PATH = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024"
CHECKPOINT = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\checkpoint_1024.txt"
NEW_CHUNK_SIZE = 1024
NEW_OVERLAP = 100

# --- SETUP ---
print("Initializing 1024-Chunk Indexer...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')
client = qdrant_client.QdrantClient(path=DB_PATH)
collection_name = "stage2_chunks_1024"

if not any(c.name == collection_name for c in client.get_collections().collections):
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

all_files = [f for f in os.listdir(MD_DIR) if f.endswith(".md")]
files_to_process = [f for f in all_files if f not in done_files]

# --- OVERNIGHT LOOP ---
print(f"Processing {len(files_to_process)} papers. Estimated time: 7 hours.")

for filename in tqdm(files_to_process, desc="Total Progress"):
    file_path = os.path.join(MD_DIR, filename)
    paper_id = filename.replace(".md", "").replace("_", "/")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
    
    # 1. Cleaning (Using Math Heuristic for the embedding path)
    cleaned_md = clean_scientific_markdown(raw_md, replace_math=True)
    
    # 2. Chunking (Recursive)
    # We use a simple meta_lookup mock here or import your real one
    chunks = chunker.splitter.split_text(cleaned_md)
    
    if chunks:
        # 3. Batch Embedding
        embeddings = model.encode(chunks, show_progress_bar=False)
        
        points = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(f"{paper_id}_{i}_1024".encode()).hexdigest()
            points.append(PointStruct(
                id=chunk_id,
                vector=vector.tolist(),
                payload={
                    "paper_id": paper_id,
                    "text_llm": chunk_text, # In a real AutoML, you'd store raw here
                    "chunk_size": 1024
                }
            ))
        
        client.upsert(collection_name=collection_name, points=points)

    # 4. Save Checkpoint
    with open(CHECKPOINT, "a") as f:
        f.write(filename + "\n")

print("🚀 1024-Chunk Indexing Complete! Your system is ready for the morning AutoML run.")

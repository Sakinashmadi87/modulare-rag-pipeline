import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from config import EXPANSION_RULES

class HybridRetriever:
    def __init__(self, db_path):
        main_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
        self.main_client = qdrant_client.QdrantClient(path=main_path)
        
        # If we are testing the 1024-db, create a second client.
        # If we are testing the 512-db (same path), just reuse the first one.
        if db_path.lower() == main_path.lower():
            self.chunk_client = self.main_client
        else:
            self.chunk_client = qdrant_client.QdrantClient(path=db_path)
            
        self.expansion_rules = EXPANSION_RULES

    def search(self, query_text, model, top_k_papers=3, top_k_chunks=5, use_expansion=False, collection_name="stage2_chunks_modern"):
        processed_query = self.expand_query(query_text) if use_expansion else query_text
        query_vector = model.encode(processed_query).tolist()

        # --- STAGE 1: Immer aus der MAIN_CLIENT ---
        stage1_results = self.main_client.query_points(
            collection_name="stage1_abstracts_only_active_pdfs",
            query=query_vector,
            limit=top_k_papers
        ).points
        
        raw_ids = [res.payload['paper_id'] for res in stage1_results]
        if not raw_ids: return []

        relevant_paper_ids = []
        for p_id in raw_ids:
            relevant_paper_ids.append(p_id)
            relevant_paper_ids.append(p_id.replace('.', '_'))

        # --- STAGE 2: Aus dem CHUNK_CLIENT ---
        stage2_results = self.chunk_client.query_points(
            collection_name=collection_name, 
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchAny(any=relevant_paper_ids))]
            ),
            limit=top_k_chunks
        ).points

        return stage2_results

    def expand_query(self, query):
        expanded = query.lower()
        for key, expansion in self.expansion_rules.items():
            if key in expanded:
                expanded += f" {expansion}"
        return expanded

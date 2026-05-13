class RAGEvaluator:
    def __init__(self, eval_data):
        self.eval_data = eval_data

    def run_benchmark(self, retriever, model, config):
        hits = 0
        for entry in self.eval_data:
            query = entry['question']
            # Erwartete ID normalisieren (immer Unterstrich)
            expected_id = entry['paper_id'].replace('.', '_').replace('/', '_')
            
            results = retriever.search(
                query, model, 
                top_k_papers=config.get('top_k_papers', 10), 
                top_k_chunks=config.get('top_k_chunks', 10),
                use_expansion=config.get('use_expansion', False),
                collection_name=config.get('collection_name', "stage2_chunks_modern")
            )
            
            # Gefundene IDs normalisieren
            found_ids = [
                str(hit.payload.get('paper_id', '')).replace('.', '_').replace('/', '_') 
                for hit in results
            ]
            
            if expected_id in found_ids:
                hits += 1
            # Optional: Hier könnte man ein Print für Fehlversuche einbauen

        return (hits / len(self.eval_data)) * 100 if self.eval_data else 0

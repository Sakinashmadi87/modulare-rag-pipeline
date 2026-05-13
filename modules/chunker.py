import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ArxivChunker:
    def __init__(self, chunk_size=512, chunk_overlap=51):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Initialize the splitter with your Grid parameters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )

    def chunk_and_fuse(self, md_content, paper_id, meta_lookup):
        """
        Takes raw markdown, splits it, and fuses each chunk with metadata.
        """
        # 1. Get metadata for this specific paper
        paper_meta = meta_lookup.get(paper_id, {"title": "Unknown", "abstract": ""})
        
        # 2. Split text
        raw_chunks = self.splitter.split_text(md_content)
        
        fused_chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Metadata Fusing (Title + Abstract Prefix) - Key for Stage 2 accuracy
            fused_text = (
                f"TITLE: {paper_meta['title']}\n"
                f"ABSTRACT: {paper_meta['abstract'][:200]}...\n\n"
                f"CHUNK CONTENT:\n{chunk}"
            )
            
            fused_chunks.append({
                "paper_id": paper_id,
                "chunk_index": i,
                "text": fused_text,
                "metadata": {
                    "title": paper_meta['title'],
                    "chunk_size": self.chunk_size,
                    "overlap": self.chunk_overlap
                }
            })
        return fused_chunks

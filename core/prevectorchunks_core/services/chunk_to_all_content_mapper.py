import numpy as np


class ChunkMapper:
    def __init__(self, client, markdown_output, embedding_model="text-embedding-3-small"):
        """
        client: OpenAI client object
        markdown_output: list of JSON objects containing at least 'markdown_text'
        embedding_model: model for embeddings
        """
        self.client = client
        self.markdown_output = markdown_output
        self.embedding_model = embedding_model

        # Precompute embeddings for markdown_output
        self.markdown_embeddings = self._compute_markdown_embeddings()

    # -----------------------------
    # Compute embeddings for all markdown items
    # -----------------------------
    def _compute_markdown_embeddings(self):
        embeddings = []
        for obj in self.markdown_output:
            markdown_text = obj.get("markdown_text", "")
            emb = self._get_embedding(markdown_text)
            embeddings.append(emb)
        return embeddings

    # -----------------------------
    # Embedding helper
    # -----------------------------
    def _get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    # -----------------------------
    # Cosine similarity
    # -----------------------------
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # -----------------------------
    # Find best matching markdown JSON for a chunk
    # -----------------------------
    def _find_best_match(self, chunk_text):
        chunk_emb = self._get_embedding(chunk_text)

        best_score = -1
        best_index = None

        for idx, md_emb in enumerate(self.markdown_embeddings):
            score = self._cosine_similarity(chunk_emb, md_emb)
            if score > best_score:
                best_score = score
                best_index = idx

        if best_index is not None:
            return self.markdown_output[best_index]
        return None

    # -----------------------------
    # Map list of chunked texts to new JSON array
    # -----------------------------
    def map_chunks(self, chunked_texts):
        new_json_array = []
        for chunk_text in chunked_texts:
            best_match = self._find_best_match(chunk_text)
            if best_match:
                combined_obj = dict(best_match)  # copy matched JSON
                combined_obj["chunked_text"] = chunk_text
                new_json_array.append(combined_obj)
        return new_json_array

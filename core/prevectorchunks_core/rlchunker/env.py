import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .reward import compute_reward
from .utils import split_sentences, get_embeddings


# %%
class ChunkEnvBatch:
    def __init__(self, docs, min_len=1, max_len=10):
        self.docs = docs
        self.min_len = min_len
        self.max_len = max_len
        self._prepare_documents()
        self.final_chunks = []  # <-- store finalized chunks

    def _prepare_documents(self):
        self.doc_sentences = [split_sentences(doc) for doc in self.docs]
        self.doc_embeddings = [get_embeddings(sents) for sents in self.doc_sentences]
        self.num_docs = len(self.docs)

    def reset_doc(self, doc_idx):
        self.pos = 0
        self.current_chunk = []
        self.current_chunk_embeddings = []
        self.sentences = self.doc_sentences[doc_idx]
        self.embeddings = self.doc_embeddings[doc_idx]
        self.final_chunks = []  # <-- store finalized chunks
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.pos >= len(self.sentences):
            self.done = True
            return None

        sent_emb = self.embeddings[self.pos]
        device = sent_emb.device  # ensures concatenation on same device

        chunk_len = len(self.current_chunk)

        if len(self.current_chunk_embeddings) > 0:
            # stack list of tensors into a single tensor
            chunk_tensor = torch.stack(self.current_chunk_embeddings)
            avg_cos = np.mean(cosine_similarity(
                chunk_tensor.cpu().numpy(),
                sent_emb.cpu().numpy().reshape(1, -1)
            ))
        else:
            avg_cos = 1.0

        chunk_stats = torch.tensor([chunk_len, avg_cos], dtype=torch.float, device=device)
        state = torch.cat([sent_emb, chunk_stats])
        return state

    def step(self, action, reward_threshold=0.2):
        reward = 0.0
        sent_emb = self.embeddings[self.pos]
        self.current_chunk.append(self.sentences[self.pos])
        self.current_chunk_embeddings.append(sent_emb)
        self.pos += 1

        if action == 1 or self.pos == len(self.sentences):
            # compute reward for the current chunk
            chunk_reward = compute_reward(torch.stack(self.current_chunk_embeddings), self.min_len, self.max_len)

            # compute reward if we added the next sentence (lookahead)
            chunk_reward_for_next_sent = -float('inf')
            if self.pos < len(self.sentences):  # make sure next sentence exists
                next_emb = self.embeddings[self.pos]
                temp_embeddings = self.current_chunk_embeddings + [next_emb]
                chunk_reward_for_next_sent = compute_reward(torch.stack(temp_embeddings), self.min_len, self.max_len)

            # finalize the chunk only if reward is high and better than adding next sentence
            if chunk_reward >= reward_threshold and chunk_reward >= chunk_reward_for_next_sent:
                reward += chunk_reward
                self.final_chunks.append(self.current_chunk.copy())
                self.current_chunk = []
                self.current_chunk_embeddings = []
            else:
                # ❌ if reward is low or next sentence improves reward, keep extending the chunk
                reward += 0.0

        done = self.pos >= len(self.sentences)
        next_state = self._get_state()
        return next_state, reward, done

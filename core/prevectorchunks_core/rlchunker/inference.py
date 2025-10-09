import os
import torch
from .model import build_policy
from .env import ChunkEnvBatch

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "policy_model.pt")
DEFAULT_INFO_PATH  = os.path.join(os.path.dirname(__file__), "pretrained", "model_info.txt")

class RLChunker:
    """
    Class-based wrapper for RL text chunking.
    Loads pretrained policy and provides easy chunk_text() method.
    """

    def __init__(self, local_path=None, device="cpu", embedding_dim=384):
        self.device = device
        self.embedding_dim = embedding_dim
        self.path = local_path or DEFAULT_MODEL_PATH
        self.policy = None
        self._load_policy()

    def _load_policy(self):
        """Load pretrained policy model"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Pretrained model not found: {self.path}")

        print(f"🔹 Loading pretrained policy from {self.path} onto {self.device}")
        self.policy = build_policy(self.embedding_dim, device=self.device)
        self.policy.load_state_dict(torch.load(self.path, map_location=self.device))
        self.policy.to(self.device)
        self.policy.eval()
        print("✅ Policy loaded successfully")

    def chunk_text(self, text, threshold=0.5, min_len=5, max_len=50):
        """
        Run RL-based chunking on a single text string.
        Returns a list of chunks.
        """
        if self.policy is None:
            raise ValueError("Policy not loaded")

        # Clean text
        clean_text = " ".join(text.split())
        env = ChunkEnvBatch([clean_text], min_len=min_len, max_len=max_len)
        state = env.reset_doc(0)

        # Disable gradients for inference
        self.policy.eval()
        chunks = []
        with torch.no_grad():
            while not env.done:
                state = state.to(self.device)
                prob_split = self.policy(state)  # model outputs probability of split
                m = torch.distributions.Bernoulli(prob_split)
                action = m.sample()
                state, reward, done = env.step(action)

        return env.final_chunks

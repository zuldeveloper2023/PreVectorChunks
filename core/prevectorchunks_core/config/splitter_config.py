# prevectorchunks_core/config.py
from dataclasses import dataclass, field

@dataclass()
class SplitterConfig:
    chunk_size: int = 300
    chunk_overlap: int = 0
    separators: list[str] = field(default_factory=lambda: ["\n"])
    split_type: str = "recursive_splitter"



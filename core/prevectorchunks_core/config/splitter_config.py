# prevectorchunks_core/config.py
from dataclasses import dataclass, field
from enum import Enum


class LLM_Structured__Output_Type(Enum):
    RECURSIVE = "RecursiveCharacterTextSplitter"
    CHARACTER = "CharacterTextSplitter"
    STANDARD = "standard"
    R_PRETRAINED_PROPOSITION = "RLBasedTextSplitterWithProposition"
    R_PRETRAINED = "RLBasedTextSplitter"

@dataclass()
class SplitterConfig:
    chunk_size: int = 300
    chunk_overlap: int = 0
    separators: list[str] = field(default_factory=lambda: ["\n"])
    split_type: str = "recursive_splitter"
    enableLLMTouchUp: bool = True
    min_rl_chunk_size: int = 5
    max_rl_chunk_size: int = 50



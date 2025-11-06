# prevectorchunks_core/config.py
from dataclasses import dataclass, field
from enum import Enum


class LLM_Structured_Output_Type(Enum):
    STANDARD = "STANDARD"
    STRUCTURED_WITH_VECTOR_DB_ID_GENERATED = "STRUCTURED_WITH_VECTOR_DB_ID_GENERATED"


@dataclass()
class SplitterConfig:
    chunk_size: int = 300
    chunk_overlap: int = 0
    separators: list[str] = field(default_factory=lambda: ["\n"])
    split_type: str = "recursive_splitter"
    enableLLMTouchUp: bool = True
    llm_structured_output_type: LLM_Structured_Output_Type = LLM_Structured_Output_Type.STRUCTURED_WITH_VECTOR_DB_ID_GENERATED
    min_rl_chunk_size: int = 5
    max_rl_chunk_size: int = 50



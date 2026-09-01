from dataclasses import dataclass
from datetime import datetime


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    author: str
    title: str
    published_at: datetime
    source_url: str
    source_type: str

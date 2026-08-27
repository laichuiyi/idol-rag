from dataclasses import dataclass
from datetime import datetime


@dataclass
class BlogDocument:
    document_id: str
    author: str
    title: str
    published_at: datetime
    source_url: str
    content: str
    images: list[str]
    source_type: str = "blog"

from typing import Dict, Optional

from pydantic import BaseModel

FAMILY_DEFAULT = 'default'

NEWS_TITLE: bytes = f'{FAMILY_DEFAULT}:title'.encode()
NEWS_CONTENT: bytes = f'{FAMILY_DEFAULT}:content'.encode()
NEWS_SRC: bytes = f'{FAMILY_DEFAULT}:src'.encode()
NEWS_DATE: bytes = f'{FAMILY_DEFAULT}:date'.encode()


class News(BaseModel):
    title: str
    content: str
    src: str
    date: str
    id: Optional[str]

    @classmethod
    def table_name(cls):
        return 'news'

    @classmethod
    def of(cls, row: Dict[bytes, bytes], rid: str = None):
        return News(title=row[NEWS_TITLE].decode(),
                    content=row[NEWS_CONTENT].decode(),
                    src=row[NEWS_SRC].decode(),
                    date=row[NEWS_DATE].decode(),
                    id=rid)


class NewsMetrics(BaseModel):
    ...

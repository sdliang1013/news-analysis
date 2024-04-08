from pydantic import BaseModel


class News(BaseModel):
    title: str
    content: str
    src: str
    date: str

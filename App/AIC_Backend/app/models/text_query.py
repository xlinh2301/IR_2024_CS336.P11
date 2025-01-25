from pydantic import BaseModel
from typing import List

class TextQuery(BaseModel):
    query: str

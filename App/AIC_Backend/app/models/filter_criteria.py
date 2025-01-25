from pydantic import BaseModel

class FilterCriteria(BaseModel):
    object: str
    operator: str
    quantity: int

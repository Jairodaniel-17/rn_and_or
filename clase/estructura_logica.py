from typing import List
from pydantic import BaseModel


class LogicaProposicional(BaseModel):
    x1: List[List[int]]
    x2: List[int]

from pydantic import BaseModel, Field
from typing import List, Optional


class MovieRequest(BaseModel):
	title: str = Field(..., description="Title of the movie to base recommendations on")


class PredictionResponse(BaseModel):
	success: bool = True
	recommendations: List[str]
	posters: Optional[List[str]] = None


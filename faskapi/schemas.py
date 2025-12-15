"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple


class NERRequest(BaseModel):
    """Request model for NER prediction"""
    text: str = Field(..., description="Input text for NER prediction", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, Thành phố Hồ Chí Minh"
            }
        }


class TokenLabel(BaseModel):
    """Token and its label"""
    token: str
    label: str


class NERResponse(BaseModel):
    """Response model for NER prediction"""
    text: str = Field(..., description="Original input text")
    tokens: List[TokenLabel] = Field(..., description="List of tokens with their labels")
    entities: Dict[str, List[str]] = Field(..., description="Extracted entities grouped by type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, Thành phố Hồ Chí Minh",
                "tokens": [
                    {"token": "Đường", "label": "O"},
                    {"token": "Nguyễn", "label": "B-STREET"},
                    {"token": "Huệ", "label": "I-STREET"}
                ],
                "entities": {
                    "STREET": ["Nguyễn Huệ"],
                    "WARD": ["Bến Nghé"],
                    "DISTRICT": ["Quận 1"],
                    "PROVINCE": ["Thành phố Hồ Chí Minh"]
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import config


# --- OCR Models ---
class OcrUrlRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="URL của hình ảnh cần OCR")


class OcrResponse(BaseModel):
    extracted_text: str
    source_type: str  # "url" hoặc "upload"
    file_name: Optional[str] = None


# --- Document Search Models ---
class DocumentSearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Nội dung truy vấn tìm kiếm")
    top_k: int = Field(
        (
            config.QDRANT_SEARCH_LIMITd
            if not config.RERANKER_ACTIVE
            else config.RERANK_TOP_N
        ),
        ge=1,
        le=20,
        description="Số lượng kết quả trả về",
    )
    # Có thể thêm các trường filter khác ở đây


class RetrievedSource(BaseModel):  # Model này đã được định nghĩa ở chatbot
    source: str
    type: str
    score: float
    content_snippet: str
    document_summary: Optional[str] = None
    document_keywords: Optional[str] = None  # Hoặc List[str]


# --- Chatbot Models ---
class ChatQuery(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    # history: Optional[List[Dict[str, str]]] = None # Nếu muốn client gửi lên


class ChatbotResponse(BaseModel):  # Đổi tên từ ChatResponse để tránh trùng
    answer: str
    retrieved_sources: Optional[List[RetrievedSource]] = None
    conversation_id: Optional[str] = None

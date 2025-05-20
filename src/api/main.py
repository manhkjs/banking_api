# src/api/main.py
from fastapi import FastAPI
from .dependencies import startup_event_handler  # Sự kiện startup
from .endpoints import ocr_router, search_router, chat_router  # Import các router

app = FastAPI(
    title="Kienlongbank AI Services API",
    description="Cung cấp các dịch vụ OCR, Tìm kiếm Tài liệu và Chatbot RAG cho Kienlongbank.",
    version="1.0.0",
)


# Đăng ký sự kiện startup
@app.on_event("startup")
async def on_app_startup():
    print("Sự kiện startup của ứng dụng API...")
    startup_event_handler()  # Gọi hàm khởi tạo tài nguyên


# Include các router
app.include_router(ocr_router.router)
app.include_router(search_router.router)
app.include_router(chat_router.router)


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Chào mừng đến với Kienlongbank AI Services API!"}


# Để chạy: uvicorn src.api.main:app --reload --port 8000 (từ thư mục gốc kienlongbank_rag_project)

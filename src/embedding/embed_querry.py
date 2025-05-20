# src/embedding/embedding_service.py
import google.generativeai as genai

# from src.utils.api_key_manager import GeminiApiKeyManager # Nhận instance của manager
# import config


def embed_query_gemini(
    user_query: str,
    api_manager,  # Instance của GeminiApiKeyManager
    embedding_model_name: str,  # từ config
    task_type: str,  # từ config
) -> list[float] | None:
    print("  Đang embedding câu hỏi...")
    embedding_params = {
        "model": embedding_model_name,  # Sử dụng model_name được truyền vào
        "content": user_query,
        "task_type": task_type,  # Sử dụng task_type được truyền vào
    }

    # Gọi qua ApiKeyManager
    response = api_manager.call_embedding_model(  # Giả sử có phương thức này trong manager
        model_name=embedding_model_name,  # Hoặc api_manager không cần model_name nếu embed_content là hàm genai
        content_to_embed=user_query,  # Sửa lại cho phù hợp với call_embedding_model
        task_type=task_type,
        call_type="Embedding Query",
    )

    if response and "embedding" in response:
        return response["embedding"]
    print("    Lỗi embedding câu hỏi hoặc response không hợp lệ.")
    return None

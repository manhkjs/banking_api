# src/embedding/embedding_service.py
# import google.generativeai as genai # Không cần trực tiếp ở đây nếu api_manager xử lý
# from src.utils.api_key_manager import GeminiApiKeyManager # Sẽ nhận instance của manager
# import config # Để lấy EMBEDDING_MODEL_NAME, EMBEDDING_TASK_TYPE_DOCUMENT, EMBEDDING_BATCH_SIZE
import time


def embed_texts_in_batches(
    texts_to_embed: list[str],
    api_manager,  # Instance của GeminiApiKeyManager
    embedding_model_name: str,
    task_type: str,
    batch_size: int,
) -> list | None:
    """
    Tạo embeddings cho một danh sách các đoạn văn bản sử dụng Gemini API, xử lý theo batch.
    """
    if not texts_to_embed:
        return []

    all_embeddings = []
    num_texts = len(texts_to_embed)

    for i in range(0, num_texts, batch_size):
        batch_texts = texts_to_embed[i : i + batch_size]
        current_batch_num = i // batch_size + 1
        total_batches = (num_texts - 1) // batch_size + 1

        print(
            f"    Đang tạo embedding cho batch {current_batch_num}/{total_batches} ({len(batch_texts)} items)..."
        )

        # Các tham số cho phương thức call_embedding_model của ApiKeyManager
        embedding_api_params = {
            "content_to_embed": batch_texts,  # Tên tham số này cần khớp với những gì call_embedding_model mong đợi
            "task_type": task_type,
        }

        # Gọi qua ApiKeyManager
        response = api_manager.call_embedding_model(
            model_name=embedding_model_name,
            # method_params=embedding_api_params, # Sửa lại cho phù hợp với hàm mới
            content_to_embed=batch_texts,  # Truyền trực tiếp
            task_type=task_type,
            call_type=f"Batch Embedding {current_batch_num}/{total_batches}",
        )

        if response and "embedding" in response:
            batch_embeddings = response["embedding"]
            # embed_content trả về list of embeddings nếu content là list
            if len(batch_embeddings) == len(batch_texts):
                all_embeddings.extend(batch_embeddings)
            else:
                print(
                    f"      Lỗi: Số lượng embedding trả về ({len(batch_embeddings)}) không khớp số lượng text đầu vào ({len(batch_texts)}). Thêm None."
                )
                all_embeddings.extend(
                    [None] * len(batch_texts)
                )  # Thêm None để giữ đúng thứ tự và số lượng
        else:
            print(
                f"    Lỗi: Batch embedding {current_batch_num} không trả về kết quả hợp lệ. Thêm None."
            )
            all_embeddings.extend([None] * len(batch_texts))  # Thêm None

        if current_batch_num < total_batches:  # Tránh sleep sau batch cuối cùng
            time.sleep(0.5)  # Sleep nhỏ giữa các batch API call

    if len(all_embeddings) != num_texts:
        print(
            f"CẢNH BÁO: Số lượng embedding cuối cùng ({len(all_embeddings)}) không khớp số lượng text ban đầu ({num_texts})."
        )
        return None  # Hoặc xử lý khác

    return all_embeddings

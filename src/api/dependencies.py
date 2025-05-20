# src/api/dependencies.py
import os
import google.generativeai as genai
import networkx as nx
from qdrant_client import QdrantClient  # Import QdrantClient trực tiếp
from mistralai import Mistral
from typing import Optional

import config  # Từ thư mục gốc
from src.utils.api_key_manager import GeminiApiKeyManager
from src.knowledge_graph.kg_loader_service import load_nx_graph_from_file
from src.vector_store.qdrant_service import (
    initialize_qdrant_and_collection as init_qdrant,  # Giữ alias nếu bạn muốn
)
from src.reranking.reranker import Reranker

# ... (Khai báo các biến toàn cục _gemini_api_manager, _mistral_client, etc. giữ nguyên) ...
_gemini_api_manager: Optional[GeminiApiKeyManager] = None
_mistral_client: Optional[Mistral] = None
_knowledge_graph: Optional[nx.DiGraph] = None
_qdrant_cli: Optional[QdrantClient] = None
_reranker_instance: Optional[Reranker] = None


def startup_event_handler():
    """Khởi tạo tất cả tài nguyên dùng chung khi server API bắt đầu."""
    global _gemini_api_manager, _mistral_client, _knowledge_graph, _qdrant_cli, _reranker_instance

    print("\nDEBUG (dependencies.py): ==============================================")
    print("DEBUG (dependencies.py): Bắt đầu hàm startup_event_handler()")
    print("API Startup: Đang khởi tạo các tài nguyên dùng chung...")

    # --- Tải API Keys và Khởi tạo Managers ---
    print("\nDEBUG (dependencies.py): --- Bước 1: Khởi tạo GeminiApiKeyManager ---")
    try:
        _gemini_api_manager = GeminiApiKeyManager(
            retry_wait_seconds=config.RETRY_WAIT_SECONDS,
            max_retries_per_key_cycle=config.MAX_RETRIES_PER_KEY_CYCLE,
            api_key_prefix=config.GEMINI_API_KEY_PREFIX,
        )
        print(
            f"DEBUG (dependencies.py): GeminiApiKeyManager đã khởi tạo: {'Thành công' if _gemini_api_manager else 'Thất bại'}"
        )
        if _gemini_api_manager:
            print(
                f"DEBUG (dependencies.py): Số lượng key Gemini đã tải: {len(_gemini_api_manager.api_keys)}"
            )
    except ValueError as e:
        print(
            f"LỖI NGHIÊM TRỌNG (API Startup): Không thể khởi tạo GeminiApiKeyManager: {e}"
        )
        raise RuntimeError(f"Không thể khởi tạo GeminiApiKeyManager: {e}")
    except Exception as e_gem_man:
        print(
            f"LỖI NGHIÊM TRỌNG (API Startup): Lỗi không mong đợi khi khởi tạo GeminiApiKeyManager: {e_gem_man}"
        )
        raise RuntimeError(
            f"Lỗi không mong đợi khi khởi tạo GeminiApiKeyManager: {e_gem_man}"
        )

    print("\nDEBUG (dependencies.py): --- Bước 2: Khởi tạo Mistral Client ---")
    mistral_key = os.getenv(config.MISTRAL_API_KEY_ENV_VAR)
    if not mistral_key:
        print(
            f"CẢNH BÁO (API Startup): Biến môi trường '{config.MISTRAL_API_KEY_ENV_VAR}' chưa được thiết lập."
        )
        print(
            "                 OCR API (Mistral) sẽ không hoạt động. _mistral_client sẽ là None."
        )
        _mistral_client = None
    else:
        try:
            _mistral_client = Mistral(api_key=mistral_key)
            print("DEBUG (dependencies.py): Mistral client đã khởi tạo thành công.")
        except Exception as e_mistral:
            print(f"LỖI (API Startup): Không thể khởi tạo Mistral client: {e_mistral}.")
            print(
                "                 OCR API (Mistral) sẽ không hoạt động. _mistral_client sẽ là None."
            )
            _mistral_client = None
    print(
        f"DEBUG (dependencies.py): Trạng thái _mistral_client: {'Đã khởi tạo' if _mistral_client else 'None'}"
    )

    print(f"\nDEBUG (dependencies.py): --- Bước 3: Tải Knowledge Graph ---")
    print(f"DEBUG (dependencies.py): Đường dẫn file KG: '{config.GRAPH_FILE_TO_LOAD}'")
    if not os.path.exists(config.GRAPH_FILE_TO_LOAD):
        print(
            f"LỖI NGHIÊM TRỌNG (API Startup): Không tìm thấy file KG tại đường dẫn trên."
        )
        raise RuntimeError(f"Không tìm thấy file KG: {config.GRAPH_FILE_TO_LOAD}")

    _knowledge_graph = load_nx_graph_from_file(config.GRAPH_FILE_TO_LOAD)
    if _knowledge_graph is None:
        print("LỖI NGHIÊM TRỌNG (API Startup): Không thể tải Knowledge Graph từ file.")
        raise RuntimeError("Không thể tải Knowledge Graph.")
    print(
        f"DEBUG (dependencies.py): Knowledge Graph đã tải: {_knowledge_graph.number_of_nodes()} nút, {_knowledge_graph.number_of_edges()} cạnh."
    )

    print("\nDEBUG (dependencies.py): --- Bước 4: Khởi tạo Qdrant Client ---")
    # Kiểm tra xem QDRANT_CONNECTION_PARAMS có được cấu hình không
    if not config.QDRANT_CONNECTION_PARAMS or (
        config.QDRANT_CONNECTION_PARAMS.get("url") is None
        and config.QDRANT_CONNECTION_PARAMS.get("host") is None
    ):
        print(
            "LỖI NGHIÊM TRỌNG (API Startup): QDRANT_CONNECTION_PARAMS không được cấu hình đúng trong config.py (thiếu url hoặc host)."
        )
        print("                 Qdrant client sẽ không được khởi tạo.")
        _qdrant_cli = None  # Đặt là None để các hàm getter báo lỗi nếu được gọi
        # raise RuntimeError("QDRANT_CONNECTION_PARAMS không hợp lệ.") # Hoặc dừng hẳn server
    else:
        _qdrant_cli = init_qdrant(  # init_qdrant là initialize_qdrant_and_collection
            connection_params=config.QDRANT_CONNECTION_PARAMS,  # << --- SỬA Ở ĐÂY ---
            collection_name=config.QDRANT_COLLECTION_NAME,
            vector_dimension=config.VECTOR_DIMENSION,
            recreate_collection=False,  # Quan trọng: không tạo lại collection khi API startup
        )

    if _qdrant_cli is None:  # init_qdrant sẽ trả về None nếu có lỗi
        print(
            "LỖI NGHIÊM TRỌNG (API Startup): Không thể khởi tạo hoặc kết nối Qdrant client / collection."
        )
        raise RuntimeError("Không thể khởi tạo Qdrant client.")
    print(
        f"DEBUG (dependencies.py): Qdrant client đã khởi tạo và collection '{config.QDRANT_COLLECTION_NAME}' đã được xác nhận."
    )

    print("\nDEBUG (dependencies.py): --- Bước 5: Khởi tạo Reranker ---")
    if hasattr(config, "RERANKER_ACTIVE") and config.RERANKER_ACTIVE:
        print(
            f"DEBUG (dependencies.py): Reranker được kích hoạt. Đang tải model '{config.RERANKER_MODEL_NAME}'..."
        )
        try:
            _reranker_instance = Reranker(config.RERANKER_MODEL_NAME)
            if _reranker_instance.model is None:
                print(
                    "CẢNH BÁO (API Startup): Không tải được model cho Reranker instance. Reranking sẽ không hoạt động."
                )
                _reranker_instance = None
            else:
                print(
                    "DEBUG (dependencies.py): Reranker instance đã tạo và model đã tải thành công."
                )
        except Exception as e_rerank:
            print(
                f"LỖI (API Startup): Không thể khởi tạo Reranker instance: {e_rerank}. Reranking sẽ bị tắt."
            )
            _reranker_instance = None
    else:
        print("DEBUG (dependencies.py): Reranker không được kích hoạt.")
        _reranker_instance = None
    print(
        f"DEBUG (dependencies.py): Trạng thái _reranker_instance: {'Đã khởi tạo với model' if _reranker_instance and _reranker_instance.model else ('Có instance nhưng không có model' if _reranker_instance else 'None')}"
    )

    print("\nAPI Startup: Khởi tạo tài nguyên dùng chung hoàn tất.")
    print("DEBUG (dependencies.py): Kết thúc hàm startup_event_handler().")
    print("DEBUG (dependencies.py): ==============================================\n")


# --- Các hàm "getter" để FastAPI inject dependencies (giữ nguyên) ---
def get_gemini_api_manager() -> GeminiApiKeyManager:
    if _gemini_api_manager is None:
        print(
            "LỖI RUNTIME (dependencies.py): Gọi get_gemini_api_manager nhưng _gemini_api_manager là None!"
        )
        raise RuntimeError(
            "Gemini API Manager chưa được khởi tạo. Lỗi cấu hình server nghiêm trọng."
        )
    return _gemini_api_manager


def get_mistral_client() -> Optional[Mistral]:
    if (
        _mistral_client is None
    ):  # Sửa: Nếu mistral client là None và được yêu cầu, có thể nên báo lỗi nếu nó là bắt buộc cho endpoint nào đó
        print(
            "CẢNH BÁO RUNTIME (dependencies.py): Gọi get_mistral_client nhưng _mistral_client là None."
        )
    return _mistral_client


def get_knowledge_graph() -> nx.DiGraph:
    if _knowledge_graph is None:
        print(
            "LỖI RUNTIME (dependencies.py): Gọi get_knowledge_graph nhưng _knowledge_graph là None!"
        )
        raise RuntimeError(
            "Knowledge Graph chưa được khởi tạo. Lỗi cấu hình server nghiêm trọng."
        )
    return _knowledge_graph


def get_qdrant_client() -> QdrantClient:
    if _qdrant_cli is None:
        print(
            "LỖI RUNTIME (dependencies.py): Gọi get_qdrant_client nhưng _qdrant_cli là None!"
        )
        raise RuntimeError(
            "Qdrant client chưa được khởi tạo. Lỗi cấu hình server nghiêm trọng."
        )
    return _qdrant_cli


def get_reranker() -> Optional[Reranker]:
    # Logic kiểm tra trong getter có thể không cần thiết nếu startup_event_handler đã xử lý kỹ
    # và các endpoint sử dụng reranker nên tự kiểm tra xem nó có None không trước khi dùng.
    return _reranker_instance

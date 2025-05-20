# chatbot_cli.py
import os
import google.generativeai as genai  # Cần cho khởi tạo model

# Import các thành phần từ config và các module trong src
import config
from src.utils.api_key_manager import GeminiApiKeyManager
from src.knowledge_graph.kg_loader_service import load_nx_graph_from_file
from src.vector_store.qdrant_service import initialize_qdrant_and_collection
from src.embedding.embed_querry import embed_query_gemini
from src.retrieval.retrieval_service import retrieve_and_compile_context
from src.llm.generation_service import generate_chatbot_response


def main_chatbot_application():
    print("Khởi tạo hệ thống Chatbot Ngân hàng RAG...")

    # 1. Khởi tạo API Key Manager cho Gemini
    try:
        gemini_api_manager = GeminiApiKeyManager(
            retry_wait_seconds=config.RETRY_WAIT_SECONDS,
            max_retries_per_key_cycle=config.MAX_RETRIES_PER_KEY_CYCLE,
            api_key_prefix=config.GEMINI_API_KEY_PREFIX,
        )
    except ValueError as e:
        print(e)
        exit()

    # 2. Tải Knowledge Graph
    kg = load_nx_graph_from_file(config.GRAPH_FILE_TO_LOAD)
    if kg is None:
        exit()

    # 3. Khởi tạo Qdrant Client
    qdrant_cli = initialize_qdrant_and_collection(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        collection_name=config.QDRANT_COLLECTION_NAME,
        vector_dimension=config.VECTOR_DIMENSION,
    )
    if qdrant_cli is None:
        exit()

    # 4. Khởi tạo Gemini Generation Model Object
    # Lưu ý: API key sẽ được set bởi GeminiApiKeyManager khi gọi API
    gemini_llm_for_generation = (
        None  # Không cần khởi tạo genai.GenerativeModel ở đây nữa nếu manager làm
    )
    # Tuy nhiên, generation_service có thể cần tên model
    # Hoặc ApiKeyManager có thể được điều chỉnh để nhận model object
    # Hiện tại, execute_generative_call trong manager của tôi tạo model object.
    print(f"Sử dụng model sinh văn bản Gemini: {config.GENERATION_MODEL_NAME}")

    print(f"\n--- Chào mừng bạn đến với Trợ lý ảo Ngân hàng Kienlongbank ---")
    print(f"Tôi có thể giúp gì cho bạn về các sản phẩm và dịch vụ của chúng tôi?")
    print("Nhập 'quit' hoặc 'exit' để thoát.")

    while True:
        user_input = input("\nBạn hỏi: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Cảm ơn bạn đã sử dụng Trợ lý ảo. Tạm biệt!")
            break
        if not user_input.strip():
            continue

        # Bước 1 Pipeline: Embedding câu hỏi
        query_vector = embed_query_gemini(
            user_query=user_input,
            api_manager=gemini_api_manager,
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            task_type=config.EMBEDDING_TASK_TYPE_QUERY,
        )
        if not query_vector:
            print(
                f"\nTrợ lý Kienlongbank: Xin lỗi, tôi không thể xử lý câu hỏi của bạn lúc này (lỗi embedding)."
            )
            continue

        # Bước 2 Pipeline: Truy xuất và tổng hợp ngữ cảnh
        compiled_context, context_parts_for_display = retrieve_and_compile_context(
            query_vector=query_vector,
            qdrant_cli=qdrant_cli,
            knowledge_graph=kg,
            qdrant_collection_name=config.QDRANT_COLLECTION_NAME,
            qdrant_search_limit=config.QDRANT_SEARCH_LIMIT,
        )

        # In ra ngữ cảnh (đã có trong hàm retrieve_and_compile_context,
        # nhưng nếu hàm đó không in thì bạn in ở đây)
        print("\n--- Thông tin được tìm thấy để cung cấp cho Trợ lý AI (Ngữ cảnh): ---")
        if not context_parts_for_display:
            print(
                "(Không có thông tin cụ thể nào được truy xuất từ cơ sở dữ liệu cho câu hỏi này.)"
            )
        else:
            for i, item in enumerate(context_parts_for_display):
                print(f"  Nguồn tham khảo {i+1}:")
                print(
                    f"    Tài liệu: '{item['source']}' (Loại: {item['type']}, Độ tương đồng: {item['score']:.4f})"
                )
                if "document_summary" in item:
                    print(f"    Tóm tắt tài liệu: {item['document_summary']}")
                if "document_keywords" in item:
                    print(f"    Từ khóa tài liệu: {item['document_keywords']}")
                print(f"    Nội dung trích dẫn (đầu): {item['content_snippet']}\n")
        print("--- Kết thúc thông tin tìm thấy ---\n")

        # Bước 3 Pipeline: Sinh câu trả lời
        answer = generate_chatbot_response(
            user_query=user_input,
            compiled_context=compiled_context,
            gemini_generation_model_name=config.GENERATION_MODEL_NAME,  # Truyền tên model
            api_manager=gemini_api_manager,
            bank_homepage_url=config.BANK_HOMEPAGE_URL,
            bank_contact_info=config.BANK_CONTACT_INFO,
            generation_prompt_guidelines=config.GENERATION_PROMPT_GUIDELINES,
        )

        print(f"\nTrợ lý Kienlongbank: {answer}")


if __name__ == "__main__":
    # Tạo các file __init__.py nếu chưa có
    project_root_for_init = os.path.dirname(os.path.abspath(__file__))
    packages_to_initialize = [
        "src",
        "src/embedding",
        "src/retrieval",
        "src/llm",
        "src/knowledge_graph",
        "src/vector_store",
        "src/utils",
        # Thêm các package khác trong src nếu bạn tạo
    ]
    for pkg_path_str in packages_to_initialize:
        path_parts = pkg_path_str.split("/")
        pkg_dir_abs = os.path.join(project_root_for_init, *path_parts)
        if not os.path.exists(pkg_dir_abs):
            os.makedirs(pkg_dir_abs, exist_ok=True)
        init_file = os.path.join(pkg_dir_abs, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass

    main_chatbot_application()

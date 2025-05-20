import os
import time
import uuid
import networkx as nx
from qdrant_client import models as qdrant_models  # Đặt alias để tránh trùng tên nếu có
import google.generativeai as genai  # Cần thiết cho version check

# Import các thành phần từ config và các module trong src
import config  # Từ thư mục gốc
from src.utils.api_key_manager import GeminiApiKeyManager
from src.knowledge_graph.kg_loader_service import load_nx_graph_from_file
from src.embedding.embedding_service import embed_texts_in_batches
from src.vector_store.qdrant_service import (
    initialize_qdrant_and_collection,
    upsert_data_to_qdrant,
)


def print_stage_header(stage_name):
    print(f"\n{'='*20} BẮT ĐẦU GIAI ĐOẠN: {stage_name} {'='*20}")


def print_stage_footer(stage_name):
    print(f"\n{'='*20} KẾT THÚC GIAI ĐOẠN: {stage_name} {'='*20}")


def extract_and_prepare_data_from_kg(knowledge_graph: nx.DiGraph) -> list:
    """
    Trích xuất văn bản và chuẩn bị metadata từ Knowledge Graph để embedding.
    """
    items_to_embed_with_metadata = []
    print("\nChuẩn bị dữ liệu từ Knowledge Graph để embedding...")

    if not knowledge_graph or knowledge_graph.number_of_nodes() == 0:
        print(
            "CẢNH BÁO: Knowledge Graph rỗng hoặc không được tải. Không có dữ liệu để chuẩn bị."
        )
        return items_to_embed_with_metadata

    for node_id, data in knowledge_graph.nodes(data=True):
        text_to_embed = None
        qdrant_id_for_point = str(uuid.uuid4())
        payload = {
            "graph_node_id": str(node_id),
            "node_type": data.get("type", "Unknown"),
            "original_text": "",
            "document_name": "",
        }

        node_type = data.get("type")
        if node_type == "Document":
            doc_name = data.get("name", "Không có tên tài liệu")
            summary = data.get("summary", "")
            # keywords đã là chuỗi string dạng "kw1, kw2, kw3" khi lưu vào graphml
            keywords_str = data.get("keywords", "")

            text_to_embed = (
                f"Tài liệu: {doc_name}.\nTừ khóa: {keywords_str}.\nTóm tắt: {summary}"
            )
            if not summary and not keywords_str:
                text_to_embed = f"Tài liệu: {doc_name}"

            payload["document_name"] = doc_name
            payload["summary"] = summary
            payload["keywords"] = (
                keywords_str.split(", ") if keywords_str else []
            )  # Chuyển lại thành list cho payload
            payload["original_text"] = text_to_embed.strip()

        elif node_type == "Chunk":
            text_to_embed = data.get("text_content", "")
            source_doc_id_from_kg = data.get("source_document_id", "")
            doc_name_from_ref = (
                source_doc_id_from_kg.replace("doc:", "")
                if source_doc_id_from_kg.startswith("doc:")
                else "Tài liệu không xác định"
            )

            payload["document_name"] = doc_name_from_ref
            payload["order_in_doc"] = data.get("order_in_doc", -1)
            payload["original_text"] = text_to_embed.strip()

        else:
            continue  # Bỏ qua các loại node không cần embedding

        if text_to_embed and text_to_embed.strip():
            items_to_embed_with_metadata.append(
                {
                    "qdrant_id": qdrant_id_for_point,
                    "text_to_embed": text_to_embed.strip(),
                    "payload": payload,
                }
            )

    print(f"Đã chuẩn bị {len(items_to_embed_with_metadata)} mục từ KG để embedding.")
    return items_to_embed_with_metadata


def run_embedding_and_indexing_pipeline(
    graph_file_path: str,
    api_manager: GeminiApiKeyManager,
    qdrant_connection_params: dict,
    qdrant_collection_name: str,
    vector_dimension: int,
    embedding_model: str,
    embedding_task_type: str,
    embedding_batch_size: int,
    recreate_qdrant_collection: bool = False,
):
    """
    Hàm chính điều phối việc tải KG, trích xuất text, embedding, và lưu vào Qdrant.
    """
    # 1. Khởi tạo Qdrant Client và Collection
    print_stage_header("KHỞI TẠO QDRANT")
    qdrant_cli = initialize_qdrant_and_collection(  # Từ src.vector_store.qdrant_service
        connection_params=qdrant_connection_params,
        collection_name=qdrant_collection_name,
        vector_dimension=vector_dimension,
        recreate_collection=recreate_qdrant_collection,
    )
    if qdrant_cli is None:
        print(
            "Không thể khởi tạo Qdrant client hoặc collection. Dừng pipeline embedding."
        )
        print_stage_footer("KHỞI TẠO QDRANT")
        return False
    print_stage_footer("KHỞI TẠO QDRANT")

    # 2. Tải Knowledge Graph
    print_stage_header("TẢI KNOWLEDGE GRAPH")
    knowledge_graph = load_nx_graph_from_file(
        graph_file_path
    )  # Từ src.knowledge_graph.kg_loader_service
    if knowledge_graph is None:
        print("Không tải được KG, dừng pipeline embedding.")
        print_stage_footer("TẢI KNOWLEDGE GRAPH")
        return False
    print_stage_footer("TẢI KNOWLEDGE GRAPH")

    # 3. Chuẩn bị dữ liệu từ KG để embedding
    print_stage_header("CHUẨN BỊ DỮ LIỆU TỪ KG")
    items_to_embed = extract_and_prepare_data_from_kg(knowledge_graph)
    if not items_to_embed:
        print("Không có dữ liệu nào từ KG để embedding.")
        print_stage_footer("CHUẨN BỊ DỮ LIỆU TỪ KG")
        # Coi như thành công nếu không có gì để embed, hoặc False nếu đây là lỗi
        return True
    print_stage_footer("CHUẨN BỊ DỮ LIỆU TỪ KG")

    # 4. Tạo Embeddings
    print_stage_header("TẠO VECTOR EMBEDDINGS (GEMINI)")
    texts_list = [item["text_to_embed"] for item in items_to_embed]

    embeddings_list = embed_texts_in_batches(  # Từ src.embedding.embedding_service
        texts_to_embed=texts_list,
        api_manager=api_manager,
        embedding_model_name=embedding_model,
        task_type=embedding_task_type,
        batch_size=embedding_batch_size,
    )

    if embeddings_list is None:
        print("LỖI NGHIÊM TRỌNG: Không thể tạo embeddings. Dừng quá trình.")
        print_stage_footer("TẠO VECTOR EMBEDDINGS (GEMINI)")
        return False

    valid_embeddings_count = sum(1 for emb in embeddings_list if emb is not None)
    print(
        f"Đã tạo thành công {valid_embeddings_count} / {len(items_to_embed)} embeddings."
    )

    if (
        valid_embeddings_count == 0 and len(items_to_embed) > 0
    ):  # Có item cần embed nhưng không embed được cái nào
        print("Không có embedding nào được tạo thành công. Dừng.")
        print_stage_footer("TẠO VECTOR EMBEDDINGS (GEMINI)")
        return False
    print_stage_footer("TẠO VECTOR EMBEDDINGS (GEMINI)")

    # 5. Chuẩn bị Points và Upsert vào Qdrant
    print_stage_header("UPSERT DỮ LIỆU VÀO QDRANT")
    points_for_qdrant = []
    for i, item_meta in enumerate(items_to_embed):
        # Đảm bảo rằng chúng ta chỉ lấy embedding nếu nó tồn tại và index không vượt quá
        if i < len(embeddings_list) and embeddings_list[i] is not None:
            points_for_qdrant.append(
                qdrant_models.PointStruct(
                    id=item_meta["qdrant_id"],
                    vector=embeddings_list[i],
                    payload=item_meta["payload"],
                )
            )

    if not points_for_qdrant:
        print(
            "Không có điểm dữ liệu hợp lệ nào (đã có embedding) để upsert vào Qdrant."
        )
        print_stage_footer("UPSERT DỮ LIỆU VÀO QDRANT")
        return True  # Coi như không có gì để làm, không phải lỗi

    upsert_status = upsert_data_to_qdrant(  # Từ src.vector_store.qdrant_service
        qdrant_cli,
        qdrant_collection_name,
        points_for_qdrant,
        batch_size=100,  # Có thể lấy từ config nếu muốn: config.QDRANT_UPSERT_BATCH_SIZE
    )
    print_stage_footer("UPSERT DỮ LIỆU VÀO QDRANT")
    return upsert_status


if __name__ == "__main__":
    # Tạo các file __init__.py (giữ nguyên từ các script trước)
    project_root_for_init = os.path.dirname(os.path.abspath(__file__))
    packages_to_initialize = [
        "src",
        "src/utils",
        "src/knowledge_graph",
        "src/embedding",
        "src/vector_store",
        "src/data_processing",
        "src/chunking",
        "src/document_store",
        "src/llm",
        "src/api",
        "src/api/endpoints",
        "src/reranking",
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

    print("Khởi tạo Pipeline Embedding Knowledge Graph vào Qdrant...")

    # 1. Khởi tạo API Key Manager
    try:
        gemini_api_manager = GeminiApiKeyManager(
            retry_wait_seconds=config.RETRY_WAIT_SECONDS,
            max_retries_per_key_cycle=config.MAX_RETRIES_PER_KEY_CYCLE,
            api_key_prefix=config.GEMINI_API_KEY_PREFIX,
        )
    except (
        ValueError
    ) as e:  # Lỗi này được ném từ __init__ của GeminiApiKeyManager nếu không có key
        print(f"LỖI KHỞI TẠO API MANAGER: {e}")
        exit()
    except (
        AttributeError
    ) as e_attr:  # Bắt lỗi nếu config thiếu các hằng số retry/prefix
        print(
            f"LỖI THIẾU CẤU HÌNH CHO API MANAGER: {e_attr}. Vui lòng kiểm tra config.py."
        )
        exit()

    # 2. Lấy các cấu hình từ config.py
    print("\n--- CẤU HÌNH PIPELINE EMBEDDING ---")
    print(f"File Knowledge Graph đầu vào: {config.GRAPH_FILE_TO_LOAD}")
    # In QDRANT_CONNECTION_PARAMS một cách an toàn (không lộ api_key)
    q_conn_params_display = {
        k: (v if k != "api_key" else "********")
        for k, v in config.QDRANT_CONNECTION_PARAMS.items()
    }
    print(f"Qdrant Connection Params: {q_conn_params_display}")
    print(f"Qdrant Collection: {config.QDRANT_COLLECTION_NAME}")
    print(f"Vector Dimension: {config.VECTOR_DIMENSION}")
    print(f"Gemini Embedding Model: {config.EMBEDDING_MODEL_NAME}")
    print(f"Embedding Task Type: {config.EMBEDDING_TASK_TYPE_QUERY}")
    print(f"Embedding Batch Size: {config.EMBEDDING_BATCH_SIZE}")
    print("----------------------------------\n")

    # 3. Xác nhận có muốn tạo lại collection Qdrant không
    recreate_qdrant_input = (
        input(
            f"Bạn có muốn XÓA và TẠO LẠI collection '{config.QDRANT_COLLECTION_NAME}' không? "
            "Điều này sẽ xóa hết dữ liệu cũ. (yes/no, mặc định là no): "
        )
        .strip()
        .lower()
    )
    should_recreate_qdrant_collection = (
        True if recreate_qdrant_input in ["yes", "y"] else False
    )
    if should_recreate_qdrant_collection:
        print(
            f"LƯU Ý: Collection '{config.QDRANT_COLLECTION_NAME}' SẼ BỊ XÓA VÀ TẠO LẠI."
        )

    confirmation = (
        input("Bạn có muốn tiếp tục với các cài đặt trên? (yes/no): ").strip().lower()
    )
    if confirmation == "yes" or confirmation == "y":
        print("\nBắt đầu quá trình embedding và lưu trữ vào Qdrant...")

        success = run_embedding_and_indexing_pipeline(
            graph_file_path=config.GRAPH_FILE_TO_LOAD,
            api_manager=gemini_api_manager,
            qdrant_connection_params=config.QDRANT_CONNECTION_PARAMS,
            qdrant_collection_name=config.QDRANT_COLLECTION_NAME,
            vector_dimension=config.VECTOR_DIMENSION,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            embedding_task_type=config.EMBEDDING_TASK_TYPE_QUERY,
            embedding_batch_size=config.EMBEDDING_BATCH_SIZE,
            recreate_qdrant_collection=should_recreate_qdrant_collection,
        )
        if success:
            print(
                "\n--- PIPELINE EMBEDDING VÀ INDEXING VÀO QDRANT ĐÃ HOÀN THÀNH THÀNH CÔNG! ---"
            )
        else:
            print(
                "\n--- PIPELINE EMBEDDING VÀ INDEXING VÀO QDRANT ĐÃ HOÀN THÀNH VỚI MỘT SỐ LỖI HOẶC KHÔNG CÓ GÌ ĐỂ LÀM. ---"
            )
    else:
        print("Đã hủy quá trình.")

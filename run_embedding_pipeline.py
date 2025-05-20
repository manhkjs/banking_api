# run_embedding_pipeline.py
import os
import time
import uuid
import networkx as nx
from qdrant_client import models  # Cho PointStruct

# Import từ các module trong src và config.py từ thư mục gốc
import config
from src.utils.api_key_manager import GeminiApiKeyManager
from src.knowledge_graph.kg_loader_service import load_nx_graph_from_file
from src.embedding.embedding_service import embed_texts_in_batches
from src.vector_store.qdrant_service import (
    initialize_qdrant_and_collection,
    upsert_data_to_qdrant,
)


def run_embedding_and_indexing_pipeline(
    graph_file_path: str,
    api_manager: GeminiApiKeyManager,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_collection_name: str,
    vector_dimension: int,
    embedding_model: str,
    embedding_task_type: str,
    embedding_batch_size: int,
    recreate_qdrant_collection: bool = False,  # Thêm tùy chọn để tạo lại collection
):
    """
    Hàm chính điều phối việc tải KG, trích xuất text, embedding, và lưu vào Qdrant.
    """
    # 1. Khởi tạo Qdrant Client và Collection
    qdrant_cli = initialize_qdrant_and_collection(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=qdrant_collection_name,
        vector_dimension=vector_dimension,
        recreate_collection=recreate_qdrant_collection,
    )
    if qdrant_cli is None:
        print(
            "Không thể khởi tạo Qdrant client hoặc collection. Dừng pipeline embedding."
        )
        return False

    # 2. Tải Knowledge Graph
    knowledge_graph = load_nx_graph_from_file(graph_file_path)
    if knowledge_graph is None:
        print("Không tải được KG, dừng pipeline embedding.")
        return False

    # 3. Chuẩn bị dữ liệu từ KG để embedding
    items_to_embed_with_metadata = []
    print("\nChuẩn bị dữ liệu từ Knowledge Graph để embedding...")
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
            doc_name = data.get("name", "Không có tên")
            summary = data.get("summary", "")
            keywords_str = data.get("keywords", "")  # Đã là chuỗi, ví dụ "kw1, kw2"

            text_to_embed = (
                f"Tài liệu: {doc_name}.\nTừ khóa: {keywords_str}.\nTóm tắt: {summary}"
            )
            if not summary and not keywords_str:
                text_to_embed = f"Tài liệu: {doc_name}"

            payload["document_name"] = doc_name
            payload["summary"] = summary
            payload["keywords"] = (
                keywords_str.split(", ")
                if keywords_str and isinstance(keywords_str, str)
                else []
            )
            payload["original_text"] = (
                text_to_embed  # Lưu text đã được cấu trúc để embed
            )

        elif node_type == "Chunk":
            text_to_embed = data.get("text_content", "")
            source_doc_id = data.get("source_document_id", "")  # Ví dụ: "doc:ten_file"
            doc_name_from_ref = (
                source_doc_id.replace("doc:", "")
                if source_doc_id.startswith("doc:")
                else "Không rõ tài liệu"
            )

            payload["document_name"] = doc_name_from_ref
            payload["order_in_doc"] = data.get("order_in_doc", -1)
            payload["original_text"] = text_to_embed  # Lưu text gốc của chunk

        else:  # Bỏ qua các loại node không xác định hoặc không cần embedding
            continue

        if text_to_embed and text_to_embed.strip():
            items_to_embed_with_metadata.append(
                {
                    "qdrant_id": qdrant_id_for_point,
                    "text_to_embed": text_to_embed.strip(),
                    "payload": payload,
                }
            )

    if not items_to_embed_with_metadata:
        print("Không tìm thấy nội dung văn bản nào trong KG để embedding.")
        return False

    print(f"Tổng số {len(items_to_embed_with_metadata)} mục sẽ được embedding.")

    # 4. Tạo Embeddings
    texts_for_embedding_list = [
        item["text_to_embed"] for item in items_to_embed_with_metadata
    ]
    print(f"\nBắt đầu tạo embeddings cho {len(texts_for_embedding_list)} mục...")

    embeddings_list = embed_texts_in_batches(
        texts_to_embed=texts_for_embedding_list,
        api_manager=api_manager,
        embedding_model_name=embedding_model,
        task_type=embedding_task_type,
        batch_size=embedding_batch_size,
    )

    if (
        embeddings_list is None
    ):  # embed_texts_in_batches trả về None nếu có lỗi nghiêm trọng
        print("LỖI NGHIÊM TRỌNG: Không thể tạo embeddings. Dừng quá trình.")
        return False

    # Lọc ra những item có embedding thành công
    valid_points_data = []
    if len(embeddings_list) == len(items_to_embed_with_metadata):
        for i, item_meta in enumerate(items_to_embed_with_metadata):
            if embeddings_list[i] is not None:  # Chỉ thêm nếu có embedding
                valid_points_data.append(
                    models.PointStruct(
                        id=item_meta["qdrant_id"],
                        vector=embeddings_list[i],
                        payload=item_meta["payload"],
                    )
                )
        successful_embeddings_count = len(valid_points_data)
        print(
            f"Đã tạo thành công {successful_embeddings_count} / {len(items_to_embed_with_metadata)} embeddings."
        )
    else:
        print(
            f"LỖI: Số lượng embedding trả về ({len(embeddings_list)}) không khớp số lượng text ({len(items_to_embed_with_metadata)})."
        )
        return False

    if not valid_points_data:
        print("Không có embedding hợp lệ nào được tạo để lưu vào Qdrant.")
        return False

    # 5. Upsert vào Qdrant
    return upsert_data_to_qdrant(
        qdrant_cli, q_collection, valid_points_data, batch_size=100
    )


if __name__ == "__main__":
    # Tạo các file __init__.py nếu chưa có (giữ nguyên từ các script trước)
    project_root_for_init = os.path.dirname(os.path.abspath(__file__))
    packages_to_initialize = [
        "src",
        "src/knowledge_graph",
        "src/embedding",
        "src/vector_store",
        "src/utils",
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
    except ValueError as e:
        print(e)
        exit()

    # 2. Xác nhận có muốn tạo lại collection Qdrant không
    recreate_qdrant = (
        input(
            f"Bạn có muốn XÓA và TẠO LẠI collection '{config.QDRANT_COLLECTION_NAME}' không? "
            "Điều này sẽ xóa hết dữ liệu cũ trong collection đó. (yes/no, mặc định là no): "
        )
        .strip()
        .lower()
    )
    should_recreate_collection = (
        True if recreate_qdrant == "yes" or recreate_qdrant == "y" else False
    )
    if should_recreate_collection:
        print(
            f"LƯU Ý: Collection '{config.QDRANT_COLLECTION_NAME}' sẽ được xóa và tạo lại."
        )

    # 3. Lấy các cấu hình từ config.py
    graph_path = config.GRAPH_FILE_TO_LOAD
    q_host = config.QDRANT_HOST
    q_port = config.QDRANT_PORT
    q_collection = config.QDRANT_COLLECTION_NAME
    vec_dim = config.VECTOR_DIMENSION
    embed_model = config.EMBEDDING_MODEL_NAME
    embed_task = config.EMBEDDING_TASK_TYPE_QUERY
    embed_batch_size = config.EMBEDDING_BATCH_SIZE

    print(f"\nChuẩn bị thực hiện embedding cho KG từ '{graph_path}'")
    print(f"Sẽ lưu vào Qdrant collection '{q_collection}' tại {q_host}:{q_port}.")
    print(f"Model embedding: {embed_model}, Kích thước vector: {vec_dim}")

    confirmation = (
        input("Bạn có muốn tiếp tục với các cài đặt trên? (yes/no): ").strip().lower()
    )
    if confirmation == "yes" or confirmation == "y":
        print("\nBắt đầu quá trình embedding và lưu trữ...")

        success = run_embedding_and_indexing_pipeline(
            graph_file_path=graph_path,
            api_manager=gemini_api_manager,
            qdrant_host=q_host,
            qdrant_port=q_port,
            qdrant_collection_name=q_collection,
            vector_dimension=vec_dim,
            embedding_model=embed_model,
            embedding_task_type=embed_task,
            embedding_batch_size=embed_batch_size,
            recreate_qdrant_collection=should_recreate_collection,
        )
        if success:
            print(
                "\nPipeline embedding và indexing vào Qdrant đã hoàn thành thành công!"
            )
        else:
            print(
                "\nPipeline embedding và indexing vào Qdrant đã hoàn thành với một số lỗi."
            )
    else:
        print("Đã hủy quá trình.")

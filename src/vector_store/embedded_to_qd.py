import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import networkx as nx
from qdrant_client import QdrantClient, models  # Thư viện Qdrant
import uuid  # Để tạo ID duy nhất cho Qdrant nếu cần

# --- CẤU HÌNH CHUNG ---
# Model Embedding
EMBEDDING_MODEL_NAME = "models/embedding-001"  # Model embedding của Gemini
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"  # Hoặc "SEMANTIC_SIMILARITY" tùy mục đích
VECTOR_DIMENSION = 768  # Kích thước vector của model embedding-001

# Qdrant
QDRANT_HOST = "localhost"  # Hoặc URL của Qdrant Cloud
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "knowledge_graph_embeddings_v3"

# File Knowledge Graph
GRAPH_FILE_TO_LOAD = "E:/kienlong/banking_ai_platform/src/data_processing/document_knowledge_graph_1.graphml"

# Gemini API Keys & Retry Logic
RETRY_WAIT_SECONDS = 60
MAX_RETRIES_PER_KEY_CYCLE = 2
EMBEDDING_BATCH_SIZE = 32  # Số lượng text gửi đi embed cùng lúc (Gemini có giới hạn)


# --- HÀM TẢI API KEYS (Giữ nguyên) ---
def load_api_keys():
    """Tải tất cả các GEMINI_API_KEY_X từ file .env."""
    load_dotenv()
    api_keys = []
    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            api_keys.append(key)
            i += 1
        else:
            break
    if not api_keys:
        print("LỖI: Không tìm thấy GEMINI_API_KEY nào trong file .env.")
    return api_keys


# --- HÀM TẢI KNOWLEDGE GRAPH (Giữ nguyên) ---
def load_knowledge_graph(graph_file_path):
    if not os.path.exists(graph_file_path):
        print(f"LỖI: File đồ thị '{graph_file_path}' không tìm thấy.")
        return None
    try:
        print(f"Đang tải đồ thị từ file: {graph_file_path}...")
        graph = nx.read_graphml(graph_file_path)
        print("Tải đồ thị thành công.")
        return graph
    except Exception as e:
        print(f"Lỗi khi tải đồ thị từ file '{graph_file_path}': {e}")
        return None


# --- HÀM TẠO EMBEDDING BẰNG GEMINI (THEO BATCH) ---
def embed_texts_with_gemini_batch(
    texts_to_embed,
    api_keys_list,
    current_key_index_ref,
    total_keys_exhausted_cycles_ref,
    task_type=EMBEDDING_TASK_TYPE,
):
    """
    Tạo embeddings cho một danh sách các đoạn văn bản sử dụng Gemini API, xử lý theo batch.
    Quản lý API key và quota.
    """
    if not api_keys_list:
        print("LỖI: Không có API key cho embedding.")
        return None
    if not texts_to_embed:
        return []

    all_embeddings = []

    for i in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
        batch_texts = texts_to_embed[i : i + EMBEDDING_BATCH_SIZE]
        print(
            f"    Đang tạo embedding cho batch {i//EMBEDDING_BATCH_SIZE + 1}/{(len(texts_to_embed) -1)//EMBEDDING_BATCH_SIZE + 1} ({len(batch_texts)} items)..."
        )

        # Vòng lặp thử key và đợi
        batch_embedded_successfully = False
        while not batch_embedded_successfully:
            current_api_key = api_keys_list[current_key_index_ref[0]]
            print(
                f"      Sử dụng API Key #{current_key_index_ref[0] + 1} cho embedding..."
            )
            try:
                genai.configure(api_key=current_api_key)
                # Gemini API cho phép gửi một list of strings để embed trong một request
                result = genai.embed_content(
                    model=EMBEDDING_MODEL_NAME,
                    content=batch_texts,  # Gửi cả batch
                    task_type=task_type,
                )
                batch_embeddings = (
                    result["embedding"]
                    if isinstance(result["embedding"], list)
                    else [result["embedding"]]
                )  # Đảm bảo là list

                # Kiểm tra xem số lượng embedding trả về có khớp không
                if len(batch_embeddings) == len(batch_texts):
                    all_embeddings.extend(batch_embeddings)
                    print(
                        f"      Embedding batch thành công với Key #{current_key_index_ref[0] + 1}."
                    )
                    total_keys_exhausted_cycles_ref[0] = 0  # Reset khi thành công
                    batch_embedded_successfully = True
                else:
                    # Lỗi không mong muốn: số lượng embedding không khớp
                    print(
                        f"      Lỗi: Số lượng embedding trả về ({len(batch_embeddings)}) không khớp số lượng text đầu vào ({len(batch_texts)})."
                    )
                    # Xử lý như lỗi, chuyển key
                    raise Exception("Mismatch in embeddings count")

            except (
                genai.types.generation_types.BlockedPromptException,
                genai.types.generation_types.StopCandidateException,
            ) as e:
                print(
                    f"      Cảnh báo/Lỗi (Blocked/Stop) với Key #{current_key_index_ref[0] + 1} (embedding): {type(e).__name__} - {e}"
                )
                # (Logic xoay vòng key và đợi như cũ)
                current_key_index_ref[0] = (current_key_index_ref[0] + 1) % len(
                    api_keys_list
                )
                if current_key_index_ref[0] == 0:
                    total_keys_exhausted_cycles_ref[0] += 1
                    print(
                        f"      Tất cả API key cho embedding đã được thử {total_keys_exhausted_cycles_ref[0]} lần và gặp vấn đề."
                    )
                    if total_keys_exhausted_cycles_ref[0] >= MAX_RETRIES_PER_KEY_CYCLE:
                        print(
                            f"      Đã thử {MAX_RETRIES_PER_KEY_CYCLE} chu kỳ. Không thể embed batch này."
                        )
                        return None  # Không thể embed batch này
                    print(f"      Đang đợi {RETRY_WAIT_SECONDS} giây...")
                    time.sleep(RETRY_WAIT_SECONDS)
            except Exception as e:
                if (
                    "429" in str(e)
                    or "resource_exhausted" in str(e).lower()
                    or "quota" in str(e).lower()
                ):
                    print(
                        f"      Lỗi Quota với Key #{current_key_index_ref[0] + 1} (embedding). Chi tiết: {e}"
                    )
                else:
                    print(
                        f"      Lỗi không xác định với Key #{current_key_index_ref[0] + 1} (embedding): {type(e).__name__} - {e}"
                    )

                current_key_index_ref[0] = (current_key_index_ref[0] + 1) % len(
                    api_keys_list
                )
                if current_key_index_ref[0] == 0:
                    total_keys_exhausted_cycles_ref[0] += 1
                    print(
                        f"      Tất cả API key cho embedding đã gặp lỗi/quota {total_keys_exhausted_cycles_ref[0]} lần."
                    )
                    if total_keys_exhausted_cycles_ref[0] >= MAX_RETRIES_PER_KEY_CYCLE:
                        print(
                            f"      Đã thử {MAX_RETRIES_PER_KEY_CYCLE} chu kỳ. Không thể embed batch này."
                        )
                        return None  # Không thể embed batch này
                    print(f"      Đang đợi {RETRY_WAIT_SECONDS} giây...")
                    time.sleep(RETRY_WAIT_SECONDS)
            time.sleep(
                0.5
            )  # Thêm sleep nhỏ giữa các batch API call để tránh quá tải nhẹ
    return all_embeddings


# --- HÀM CHÍNH ĐỂ EMBED KG VÀ LƯU VÀO QDRANT ---
def embed_kg_and_store_in_qdrant(graph, qdrant_client, collection_name, api_keys):
    """
    Trích xuất text từ KG, tạo embeddings, và lưu vào Qdrant.
    """
    if not api_keys:
        print("LỖI: Không có API key cho embedding.")
        return
    if graph is None:
        print("LỖI: Knowledge Graph rỗng, không có gì để embed.")
        return

    items_to_embed_with_metadata = []

    print("/nChuẩn bị dữ liệu từ Knowledge Graph để embedding...")
    # 1. Xử lý Document Nodes
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "Document":
            doc_name = data.get("name", "Không có tên")
            summary = data.get("summary", "")
            keywords_str = data.get("keywords", "")

            doc_representative_text = (
                f"Tài liệu: {doc_name}./nTừ khóa: {keywords_str}./nTóm tắt: {summary}"
            )
            if not summary and not keywords_str:
                doc_representative_text = f"Tài liệu: {doc_name}"

            if doc_representative_text.strip():
                # << --- SỬA ĐỔI ID CHO QDRANT --- >>
                # qdrant_point_id = f"doc_summary:{doc_name}" # ID CŨ GÂY LỖI
                qdrant_point_id = str(uuid.uuid4())  # TẠO UUID MỚI

                payload = {
                    "graph_node_id": node_id,  # Vẫn lưu ID gốc của graph node trong payload
                    "document_name": doc_name,
                    "node_type": "DocumentSummary",
                    "original_text": doc_representative_text,
                    "summary": summary,
                    "keywords": keywords_str.split(", ") if keywords_str else [],
                }
                items_to_embed_with_metadata.append(
                    {
                        "id": qdrant_point_id,  # ID mới cho Qdrant
                        "text_to_embed": doc_representative_text,
                        "payload": payload,
                    }
                )

    # 2. Xử lý Chunk Nodes
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "Chunk":
            chunk_text = data.get("text_content", "")
            doc_id_ref = data.get("source_document_id", "")
            doc_name_from_ref = (
                doc_id_ref.replace("doc:", "")
                if doc_id_ref.startswith("doc:")
                else "Không rõ tài liệu"
            )

            if chunk_text.strip():
                # << --- SỬA ĐỔI ID CHO QDRANT --- >>
                # qdrant_point_id = node_id # ID CŨ GÂY LỖI (vì nó là chuỗi như "chunk:doc_name_0")
                qdrant_point_id = str(uuid.uuid4())  # TẠO UUID MỚI

                payload = {
                    "graph_node_id": node_id,  # Vẫn lưu ID gốc của graph node trong payload
                    "document_name": doc_name_from_ref,
                    "node_type": "Chunk",
                    "original_text": chunk_text,
                    "order_in_doc": data.get("order_in_doc", -1),
                }
                items_to_embed_with_metadata.append(
                    {
                        "id": qdrant_point_id,  # ID mới cho Qdrant
                        "text_to_embed": chunk_text,
                        "payload": payload,
                    }
                )

    if not items_to_embed_with_metadata:
        print("Không tìm thấy nội dung văn bản nào trong KG để embedding.")
        return

    print(f"Tổng số {len(items_to_embed_with_metadata)} mục sẽ được embedding.")

    # ... (Phần tạo embeddings giữ nguyên) ...
    texts_for_embedding = [
        item["text_to_embed"] for item in items_to_embed_with_metadata
    ]
    current_key_idx_ref = [0]
    total_exhausted_cycles_ref = [0]
    print("/nBắt đầu tạo embeddings...")
    embeddings = embed_texts_with_gemini_batch(
        texts_for_embedding, api_keys, current_key_idx_ref, total_exhausted_cycles_ref
    )

    if embeddings is None or len(embeddings) != len(items_to_embed_with_metadata):
        print(
            "LỖI: Không thể tạo embeddings hoặc số lượng không khớp. Dừng quá trình lưu vào Qdrant."
        )
        return
    print(f"Đã tạo thành công {len(embeddings)} embeddings.")

    # Chuẩn bị points để upsert vào Qdrant
    points_to_upsert = []
    for i, item_meta in enumerate(items_to_embed_with_metadata):
        points_to_upsert.append(
            models.PointStruct(
                id=item_meta["id"],  # Sử dụng ID đã tạo (UUID hoặc số nguyên)
                vector=embeddings[i],
                payload=item_meta["payload"],
            )
        )

    # ... (Phần upsert vào Qdrant giữ nguyên) ...
    print(
        f"/nBắt đầu upsert {len(points_to_upsert)} điểm vào Qdrant collection '{collection_name}'..."
    )
    qdrant_batch_size = 100
    for i in range(0, len(points_to_upsert), qdrant_batch_size):
        batch_points = points_to_upsert[i : i + qdrant_batch_size]
        try:
            qdrant_client.upsert(
                collection_name=collection_name, points=batch_points, wait=True
            )
            print(
                f"  Đã upsert batch {i//qdrant_batch_size + 1}/{(len(points_to_upsert)-1)//qdrant_batch_size + 1} ({len(batch_points)} điểm) vào Qdrant."
            )
        except Exception as e_qdrant:
            print(f"  Lỗi khi upsert batch vào Qdrant: {e_qdrant}")
        time.sleep(0.1)

    print(f"Hoàn thành việc upsert dữ liệu vào Qdrant collection '{collection_name}'.")
    try:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        print(
            f"Thông tin collection '{collection_name}': Số điểm hiện tại = {collection_info.points_count}"
        )
    except Exception as e_info:
        print(f"Không thể lấy thông tin collection sau khi upsert: {e_info}")


# --- KHỐI THỰC THI CHÍNH ---
if __name__ == "__main__":
    # 1. Tải API keys
    gemini_api_keys = load_api_keys()
    if not gemini_api_keys:
        print("Không tìm thấy API key nào của Gemini. Dừng chương trình.")
        exit()
    print(f"Đã tải {len(gemini_api_keys)} API key của Gemini.")
    print(f"Phiên bản google-generativeai: {genai.__version__}")

    # 2. Tải Knowledge Graph
    knowledge_graph = load_knowledge_graph(GRAPH_FILE_TO_LOAD)
    if knowledge_graph is None:
        print(
            f"Không thể tải Knowledge Graph từ file '{GRAPH_FILE_TO_LOAD}'. Dừng chương trình."
        )
        exit()
    print(
        f"Knowledge Graph có {knowledge_graph.number_of_nodes()} nút và {knowledge_graph.number_of_edges()} cạnh."
    )

    # 3. Khởi tạo Qdrant Client và Collection
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"Đã kết nối tới Qdrant tại {QDRANT_HOST}:{QDRANT_PORT}.")

        # Kiểm tra xem collection đã tồn tại chưa
        try:
            collection_info = qdrant_client.get_collection(
                collection_name=QDRANT_COLLECTION_NAME
            )
            print(f"Collection '{QDRANT_COLLECTION_NAME}' đã tồn tại.")
            print(f"  Kích thước vector: {collection_info.config.params.vectors.size}")
            print(
                f"  Distance metric: {collection_info.config.params.vectors.distance}"
            )
            if collection_info.config.params.vectors.size != VECTOR_DIMENSION:
                print(
                    f"CẢNH BÁO: Kích thước vector của collection hiện tại ({collection_info.config.params.vectors.size}) "
                    f"không khớp với kích thước vector embedding mong muốn ({VECTOR_DIMENSION})."
                )
                # Bạn có thể quyết định xóa và tạo lại collection, hoặc dừng lại.
                # Ví dụ: qdrant_client.delete_collection(QDRANT_COLLECTION_NAME)
                #        print(f"Đã xóa collection cũ.")
                #        raise ValueError("Cần tạo lại collection với kích thước vector đúng.")
        except Exception:  # Thường là lỗi "Not found: Collection `...` doesn't exist!"
            print(
                f"Collection '{QDRANT_COLLECTION_NAME}' chưa tồn tại. Đang tạo mới..."
            )
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION, distance=models.Distance.COSINE
                ),
            )
            print(
                f"Đã tạo collection '{QDRANT_COLLECTION_NAME}' với kích thước vector {VECTOR_DIMENSION} và distance Cosine."
            )

    except Exception as e_q_init:
        print(f"LỖI: Không thể kết nối hoặc tạo collection trong Qdrant: {e_q_init}")
        exit()

    # 4. Xác nhận trước khi chạy
    print(
        f"/nChuẩn bị thực hiện embedding cho KG và lưu vào Qdrant collection '{QDRANT_COLLECTION_NAME}'."
    )
    confirmation = input("Bạn có muốn tiếp tục? (yes/no): ").strip().lower()
    if confirmation == "yes" or confirmation == "y":
        print("/nBắt đầu quá trình embedding và lưu trữ...")
        embed_kg_and_store_in_qdrant(
            knowledge_graph, qdrant_client, QDRANT_COLLECTION_NAME, gemini_api_keys
        )
    else:
        print("Đã hủy quá trình.")

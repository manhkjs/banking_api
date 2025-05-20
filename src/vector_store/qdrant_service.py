# src/vector_store/qdrant_service.py
from qdrant_client import (
    QdrantClient,
    models,
)  # Đảm bảo models được import từ qdrant_client
import time


def initialize_qdrant_and_collection(
    connection_params: dict,  # <<---- THAM SỐ ĐÚNG LÀ ĐÂY
    collection_name: str,
    vector_dimension: int,
    distance_metric=models.Distance.COSINE,  # distance_metric từ models của qdrant_client
    recreate_collection: bool = False,
) -> QdrantClient | None:
    """
    Khởi tạo Qdrant client và đảm bảo collection tồn tại với cấu hình đúng.
    connection_params là một dict chứa 'url' (và 'api_key' nếu có) cho cloud,
    hoặc 'host' và 'port' cho local.
    """
    if not connection_params or not (
        connection_params.get("url") or connection_params.get("host")
    ):
        print(
            "LỖI (Qdrant Service): Tham số kết nối Qdrant (connection_params) không hợp lệ hoặc thiếu url/host."
        )
        return None

    try:
        # Thêm timeout mặc định nếu chưa có trong connection_params
        final_connection_params = connection_params.copy()
        if "timeout" not in final_connection_params:
            final_connection_params["timeout"] = 20  # Giây

        if final_connection_params.get("url"):
            # Che api_key khi in ra log
            display_params = {
                k: (v if k != "api_key" else "********")
                for k, v in final_connection_params.items()
            }
            print(
                f"Thông tin (Qdrant Service): Đang thử kết nối tới Qdrant với params: {display_params}"
            )
        else:  # Local connection
            print(
                f"Thông tin (Qdrant Service): Đang thử kết nối tới Qdrant local tại {final_connection_params.get('host')}:{final_connection_params.get('port')}..."
            )

        client = QdrantClient(**final_connection_params)  # Sử dụng ** để unpack dict

        # Kiểm tra collection
        existing_collections_response = client.get_collections()
        existing_collection_names = [
            col.name for col in existing_collections_response.collections
        ]
        collection_exists = collection_name in existing_collection_names

        if collection_exists and recreate_collection:
            print(
                f"Collection '{collection_name}' tồn tại và recreate_collection=True. Đang xóa collection cũ..."
            )
            client.delete_collection(collection_name=collection_name)
            print(f"Đã xóa collection '{collection_name}'.")
            collection_exists = False  # Để tạo lại ở bước sau

        if collection_exists:
            collection_info = client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' đã tồn tại.")

            current_vector_size = None
            vectors_cfg = collection_info.config.params.vectors

            if isinstance(vectors_cfg, models.VectorParams):
                current_vector_size = vectors_cfg.size
            elif isinstance(vectors_cfg, dict):  # Dành cho named vectors
                if "" in vectors_cfg and isinstance(
                    vectors_cfg[""], models.VectorParams
                ):  # Vector mặc định không tên
                    current_vector_size = vectors_cfg[""].size
                elif vectors_cfg:  # Nếu không có vector mặc định, lấy cái đầu tiên
                    first_vector_name = list(vectors_cfg.keys())[0]
                    if isinstance(vectors_cfg[first_vector_name], models.VectorParams):
                        current_vector_size = vectors_cfg[first_vector_name].size

            if current_vector_size is not None:
                print(
                    f"  Kích thước vector hiện tại của collection: {current_vector_size}"
                )
                if current_vector_size != vector_dimension:
                    print(
                        f"CẢNH BÁO NGHIÊM TRỌNG: Kích thước vector của collection '{collection_name}' ({current_vector_size}) "
                        f"KHÔNG KHỚP với kích thước vector embedding mong muốn ({vector_dimension})."
                    )
                    print(
                        "Vui lòng xóa collection thủ công, đặt recreate_collection=True, hoặc cập nhật VECTOR_DIMENSION."
                    )
                    return None
            else:
                print(
                    f"CẢNH BÁO: Không xác định được kích thước vector cho collection '{collection_name}'. Cấu trúc: {vectors_cfg}"
                )

        else:  # Collection không tồn tại hoặc đã bị xóa để tạo lại
            print(
                f"Collection '{collection_name}' chưa tồn tại hoặc được yêu cầu tạo lại. Đang tạo mới..."
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension, distance=distance_metric
                ),
            )
            print(
                f"Đã tạo collection '{collection_name}' với kích thước vector {vector_dimension} và distance {distance_metric}."
            )

        # Kiểm tra lại một lần nữa sau khi có thể đã tạo
        client.get_collection(collection_name=collection_name)
        print(f"Kết nối và thiết lập collection '{collection_name}' thành công.")
        return client

    except Exception as e:
        print(
            f"LỖI trong initialize_qdrant_and_collection cho '{collection_name}': {type(e).__name__} - {e}"
        )
        # import traceback; traceback.print_exc() # Bật để debug nếu cần thiết
        return None


def upsert_data_to_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str,
    points_to_upsert: list[
        models.PointStruct
    ],  # Sử dụng models.PointStruct từ qdrant_client
    batch_size: int = 100,
):
    """
    Upsert một danh sách các PointStruct vào Qdrant theo batch.
    """
    if not points_to_upsert:
        print("Không có điểm dữ liệu nào để upsert vào Qdrant.")
        return False

    print(
        f"Chuẩn bị upsert {len(points_to_upsert)} điểm vào Qdrant collection '{collection_name}'..."
    )
    success_count = 0
    total_points = len(points_to_upsert)
    num_batches = (total_points - 1) // batch_size + 1 if total_points > 0 else 0

    for i in range(0, total_points, batch_size):
        batch_points = points_to_upsert[i : i + batch_size]
        current_batch_num = i // batch_size + 1
        print(
            f"  Đang upsert batch {current_batch_num}/{num_batches} ({len(batch_points)} điểm)..."
        )
        try:
            qdrant_client.upsert(
                collection_name=collection_name, points=batch_points, wait=True
            )
            success_count += len(batch_points)
        except Exception as e_qdrant_upsert:
            print(
                f"    Lỗi khi upsert batch {current_batch_num} vào Qdrant: {e_qdrant_upsert}"
            )
        if num_batches > 1 and current_batch_num < num_batches:
            time.sleep(0.1)

    print(
        f"Hoàn thành việc upsert dữ liệu. {success_count}/{total_points} điểm đã được thử upsert."
    )
    try:
        collection_info_after = qdrant_client.get_collection(
            collection_name=collection_name
        )
        print(
            f"Thông tin collection '{collection_name}': Số điểm hiện tại = {collection_info_after.points_count}"
        )
    except Exception as e_info:
        print(f"Không thể lấy thông tin collection sau khi upsert: {e_info}")
    return success_count == total_points


def search_qdrant_collection(  # Hàm này dùng cho chatbot, không phải cho pipeline embedding
    qdrant_client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int,
) -> list:
    """
    Thực hiện tìm kiếm vector trong Qdrant collection.
    """
    try:
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        return search_results
    except Exception as e_q_search:
        print(
            f"    Lỗi khi tìm kiếm trên Qdrant collection '{collection_name}': {e_q_search}"
        )
        return []

# src/document_store/mongodb_service.py
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError
from datetime import datetime, timezone
import time

# Biến client và db toàn cục cho module
_mongo_client_instance = None
_mongo_database_instance = None


def connect_to_mongodb(
    mongo_uri: str, db_name: str, max_connection_tries=3
) -> MongoClient | None:  # Trả về db instance
    """
    Thiết lập kết nối đến MongoDB bằng URI (đã bao gồm credentials) và ServerApi.
    Chọn database được chỉ định. Trả về database instance.
    """
    global _mongo_client_instance, _mongo_database_instance

    if mongo_uri is None:  # Kiểm tra mongo_uri trước
        print("LỖI (MongoDB): Chuỗi kết nối MONGO_URI không được cung cấp hoặc rỗng.")
        return None

    if _mongo_client_instance and _mongo_database_instance:
        try:
            _mongo_client_instance.admin.command("ping")
            return _mongo_database_instance
        except ConnectionFailure:
            print(
                "Thông tin (MongoDB): Kết nối MongoDB bị mất. Đang cố gắng kết nối lại..."
            )
            _mongo_client_instance = None
            _mongo_database_instance = None

    current_tries = 0
    while current_tries < max_connection_tries:
        current_tries += 1
        try:
            print(
                f"Thông tin (MongoDB): Đang kết nối tới MongoDB (lần thử {current_tries}/{max_connection_tries})..."
            )
            client = MongoClient(
                mongo_uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=10000
            )
            client.admin.command("ping")
            print("Pinged your deployment. You successfully connected to MongoDB!")
            _mongo_client_instance = client
            _mongo_database_instance = _mongo_client_instance[db_name]
            print(
                f"Thông tin (MongoDB): Kết nối MongoDB thành công. Đã chọn database: '{_mongo_database_instance.name}'."
            )
            return _mongo_database_instance
        except (ConnectionFailure, ConfigurationError) as e:
            print(
                f"LỖI (MongoDB): Không thể kết nối tới MongoDB server (lần thử {current_tries}). Lỗi: {e}"
            )
            if current_tries >= max_connection_tries:
                print("LỖI (MongoDB): Đã thử kết nối tối đa số lần. Bỏ cuộc.")
                return None
            print("Thông tin (MongoDB): Sẽ thử lại sau 5 giây...")
            time.sleep(5)
        except Exception as e_unhandled:
            print(
                f"LỖI (MongoDB): Lỗi không xác định khi kết nối MongoDB: {e_unhandled}"
            )
            return None
    return None


def get_mongodb_collection(db_instance, collection_name: str):
    """Lấy một collection cụ thể từ database instance đã kết nối."""
    if db_instance is None:  # SỬA Ở ĐÂY
        print(
            "LỖI (MongoDB): Database instance không hợp lệ (chưa kết nối thành công?)."
        )
        return None
    try:
        return db_instance[collection_name]
    except Exception as e:
        print(
            f"LỖI (MongoDB): Không thể truy cập collection '{collection_name}'. Lỗi: {e}"
        )
        return None


def save_or_update_processed_document(
    mongo_collection,  # Đây là đối tượng Collection
    document_id: str,
    original_pdf_filename: str,
    processed_markdown_content: str,
    version: int = 1,
    additional_metadata: dict = None,
):
    if mongo_collection is None:  # SỬA Ở ĐÂY
        print("LỖI (MongoDB): Đối tượng collection không hợp lệ (None).")
        return False

    doc_data = {
        "original_pdf_filename": original_pdf_filename,
        "processed_markdown_content": processed_markdown_content,
        "last_updated_mongodb": datetime.now(timezone.utc),
        "version": version,
    }
    if additional_metadata and isinstance(additional_metadata, dict):
        doc_data.update(additional_metadata)
    try:
        result = mongo_collection.update_one(
            {"_id": document_id}, {"$set": doc_data}, upsert=True
        )
        if result.upserted_id or result.modified_count > 0 or result.matched_count > 0:
            return True
        else:
            # Trường hợp này có thể xảy ra nếu $set không làm thay đổi gì (dữ liệu y hệt)
            # và không có upsert (nhưng ở đây có upsert=True nên ít xảy ra nếu không có lỗi)
            print(
                f"CẢNH BÁO (MongoDB): Lệnh update_one không có hiệu lực cho tài liệu '{document_id}' (có thể nội dung không đổi). Matched: {result.matched_count}, Modified: {result.modified_count}"
            )
            return True  # Vẫn coi là thành công nếu không có lỗi
    except OperationFailure as e_op:
        print(
            f"LỖI OperationFailure (MongoDB): khi lưu/cập nhật '{document_id}': {e_op.details}"
        )
    except Exception as e_unhandled:
        print(
            f"LỖI không xác định (MongoDB): khi lưu/cập nhật '{document_id}': {e_unhandled}"
        )
    return False


def get_processed_document(mongo_collection, document_id: str) -> dict | None:
    if mongo_collection is None:  # SỬA Ở ĐÂY
        print("LỖI (MongoDB): Đối tượng collection không hợp lệ (None).")
        return None
    try:
        document = mongo_collection.find_one({"_id": document_id})
        return document
    except Exception as e:
        print(f"LỖI (MongoDB): khi truy xuất tài liệu '{document_id}': {e}")
        return None

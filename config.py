import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(DOTENV_PATH):
    print(f"DEBUG (config.py): Đang tải .env từ: {DOTENV_PATH}")
    load_dotenv(dotenv_path=DOTENV_PATH)
else:
    print(
        f"CẢNH BÁO (config.py): File .env không tìm thấy tại {DOTENV_PATH}. Các biến môi trường cần được set thủ công nếu không có sẵn."
    )
# --- CÁC CẤU HÌNH KHÁC GIỮ NGUYÊN ---
# API Keys (Tên biến môi trường)
MISTRAL_API_KEY_ENV_VAR = "MISTRAL_API_KEY"
GEMINI_API_KEY_PREFIX = "GEMINI_API_KEY_"
VECTOR_DIMENSION = int(
    os.getenv("VECTOR_DIMENSION_ENV", "768")
)  # Đọc từ .env, mặc định 768

# --- CẤU HÌNH QDRANT (CHỈ CLOUD) ---
QDRANT_CLOUD_URI_ENV_VAR = "QDRANT_CLOUD_URI"
QDRANT_API_KEY_ENV_VAR = "QDRANT_API_KEY"
QDRANT_COLLECTION_NAME_ENV_VAR = "QDRANT_COLLECTION_NAME_ENV"

QDRANT_CLOUD_URI = os.getenv(QDRANT_CLOUD_URI_ENV_VAR)
QDRANT_API_KEY = os.getenv(QDRANT_API_KEY_ENV_VAR)  # Có thể là None

QDRANT_COLLECTION_NAME = os.getenv(
    QDRANT_COLLECTION_NAME_ENV_VAR, "knowledge_graph_embeddings_v2"
)

QDRANT_CONNECTION_PARAMS = {}
if QDRANT_CLOUD_URI:
    QDRANT_CONNECTION_PARAMS["url"] = QDRANT_CLOUD_URI
    if QDRANT_API_KEY:
        QDRANT_CONNECTION_PARAMS["api_key"] = QDRANT_API_KEY
    # Qdrant Cloud thường dùng port 6333 cho REST/HTTP và 6334 cho gRPC.
    # Nếu URI của bạn chưa có port, và client bạn dùng cần nó, bạn có thể thêm vào đây.
    # Ví dụ: if not (":6333" in QDRANT_CLOUD_URI or ":6334" in QDRANT_CLOUD_URI):
    #            QDRANT_CONNECTION_PARAMS['port'] = 6333 # Hoặc 6334 tùy client
    print(
        f"Thông tin (config.py): Qdrant sẽ kết nối bằng Cloud URI: {QDRANT_CLOUD_URI} (API key {'đã' if QDRANT_API_KEY else 'chưa/không'} cấu hình)."
    )
else:
    print(
        f"LỖI (config.py): QDRANT_CLOUD_URI CHƯA được thiết lập trong file .env. Không thể cấu hình kết nối Qdrant Cloud."
    )

VECTOR_DIMENSION = 768
MONGO_USERNAME_ENV_VAR = "MONGO_USERNAME"
MONGO_PASSWORD_ENV_VAR = "MONGO_PASSWORD"
MONGO_CLUSTER_ADDRESS_ENV_VAR = (
    "MONGO_CLUSTER_ADDRESS"  # Ví dụ: "cluster0.dxaqaqr.mongodb.net"
)
MONGO_DB_NAME_ENV_VAR = "MONGO_DATABASE_NAME"  # Tên database bạn muốn dùng
MONGO_PROCESSED_DOCS_COLLECTION_ENV_VAR = (
    "MONGO_PROCESSED_DOCS_COLLECTION"  # Tên collection
)

# Đọc các giá trị từ biến môi trường
_mongo_user = os.getenv(MONGO_USERNAME_ENV_VAR)
_mongo_pass = os.getenv(MONGO_PASSWORD_ENV_VAR)
_mongo_cluster = os.getenv(MONGO_CLUSTER_ADDRESS_ENV_VAR)

# Xây dựng MONGO_CONNECTION_URI từ các thành phần
MONGO_CONNECTION_URI = None  # Khởi tạo là None
if _mongo_user and _mongo_pass and _mongo_cluster:
    # URI này dành cho MongoDB Atlas, có appName
    MONGO_CONNECTION_URI = f"mongodb+srv://{_mongo_user}:{_mongo_pass}@{_mongo_cluster}/?retryWrites=true&w=majority&appName=Cluster0"
    print(
        f"DEBUG (config.py): MONGO_CONNECTION_URI đã được xây dựng (thông tin nhạy cảm không được in ra)."
    )
elif os.getenv(
    "MONGO_CONNECTION_URI"
):  # Cho phép đặt MONGO_CONNECTION_URI trực tiếp trong .env
    MONGO_CONNECTION_URI = os.getenv("MONGO_CONNECTION_URI")
    print(
        f"DEBUG (config.py): MONGO_CONNECTION_URI được tải trực tiếp từ biến môi trường."
    )
else:
    print(
        "CẢNH BÁO (config.py): Không đủ thông tin (USERNAME, PASSWORD, CLUSTER_ADDRESS) để xây dựng MONGO_CONNECTION_URI và cũng không có biến MONGO_CONNECTION_URI trực tiếp trong .env."
    )
    print("Kết nối MongoDB có thể thất bại hoặc sẽ sử dụng URI mặc định nếu có.")
    # MONGO_CONNECTION_URI = "mongodb://localhost:27017/" # Fallback cho local nếu muốn (không khuyến khích cho Atlas)

MONGO_DATABASE_NAME = os.getenv(
    MONGO_DB_NAME_ENV_VAR, "KienlongbankRAG_DefaultDB"
)  # <<< ĐÂY LÀ BIẾN SCRIPT ĐANG DÙNG
MONGO_PROCESSED_DOCS_COLLECTION_NAME = os.getenv(
    MONGO_PROCESSED_DOCS_COLLECTION_ENV_VAR, "ProcessedMarkdown_DefaultCol"
)
# Models
MISTRAL_OCR_MODEL_NAME = "mistral-ocr-latest"
GEMINI_TEXT_REFINER_MODEL = "gemini-1.5-flash-latest"
GEMINI_CHUNK_DELIMITER_MODEL = "gemini-1.5-flash-latest"
GEMINI_SUMMARIZATION_MODEL = "gemini-2.0-flash"
GENERATION_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/embedding-001"
EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
# Task type khi embedding câu hỏi của người dùng để tìm kiếm (dùng trong chatbot)
EMBEDDING_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
# Số lượng đoạn văn bản gửi cho Gemini API trong một lượt gọi embedding
EMBEDDING_BATCH_SIZE = 32
# ... (các model khác nếu có)
MAX_DOC_CHARS_FOR_SUMMARY = 200000
BANK_HOMEPAGE_URL = "https://kienlongbank.com"
BANK_CONTACT_INFO = "Tổng đài 19006929 hoặc chi nhánh Kienlongbank gần nhất"
# --- CẤU HÌNH ĐƯỜNG DẪN ---
PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)  # Giả sử config.py ở thư mục gốc
BASE_DATA_PIPELINE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data")

# Thư mục INPUT cho toàn bộ pipeline xử lý Markdown bằng Gemini
# Đây có thể là output của giai đoạn OCR, hoặc một thư mục chứa file .md thô ban đầu của bạn.
# Đảm bảo rằng bạn đã có file .md trong thư mục này.
INPUT_OCR_MD_FOLDER = os.path.join(
    BASE_DATA_PIPELINE_OUTPUT_PATH, "02_mistral_ocr_md_output"
)  # HOẶC THƯ MỤC MD THÔ CỦA BẠN

# Thư mục OUTPUT của Giai đoạn 1 (Sửa lỗi chính tả & MD cơ bản) / INPUT của Giai đoạn 2
REFINED_MD_OUTPUT_FOLDER = os.path.join(
    BASE_DATA_PIPELINE_OUTPUT_PATH, "03a_gemini_refined_md"
)  # <<< ĐẢM BẢO DÒNG NÀY CÓ

# Thư mục OUTPUT của Giai đoạn 2 (Chuẩn hóa Heading & Chèn Delimiter) / INPUT của Giai đoạn 3 (KG Builder)
CHUNKED_MD_OUTPUT_FOLDER = os.path.join(
    BASE_DATA_PIPELINE_OUTPUT_PATH, "03b_gemini_chunked_md"
)  # <<< ĐẢM BẢO DÒNG NÀY CÓ

# Thư mục và file OUTPUT cho Knowledge Graph (Giai đoạn 3)
KG_OUTPUT_DIR = os.path.join(BASE_DATA_PIPELINE_OUTPUT_PATH, "04_knowledge_graph")
KG_GRAPHML_OUTPUT_FILE = os.path.join(
    KG_OUTPUT_DIR, "document_knowledge_graph.graphml"
)  # Sử dụng tên file bạn đã dùng

# --- CÁC HẰNG SỐ KHÁC ---
CHUNK_DELIMITER = "\n\n---CHUNK_DELIMITER---\n\n"
DOCUMENT_LANGUAGE = "tiếng Việt, English"
MAX_INPUT_CHARS_GEMINI = 280000
RETRY_WAIT_SECONDS = 60
MAX_RETRIES_PER_KEY_CYCLE = 2
OUTPUT_FILE_SUFFIX_AFTER_GEMINI_PROCESSING = (
    "_processed.md"  # Đã dùng cho output của GĐ2 (Chunked)
)

# In ra để kiểm tra khi config.py được import (tùy chọn)
print(f"DEBUG (config.py): INPUT_OCR_MD_FOLDER = {INPUT_OCR_MD_FOLDER}")
print(f"DEBUG (config.py): REFINED_MD_OUTPUT_FOLDER = {REFINED_MD_OUTPUT_FOLDER}")
print(f"DEBUG (config.py): CHUNKED_MD_OUTPUT_FOLDER = {CHUNKED_MD_OUTPUT_FOLDER}")
print(f"DEBUG (config.py): KG_GRAPHML_OUTPUT_FILE = {KG_GRAPHML_OUTPUT_FILE}")

# GRAPH_FILE_TO_LOAD = os.path.join(_PROJECT_ROOT, "src", "data_processing", "document_knowledge_graph.graphml")
# Sửa lại đường dẫn cho phù hợp với vị trí file graphml của bạn. Ví dụ:
GRAPH_FILE_TO_LOAD = os.path.join(KG_OUTPUT_DIR, "document_knowledge_graph.graphml")

# GRAPH_FILE_TO_LOAD = "E:/kienlong/banking_ai_platform/src/data_processing/document_knowledge_graph.graphml"

# Gemini API Keys & Retry Logic
RETRY_WAIT_SECONDS = 60
MAX_RETRIES_PER_KEY_CYCLE = 2
# API_CALL_TIMEOUT_SECONDS = 120 # Hiện tại chưa dùng trong safe_gemini_api_call

# Retrieval
QDRANT_SEARCH_LIMIT = 5  # Giảm từ 5 xuống 3 để context gọn hơn, có thể tùy chỉnh
RERANKER_ACTIVE = True  # Đặt thành False để tắt reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5  # Lấy top 3 sau khi rerank (phải <= QDRANT_SEARCH_LIMIT)
GENERATION_PROMPT_GUIDELINES = f"""
1.  **Phong cách giao tiếp:** Luôn trả lời như một nhân viên ngân hàng đang tư vấn trực tiếp: tự nhiên, thân thiện, gần gũi và chuyên nghiệp. Tuyệt đối không sử dụng các cụm từ máy móc như "dựa trên tài liệu tham khảo...", "thông tin truy xuất được cho thấy...", "trong ngữ cảnh được cung cấp...". Hãy diễn giải thông tin một cách tự nhiên.

2.  **Xử lý câu hỏi của khách hàng một cách linh hoạt:**

    * **Khi "THÔNG TIN THAM KHẢO TỪ CƠ SỞ DỮ LIỆU NỘI BỘ" có ích:**
        * Nếu "CÂU HỎI CỦA NGƯỜI DÙNG" liên quan đến sản phẩm, dịch vụ, quy định, điều khoản cụ thể của Kienlongbank VÀ "THÔNG TIN THAM KHẢO TỪ CƠ SỞ DỮ LIỆU NỘI BỘ" cung cấp thông tin liên quan và hữu ích, hãy **ưu tiên sử dụng thông tin này** để xây dựng câu trả lời. Hãy tổng hợp, diễn giải thông tin đó một cách dễ hiểu, không chỉ đơn thuần là lặp lại.
        * Nếu thông tin tham khảo có vẻ liên quan nhưng chưa đủ chi tiết để trả lời trọn vẹn câu hỏi cụ thể đó, bạn có thể đề cập ngắn gọn những gì bạn tìm thấy và sau đó lịch sự sử dụng "Hướng dẫn Fallback" ở mục 3.

    * **Khi "THÔNG TIN THAM KHẢO TỪ CƠ SỞ DỮ LIỆU NỘI BỘ" không có ích hoặc câu hỏi mang tính chất chung:**
        * Nếu "CÂU HỎI CỦA NGƯỜI DÙNG" là về kiến thức ngân hàng phổ thông (ví dụ: "thẻ tín dụng là gì?", "làm sao để tiết kiệm tiền hiệu quả?"), các câu hỏi giao tiếp thông thường (chào hỏi, cảm ơn, hỏi thăm), hoặc nếu "THÔNG TIN THAM KHẢO TỪ CƠ SỞ DỮ LIỆU NỘI BỘ" trống hoặc không liên quan, hãy tự tin trả lời dựa trên kiến thức chung của bạn như một chuyên viên ngân hàng hiểu biết. Hãy giải thích rõ ràng, đưa ra lời khuyên hữu ích nếu phù hợp.

3.  **"Hướng dẫn Fallback" (Khi thông tin cụ thể về Kienlongbank không có sẵn hoặc bạn không chắc chắn):**
    * Nếu "CÂU HỎI CỦA NGƯỜI DÙNG" đòi hỏi thông tin rất cụ thể về chính sách, sản phẩm đặc thù, quy trình nội bộ của Kienlongbank MÀ "THÔNG TIN THAM KHẢO TỪ CƠ SỞ DỮ LIỆU NỘI BỘ" không cung cấp (hoặc không đủ chi tiết) VÀ kiến thức chung của bạn cũng không thể trả lời một cách chính xác và đầy đủ, thì hãy sử dụng câu trả lời sau:
        "Rất tiếc, hiện tại tôi chưa thể cung cấp thông tin chính xác và đầy đủ nhất về nội dung này. Để được hỗ trợ cụ thể và cập nhật, bạn vui lòng truy cập trang web chính thức của Kienlongbank tại {BANK_HOMEPAGE_URL} hoặc liên hệ trực tiếp với chúng tôi qua {BANK_CONTACT_INFO} nhé!"

4.  **Tóm lại:** Hãy hành xử như một chuyên viên tư vấn giỏi. Nếu có thông tin cụ thể từ hệ thống, hãy diễn giải nó. Nếu không, hãy dùng kiến thức chung của bạn cho các câu hỏi phù hợp. Đối với các yêu cầu thông tin chuyên sâu, cụ thể về Kienlongbank mà không có sẵn, hãy hướng dẫn khách hàng đến các kênh chính thức. Luôn giữ thái độ sẵn sàng giúp đỡ.
"""

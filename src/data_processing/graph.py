import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import networkx as nx  # Thư viện để làm việc với đồ thị
import json  # Để xử lý output JSON từ Gemini

# --- CẤU HÌNH CHUNG ---
MODEL_NAME_FOR_SUMMARIZATION = "gemini-1.5-flash-latest"  # Model cho tóm tắt và keyword
CHUNK_DELIMITER = "\n\n---CHUNK_DELIMITER---\n\n"
RETRY_WAIT_SECONDS = 60
MAX_RETRIES_PER_KEY_CYCLE = 2
GRAPH_OUTPUT_FILE = "document_knowledge_graph_1.graphml"


# --- HÀM TẢI API KEYS (Giữ nguyên từ script trước) ---
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


# --- HÀM LẤY SUMMARY VÀ KEYWORDS TỪ GEMINI ---
def get_document_summary_keywords_gemini(
    document_content,
    api_keys_list,
    current_key_index_ref,
    total_keys_exhausted_cycles_ref,
):
    """
    Sử dụng Gemini để tạo tóm tắt và trích xuất keywords cho nội dung tài liệu.

    Args:
        document_content (str): Toàn bộ nội dung của tài liệu.
        api_keys_list, current_key_index_ref, total_keys_exhausted_cycles_ref: Quản lý API key.

    Returns:
        dict: {"summary": "...", "keywords": ["kw1", "kw2", ...]} hoặc None nếu thất bại.
    """
    if not api_keys_list:
        return None

    prompt = f"""
Bạn là một trợ lý AI chuyên tóm tắt tài liệu và trích xuất từ khóa.
Cho nội dung tài liệu dưới đây (định dạng Markdown):

--- DOCUMENT CONTENT START ---
{document_content[:200000]} 
--- DOCUMENT CONTENT END ---
(Lưu ý: Nội dung trên có thể đã được cắt bớt nếu quá dài. Hãy tóm tắt dựa trên phần được cung cấp.)

Hãy thực hiện các yêu cầu sau và trả về kết quả DUY NHẤT dưới dạng một đối tượng JSON:
1.  **summary:** Tạo một bản tóm tắt ngắn gọn (khoảng 3-5 câu) nêu bật nội dung chính và mục đích của toàn bộ tài liệu.
2.  **keywords:** Liệt kê từ 5 đến 7 từ khóa hoặc cụm từ khóa quan trọng nhất mô tả chủ đề và các khía cạnh chính của tài liệu. Trả về dưới dạng một danh sách các chuỗi.

Ví dụ định dạng JSON output:
{{
  "summary": "Đây là bản tóm tắt nội dung chính của tài liệu, nêu bật các điểm quan trọng A, B, và C, nhằm mục đích D.",
  "keywords": ["từ khóa 1", "chủ đề chính A", "khái niệm B", "thuật ngữ C", "từ khóa 5"]
}}

Chỉ trả về đối tượng JSON hợp lệ, không thêm bất kỳ giải thích hay văn bản nào khác.
"""
    # Giới hạn độ dài đầu vào để tránh lỗi token limit của Gemini Flash
    # Gemini 1.5 Flash có context window lớn, nhưng vẫn nên cẩn thận
    # Nếu dùng Pro thì có thể tăng giới hạn này lên
    # Nếu document_content quá dài, API có thể bị timeout hoặc lỗi
    # Cần chiến lược xử lý tài liệu dài tốt hơn nếu đây là trường hợp thường xuyên

    while True:
        current_api_key = api_keys_list[current_key_index_ref[0]]
        print(
            f"    Đang sử dụng API Key #{current_key_index_ref[0] + 1} cho tóm tắt/keywords..."
        )
        try:
            genai.configure(api_key=current_api_key)
            model = genai.GenerativeModel(MODEL_NAME_FOR_SUMMARIZATION)
            generation_config = genai.types.GenerationConfig(temperature=0.2)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if response.parts:
                response_text = response.text
                # Cố gắng loại bỏ ```json và ``` nếu có
                if response_text.strip().startswith("```json"):
                    response_text = response_text.strip()[7:]
                if response_text.strip().endswith("```"):
                    response_text = response_text.strip()[:-3]

                try:
                    data = json.loads(response_text.strip())
                    if "summary" in data and "keywords" in data:
                        print(
                            f"    Tóm tắt/keywords thành công với Key #{current_key_index_ref[0] + 1}."
                        )
                        total_keys_exhausted_cycles_ref[0] = 0
                        return data
                    else:
                        print(
                            f"    Lỗi: JSON response từ Gemini không chứa 'summary' hoặc 'keywords'. Response: {data}"
                        )
                        raise genai.types.generation_types.BlockedPromptException(
                            "JSON không đúng định dạng"
                        )
                except json.JSONDecodeError as e_json:
                    print(
                        f"    Lỗi giải mã JSON từ Gemini (Key #{current_key_index_ref[0] + 1}): {e_json}"
                    )
                    print(f"    Response text nhận được: {response_text}")
                    raise genai.types.generation_types.BlockedPromptException(
                        "Lỗi giải mã JSON"
                    )
            else:
                error_message = f"    Lỗi: Response tóm tắt/keywords từ Gemini không có nội dung (Key #{current_key_index_ref[0] + 1})."
                if response.prompt_feedback:
                    error_message += f" Prompt Feedback: {response.prompt_feedback}"
                print(error_message)
                raise genai.types.generation_types.BlockedPromptException(
                    "Response trống"
                )

        except (
            genai.types.generation_types.BlockedPromptException,
            genai.types.generation_types.StopCandidateException,
        ) as e:
            print(
                f"    Cảnh báo/Lỗi (Blocked/Stop) với Key #{current_key_index_ref[0] + 1} (tóm tắt/keywords): {type(e).__name__} - {e}"
            )
            current_key_index_ref[0] = (current_key_index_ref[0] + 1) % len(
                api_keys_list
            )
            if current_key_index_ref[0] == 0:
                total_keys_exhausted_cycles_ref[0] += 1
                print(
                    f"    Tất cả API key cho tóm tắt/keywords đã được thử {total_keys_exhausted_cycles_ref[0]} lần và gặp vấn đề."
                )
                if total_keys_exhausted_cycles_ref[0] >= MAX_RETRIES_PER_KEY_CYCLE:
                    print(
                        f"    Đã thử {MAX_RETRIES_PER_KEY_CYCLE} chu kỳ. Bỏ qua tóm tắt/keywords cho file này."
                    )
                    return {
                        "summary": "Không thể tạo tóm tắt tự động.",
                        "keywords": [],
                    }  # Trả về giá trị mặc định
                print(f"    Đang đợi {RETRY_WAIT_SECONDS} giây...")
                time.sleep(RETRY_WAIT_SECONDS)
        except Exception as e:
            if (
                "429" in str(e)
                or "resource_exhausted" in str(e).lower()
                or "quota" in str(e).lower()
            ):
                print(
                    f"    Lỗi Quota với Key #{current_key_index_ref[0] + 1} (tóm tắt/keywords). Chi tiết: {e}"
                )
            else:
                print(
                    f"    Lỗi không xác định với Key #{current_key_index_ref[0] + 1} (tóm tắt/keywords): {type(e).__name__} - {e}"
                )
            current_key_index_ref[0] = (current_key_index_ref[0] + 1) % len(
                api_keys_list
            )
            if current_key_index_ref[0] == 0:
                total_keys_exhausted_cycles_ref[0] += 1
                print(
                    f"    Tất cả API key cho tóm tắt/keywords đã gặp lỗi/quota {total_keys_exhausted_cycles_ref[0]} lần."
                )
                if total_keys_exhausted_cycles_ref[0] >= MAX_RETRIES_PER_KEY_CYCLE:
                    print(
                        f"    Đã thử {MAX_RETRIES_PER_KEY_CYCLE} chu kỳ. Bỏ qua tóm tắt/keywords cho file này."
                    )
                    return {
                        "summary": "Không thể tạo tóm tắt tự động.",
                        "keywords": [],
                    }  # Trả về giá trị mặc định
                print(f"    Đang đợi {RETRY_WAIT_SECONDS} giây...")
                time.sleep(RETRY_WAIT_SECONDS)


# --- HÀM XÂY DỰNG KNOWLEDGE GRAPH ---
def build_document_knowledge_graph(processed_md_folder, api_keys):
    """
    Xây dựng Knowledge Graph từ các file .md đã được xử lý (chứa delimiter chunk).
    Sử dụng Gemini để tạo summary và keywords cho mỗi tài liệu.
    """
    if not api_keys:
        print("Không có API key để thực hiện. Dừng chương trình.")
        return None

    if not os.path.isdir(processed_md_folder):
        print(f"LỖI: Thư mục đầu vào '{processed_md_folder}' không tồn tại.")
        return None

    G = nx.DiGraph()  # Tạo đồ thị có hướng

    current_key_idx_ref = [0]
    total_exhausted_cycles_ref = [0]

    print(f"\nBắt đầu xây dựng Knowledge Graph từ thư mục: {processed_md_folder}")

    file_count = 0
    for filename in os.listdir(processed_md_folder):
        # Giả sử file output từ bước trước có dạng ten_file_processed.md
        if filename.endswith(
            ("_processed.md", ".chunked.md", ".md")
        ):  # Mở rộng để chấp nhận các đuôi file có thể
            file_count += 1
            file_path = os.path.join(processed_md_folder, filename)
            # Lấy tên tài liệu gốc, loại bỏ phần suffix nếu có
            doc_name = (
                filename.replace("_processed.md", "")
                .replace(".chunked.md", "")
                .replace(".md", "")
            )

            print(f"\nĐang xử lý tài liệu: {filename} (Tài liệu #{file_count})")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    full_document_content = f.read()

                if not full_document_content.strip():
                    print(f"  Cảnh báo: File '{filename}' rỗng. Bỏ qua.")
                    continue

                # 1. Tạo Summary và Keywords cho toàn bộ tài liệu bằng Gemini
                print(f"  Đang tạo tóm tắt và keywords cho '{doc_name}'...")
                doc_info = get_document_summary_keywords_gemini(
                    full_document_content,
                    api_keys,
                    current_key_idx_ref,
                    total_exhausted_cycles_ref,
                )

                summary = "Không có tóm tắt."
                keywords = []
                if doc_info:
                    summary = doc_info.get("summary", "Lỗi khi tạo tóm tắt.")
                    keywords = doc_info.get("keywords", [])

                # 2. Tạo Node Gốc (Document Node)
                doc_node_id = f"doc:{doc_name}"
                G.add_node(
                    doc_node_id,
                    type="Document",
                    name=doc_name,
                    original_filename=filename,
                    summary=summary,
                    keywords=(
                        ", ".join(keywords) if keywords else ""
                    ),  # Lưu keywords dạng chuỗi
                    # full_text=full_document_content # Tùy chọn: lưu toàn bộ text vào node
                )
                print(
                    f"  Đã tạo Document Node: {doc_node_id} (Summary: '{summary[:50]}...', Keywords: {keywords})"
                )

                # 3. Tách tài liệu thành các Chunks
                text_chunks = full_document_content.split(CHUNK_DELIMITER)

                previous_chunk_node_id = None
                for i, chunk_text_content in enumerate(text_chunks):
                    clean_chunk_text = chunk_text_content.strip()
                    if clean_chunk_text:
                        chunk_node_id = f"chunk:{doc_name}_{i}"
                        G.add_node(
                            chunk_node_id,
                            type="Chunk",
                            text_content=clean_chunk_text,
                            order_in_doc=i,
                            source_document_id=doc_node_id,
                        )

                        # Tạo cạnh HAS_CHUNK
                        G.add_edge(doc_node_id, chunk_node_id, type="HAS_CHUNK")

                        # Tạo cạnh NEXT_CHUNK
                        if previous_chunk_node_id:
                            G.add_edge(
                                previous_chunk_node_id, chunk_node_id, type="NEXT_CHUNK"
                            )
                        previous_chunk_node_id = chunk_node_id
                print(f"  Đã tạo và liên kết {len(text_chunks)} chunks cho {doc_name}.")

            except Exception as e_file:
                print(f"  Lỗi nghiêm trọng khi xử lý file '{filename}': {e_file}")
                import traceback

                traceback.print_exc()

    print("\n--- HOÀN THÀNH XÂY DỰNG KNOWLEDGE GRAPH (trong bộ nhớ) ---")
    if G.number_of_nodes() > 0:
        print(f"Tổng số nút trong đồ thị: {G.number_of_nodes()}")
        print(f"Tổng số cạnh trong đồ thị: {G.number_of_edges()}")

        # 4. Lưu đồ thị ra file (ví dụ: GraphML)
        try:
            nx.write_graphml(G, GRAPH_OUTPUT_FILE)
            print(f"Đồ thị đã được lưu vào file: {GRAPH_OUTPUT_FILE}")
            print(f"Bạn có thể mở file này bằng các công cụ như Gephi để xem.")
        except Exception as e_save:
            print(f"Lỗi khi lưu đồ thị ra file {GRAPH_OUTPUT_FILE}: {e_save}")
    else:
        print(
            "Không có nút nào được tạo trong đồ thị. Vui lòng kiểm tra lại file đầu vào và log."
        )

    return G


# --- KHỐI THỰC THI CHÍNH ---
if __name__ == "__main__":
    # --- CẤU HÌNH ---
    # Thư mục chứa các file .md đã được Gemini xử lý (sửa lỗi, chuẩn hóa, chèn delimiter chunk)
    input_processed_md_folder = "E:/kienlong/banking_ai_platform/data/processed_data/gemini_processed_markdown_v2"

    # --- KẾT THÚC CẤU HÌNH ---

    gemini_api_keys = load_api_keys()
    if not gemini_api_keys:
        print("Không tìm thấy API key nào của Gemini. Dừng chương trình.")
        exit()

    print(f"Đã tải {len(gemini_api_keys)} API key của Gemini.")
    print(f"Phiên bản google-generativeai: {genai.__version__}")

    valid_config = True
    if not input_processed_md_folder:
        print("LỖI: 'input_processed_md_folder' chưa được cấu hình (trống).")
        valid_config = False
    elif not os.path.isdir(input_processed_md_folder):
        print(
            f"LỖI: Đường dẫn đầu vào '{input_processed_md_folder}' không phải là một thư mục hoặc không tồn tại."
        )
        valid_config = False

    if valid_config:
        print(f"Thư mục file .md đã xử lý đầu vào: {input_processed_md_folder}")
        print(f"File đồ thị sẽ được lưu tại: {GRAPH_OUTPUT_FILE}")
        print(
            f"Sử dụng mô hình Gemini cho tóm tắt/keywords: {MODEL_NAME_FOR_SUMMARIZATION}"
        )

        confirmation = (
            input("Bạn có muốn tiếp tục với các cài đặt trên? (yes/no): ")
            .strip()
            .lower()
        )
        if confirmation == "yes" or confirmation == "y":
            print("\nBắt đầu quá trình xây dựng Knowledge Graph...")
            knowledge_graph = build_document_knowledge_graph(
                input_processed_md_folder, gemini_api_keys
            )
            # Bạn có thể làm gì đó với knowledge_graph ở đây nếu cần
        else:
            print("Đã hủy quá trình xây dựng Knowledge Graph.")
    else:
        print("Vui lòng kiểm tra lại cấu hình đường dẫn.")

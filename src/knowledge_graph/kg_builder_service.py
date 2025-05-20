# src/knowledge_graph/kg_builder_service.py
import os
import networkx as nx
import json
import time  # Thêm time để sleep nếu cần
import google.generativeai as genai  # Cần cho GenerationConfig và các type khác

# Giả sử GeminiApiKeyManager được truyền vào, và config được truyền hoặc import
# from ..utils.api_key_manager import GeminiApiKeyManager
# import config # (Nếu config được truy cập trực tiếp)


def _get_doc_summary_keywords_from_gemini(
    document_content: str,
    api_manager,  # Instance của GeminiApiKeyManager
    model_name: str,
    max_input_chars: int,
) -> dict:
    """
    Sử dụng Gemini để tạo tóm tắt và trích xuất keywords cho nội dung tài liệu.
    Trả về dict {"summary": "...", "keywords": ["kw1", ...]} hoặc giá trị mặc định khi thất bại.
    """
    # Prompt giữ nguyên như trong script bạn cung cấp
    prompt = f"""
Bạn là một trợ lý AI chuyên tóm tắt tài liệu và trích xuất từ khóa.
Cho nội dung tài liệu dưới đây (định dạng Markdown):

--- DOCUMENT CONTENT START ---
{document_content[:max_input_chars]} 
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
    api_params_for_method = {
        "contents": prompt,
        "generation_config": genai.types.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",  # Yêu cầu Gemini trả về JSON
        ),
        "safety_settings": [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ],
    }

    response = (
        api_manager.execute_generative_call(  # Sử dụng phương thức mới từ ApiKeyManager
            model_name_to_use=model_name,
            api_params_for_method=api_params_for_method,
            call_type="DocSummaryKeywords",
        )
    )

    if response and response.parts:
        response_text = response.text
        try:
            data = json.loads(response_text.strip())
            if (
                "summary" in data
                and "keywords" in data
                and isinstance(data["keywords"], list)
            ):
                return data
            else:
                print(
                    f"    Lỗi (KG Builder): JSON response tóm tắt/keywords không đúng cấu trúc. Data: {data}"
                )
        except json.JSONDecodeError as e_json:
            print(
                f"    Lỗi (KG Builder) giải mã JSON tóm tắt/keywords: {e_json}. Response text: {response_text}"
            )

    return {"summary": "Không thể tạo tóm tắt tự động do lỗi API.", "keywords": []}


def build_kg_from_markdown_files(
    processed_md_folder: str,
    api_manager,  # Instance của GeminiApiKeyManager
    summarization_model_name: str,
    chunk_delimiter: str,
    max_doc_chars_for_summary: int,
) -> nx.DiGraph | None:
    """
    Xây dựng Knowledge Graph từ các file markdown đã được xử lý (chứa delimiter chunk).
    """
    if not os.path.isdir(processed_md_folder):
        print(
            f"LỖI (KG Builder): Thư mục đầu vào '{processed_md_folder}' không tồn tại."
        )
        return None

    G = nx.DiGraph()
    print(f"\nBắt đầu xây dựng Knowledge Graph từ thư mục: {processed_md_folder}")
    file_count = 0

    files_to_process = [
        f
        for f in os.listdir(processed_md_folder)
        if f.endswith(("_processed.md", ".chunked.md", ".md"))
    ]

    if not files_to_process:
        print(f"Không tìm thấy file .md phù hợp trong '{processed_md_folder}'.")
        return None

    for filename in files_to_process:
        file_count += 1
        file_path = os.path.join(processed_md_folder, filename)
        doc_name = (
            filename.replace("_processed.md", "")
            .replace(".chunked.md", "")
            .replace(".md", "")
        )

        print(
            f"\n  Đang xử lý tài liệu KG: {filename} (Tài liệu #{file_count}/{len(files_to_process)})..."
        )
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_document_content = f.read()
            if not full_document_content.strip():
                print(f"    Cảnh báo: File '{filename}' rỗng. Bỏ qua.")
                continue

            print(f"    Đang tạo tóm tắt và keywords cho '{doc_name}'...")
            doc_info = _get_doc_summary_keywords_from_gemini(
                full_document_content,
                api_manager,
                summarization_model_name,
                max_doc_chars_for_summary,
            )

            summary = doc_info.get("summary", "N/A")
            keywords = doc_info.get("keywords", [])

            doc_node_id = f"doc:{doc_name}"
            G.add_node(
                doc_node_id,
                type="Document",
                name=doc_name,
                original_filename=filename,
                summary=summary,
                keywords=", ".join(keywords) if isinstance(keywords, list) else "",
            )
            # Bên trong hàm build_kg_from_markdown_files, sau khi có `summary` và `keywords`

            summary_preview = summary[:50].strip()
            if len(summary) > 50:
                summary_preview += "..."
            # Thay thế dòng print cũ bằng dòng này:
            print(
                f"    Đã tạo Document Node: {doc_node_id} (Summary: '{summary_preview}', Keywords: {keywords})"
            )
            text_chunks = full_document_content.split(chunk_delimiter)
            previous_chunk_node_id = None
            num_valid_chunks = 0
            for i, chunk_text_content in enumerate(text_chunks):
                clean_chunk_text = chunk_text_content.strip()
                if clean_chunk_text:
                    num_valid_chunks += 1
                    chunk_node_id = f"chunk:{doc_name}_{i}"
                    G.add_node(
                        chunk_node_id,
                        type="Chunk",
                        text_content=clean_chunk_text,
                        order_in_doc=i,
                        source_document_id=doc_node_id,
                    )
                    G.add_edge(doc_node_id, chunk_node_id, type="HAS_CHUNK")
                    if previous_chunk_node_id:
                        G.add_edge(
                            previous_chunk_node_id, chunk_node_id, type="NEXT_CHUNK"
                        )
                    previous_chunk_node_id = chunk_node_id
            print(
                f"    Đã thêm {num_valid_chunks} chunks hợp lệ cho '{doc_name}' vào KG."
            )
        except Exception as e_kg_file:
            print(f"    Lỗi nghiêm trọng khi xử lý file {filename} cho KG: {e_kg_file}")
            import traceback

            traceback.print_exc()

        if file_count < len(
            files_to_process
        ):  # Thêm sleep giữa các file để tránh quá tải API
            time.sleep(1)

    if G.number_of_nodes() == 0:
        print("Không có nút nào được tạo trong đồ thị. Kiểm tra file đầu vào và log.")
        return None
    return G

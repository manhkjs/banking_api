# src/chunking/delimiter_service.py
import google.generativeai as genai  # Cần cho type hints

# from ..utils.api_key_manager import GeminiApiKeyManager # Sửa import nếu cần


def standardize_headings_and_insert_delimiters(
    refined_markdown_content: str,
    api_manager,  # Instance của GeminiApiKeyManager
    model_name: str,
    chunk_delimiter: str,
    max_input_chars: int,
    language: str = "tiếng Việt, English",  # Ngôn ngữ có thể không cần thiết cho bước này nếu chỉ là cấu trúc
) -> str | None:
    """
    Sử dụng Gemini API để chuẩn hóa cấu trúc heading và chèn delimiter vào văn bản Markdown đã được tinh chỉnh.
    """
    if len(refined_markdown_content) > max_input_chars:
        print(
            f"    CẢNH BÁO (Delimiter): Nội dung Markdown quá dài ({len(refined_markdown_content)} chars). Cắt bớt còn {max_input_chars}."
        )
        refined_markdown_content = refined_markdown_content[:max_input_chars]

    prompt = f"""
Bạn là một chuyên gia cấu trúc và phân đoạn văn bản Markdown.
Văn bản Markdown bằng {language} dưới đây được cho là đã được sửa lỗi chính tả và các lỗi cú pháp Markdown cơ bản.
Nhiệm vụ của bạn là thực hiện các công việc sau theo đúng trình tự, trả về một văn bản Markdown DUY NHẤT đã hoàn thiện:

1.  **Bước 1: Chuẩn hóa Cấu trúc Tiêu đề (Headings) (RẤT QUAN TRỌNG):**
    * **Tiêu đề chính (Node gốc):** Nếu có một tiêu đề chính bao quát toàn bộ tài liệu (thường xuất hiện ở đầu tiên và là cấp cao nhất), hãy đảm bảo nó được định dạng là H1 (sử dụng một dấu `#`). Ví dụ: `# Tên Đơn Đăng Ký Chính`.
    * **Các Phần/Mục lớn:** Các phần lớn của tài liệu, ví dụ như "Phần I", "Phần II", "Mục A", "Điều 1", phải được định dạng là H2 (sử dụng hai dấu `##`). Ví dụ: `## Phần I: Thông Tin Chung`. **TUYỆT ĐỐI KHÔNG sử dụng H1 (`#`) cho các phần như "Phần II", "Phần III" này nếu chúng không phải là tiêu đề chính của toàn bộ tài liệu.**
    * **Các Mục con:** Các mục con trực tiếp của H2 nên là H3 (sử dụng ba dấu `###`). Ví dụ: `### 1.1. Chi tiết mục con`.
    * **Các cấp độ sâu hơn:** Tiếp tục sử dụng H4 (`####`), H5 (`#####`) cho các cấp độ tiêu đề sâu hơn nếu cần.
    * Mục tiêu là tạo ra một cấu trúc tiêu đề Markdown phân cấp rõ ràng và logic, phù hợp để thể hiện cấu trúc của một tài liệu chính thức.
    * Chuẩn hóa thêm khoảng trắng xung quanh các yếu tố Markdown nếu cần (ví dụ: một dòng trống trước và sau tiêu đề).

2.  **Bước 2: Chia văn bản thành các đoạn (chunks):**
    * Sau khi văn bản đã được **chuẩn hóa cấu trúc tiêu đề và định dạng Markdown ở Bước 1**, hãy chèn một dấu phân cách đặc biệt là "{chunk_delimiter.strip()}" vào những vị trí hợp lý.
    * Mỗi chunk nên là một đơn vị thông tin tương đối độc lập và có ngữ nghĩa.
    * Cố gắng không phá vỡ các bảng, danh sách dài, hoặc khối mã Markdown giữa chừng khi chèn delimiter. Nếu một bảng hoặc danh sách dài, bạn có thể chèn delimiter trước hoặc sau toàn bộ cấu trúc đó.
    * Ưu tiên chèn delimiter **sau** một khối tiêu đề hoàn chỉnh (ví dụ: sau toàn bộ nội dung của một mục H2 hoặc H3 đã được chuẩn hóa), hoặc giữa các đoạn văn (paragraphs) có sự chuyển ý rõ ràng hoặc ít liên kết chặt chẽ về mặt ngữ nghĩa.
    * Dấu phân cách "{chunk_delimiter.strip()}" phải nằm trên một dòng riêng của chính nó.

**YÊU CẦU ĐẦU RA TUYỆT ĐỐI:**
* Giữ nguyên tuyệt đối ý nghĩa và cấu trúc logic cơ bản của nội dung gốc (ngoài việc chuẩn hóa heading). Chỉnh sửa chỉ nhằm chuẩn hóa Markdown theo quy tắc trên, và thêm delimiter.
* **Chỉ trả về nội dung văn bản Markdown đã được xử lý hoàn chỉnh theo đúng 2 bước trên.** Không thêm bất kỳ lời giải thích, bình luận, lời chào, phần mở đầu hay phần kết luận nào khác ngoài văn bản Markdown đã được xử lý.

Văn bản Markdown cần xử lý (đã qua bước sửa lỗi chính tả và cú pháp cơ bản):
---
{refined_markdown_content}
---

Văn bản Markdown đã được chuẩn hóa heading và chèn dấu phân cách chunk:
"""
    api_params = {
        "contents": prompt,
        "generation_config": genai.types.GenerationConfig(
            temperature=0.1
        ),  # Nhiệt độ thấp để bám sát chỉ dẫn
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

    response = api_manager.execute_generative_call(
        model_name_to_use=model_name,
        api_params_for_method=api_params,
        call_type="Heading Standardization & Chunk Delimiting",
    )

    if response and response.parts:
        return response.text
    else:
        if (
            response
            and hasattr(response, "prompt_feedback")
            and response.prompt_feedback
        ):
            print(
                f"    Lỗi (Delimiter): Response từ Gemini không có nội dung. Feedback: {response.prompt_feedback}"
            )
        else:
            print(
                f"    Lỗi (Delimiter): Response từ Gemini không có nội dung hoặc response là None."
            )
        return None

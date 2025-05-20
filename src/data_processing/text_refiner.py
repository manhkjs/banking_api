# src/data_processing/text_refiner_service.py
import google.generativeai as genai  # Cần cho type hints và GenerationConfig

# from ..utils.api_key_manager import GeminiApiKeyManager # Sửa đường dẫn import nếu cần
# import config # Để lấy CHUNK_DELIMITER nếu bạn muốn dùng ở đây


def refine_text_spellcheck_basic_md(
    raw_markdown_content: str,
    api_manager,  # Instance của GeminiApiKeyManager
    model_name: str,
    max_input_chars: int,
    # Thêm biến chunk_delimiter từ config vào đây để có thể tham chiếu trong prompt
    # Hoặc hardcode một chuỗi mô tả như "---DẤU PHÂN CÁCH CHUNK---"
    # Cách tốt hơn là không đề cập đến delimiter cụ thể nếu không muốn nó xuất hiện.
    language: str = "tiếng Việt, English",
) -> str | None:
    """
    Sử dụng Gemini API để sửa lỗi chính tả, ngữ pháp và các lỗi cú pháp Markdown cơ bản.
    KHÔNG thay đổi cấu trúc heading hoặc chèn delimiter ở bước này.
    """
    if len(raw_markdown_content) > max_input_chars:
        print(
            f"    CẢNH BÁO (Refiner): Nội dung Markdown quá dài ({len(raw_markdown_content)} chars). Cắt bớt còn {max_input_chars}."
        )
        raw_markdown_content = raw_markdown_content[:max_input_chars]

    # Sửa lại phần prompt liên quan đến delimiter
    # Thay vì dùng biến không tồn tại, mô tả nó bằng một chuỗi cố định hoặc loại bỏ nếu không cần thiết
    # Trong trường hợp này, chúng ta muốn LLM KHÔNG chèn delimiter, nên có thể chỉ cần nói rõ.

    # Lấy CHUNK_DELIMITER từ config để minh họa trong prompt (nếu muốn LLM biết nó là gì để tránh)
    # Cách 1: Truyền chunk_delimiter vào hàm này (thêm tham số cho hàm)
    # from config import CHUNK_DELIMITER as actual_chunk_delimiter_from_config
    # chunk_delimiter_for_prompt_avoidance = actual_chunk_delimiter_from_config.strip()

    # Cách 2: Hardcode một chuỗi mô tả (an toàn hơn nếu không muốn truyền thêm tham số)
    chunk_delimiter_description_for_prompt = '"---CHUNK_DELIMITER---"'  # Mô tả chung

    prompt = f"""
Bạn là một trợ lý biên tập văn bản Markdown chuyên nghiệp và cẩn thận, có kinh nghiệm làm việc với các văn bản được chuyển đổi từ hình ảnh qua OCR.
Nhiệm vụ của bạn là đọc văn bản Markdown bằng {language} dưới đây. Văn bản này là kết quả từ quá trình OCR và có thể chứa nhiều lỗi về chính tả, ngữ pháp, cú pháp Markdown, cũng như các câu hoặc đoạn văn ngắn bị lộn xộn, khó hiểu về mặt logic. Hãy thực hiện các công việc sau để cải thiện chất lượng và tính logic của nó:

1.  **Kiểm tra và sửa lỗi chính tả và ngữ pháp:** Sửa tất cả các lỗi chính tả và ngữ pháp {language} một cách chính xác. Đảm bảo cách dùng từ tự nhiên, phù hợp với văn phong chung của tài liệu.
2.  **Cải thiện tính logic và sự mạch lạc của câu/đoạn văn ngắn (do lỗi OCR):**
    * Vì đây là văn bản OCR, một số câu hoặc cụm từ có thể bị nhận dạng sai, từ ngữ bị đảo lộn hoặc các câu ngắn liền kề thiếu sự liên kết tự nhiên.
    * Hãy **điều chỉnh lại từ ngữ trong câu, hoặc sắp xếp lại thứ tự các câu ngắn liền kề trong cùng một đoạn văn** nếu điều đó giúp nội dung trở nên rõ ràng, mạch lạc và logic hơn, phản ánh đúng ý định có thể có của văn bản gốc.
    * **Ví dụ:** Nếu OCR ra "Ngân hàng cung cấp dịch vụ vay. Lãi suất ưu đãi.", bạn có thể sửa thành "Ngân hàng cung cấp dịch vụ vay với lãi suất ưu đãi." nếu ngữ cảnh cho phép và đó là cách diễn đạt tự nhiên hơn. Hoặc nếu có các ý rời rạc do OCR, hãy cố gắng kết nối chúng một cách nhẹ nhàng nếu rõ ràng về mặt ý nghĩa.
    * **Quan trọng:** Chỉ thực hiện các thay đổi nhỏ ở cấp độ câu hoặc một vài câu liền kề để cải thiện tính logic cục bộ và dòng chảy của văn bản.
    * **TUYỆT ĐỐI KHÔNG thay đổi các thông tin thực tế, dữ liệu số, tên riêng, hoặc ý nghĩa cốt lõi của văn bản.** Mục tiêu là làm cho văn bản dễ đọc và dễ hiểu hơn sau OCR, không phải viết lại hoàn toàn nội dung hay thêm thông tin mới.
3.  **Sửa lỗi cú pháp Markdown cơ bản:**
    * Sửa các lỗi cú pháp Markdown đơn giản như danh sách không đúng định dạng (ví dụ: thiếu dấu -, *, 1. ở đầu dòng), khối mã (` ``` `) hoặc trích dẫn (`>`) không được đóng đúng cách hoặc bị ngắt quãng.
    * Chuẩn hóa khoảng trắng thừa không cần thiết có thể ảnh hưởng đến hiển thị Markdown cơ bản (ví dụ: nhiều dấu cách liên tiếp trong một dòng văn bản, quá nhiều dòng trống không cần thiết giữa các đoạn văn ngắn).
    * Loại bỏ các ký tự lạ, nhiễu từ OCR rõ ràng mà không phải là một phần của nội dung hay cú pháp Markdown (ví dụ: các ký tự không thể in được, các mảnh vỡ từ không có nghĩa, các ký tự lặp lại không cần thiết).

**CÁC ĐIỀU TUYỆT ĐỐI KHÔNG LÀM Ở BƯỚC NÀY:**
* **KHÔNG được thay đổi cấu trúc tiêu đề (headings #, ##, ###, v.v.) dưới bất kỳ hình thức nào.** Giữ nguyên các cấp độ tiêu đề và nội dung chữ của tiêu đề như trong văn bản gốc.
* **KHÔNG được chèn bất kỳ dấu phân cách chunk nào, ví dụ như {chunk_delimiter_description_for_prompt}, vào văn bản.**
* **KHÔNG được thêm thông tin mới không có trong văn bản gốc hoặc xóa bỏ các phần thông tin quan trọng.**

**YÊU CẦU ĐẦU RA:**
* Chỉ trả về nội dung văn bản Markdown đã được sửa lỗi chính tả, ngữ pháp, cải thiện tính logic cục bộ ở mức câu/đoạn ngắn, và sửa lỗi cú pháp Markdown cơ bản.
* Văn bản trả về phải giữ nguyên cấu trúc heading hiện có trong văn bản gốc.
* Không thêm bất kỳ lời giải thích, bình luận, lời chào hay bất kỳ văn bản nào khác ngoài nội dung đã được xử lý.

Văn bản Markdown cần xử lý (kết quả từ OCR):
---
{raw_markdown_content}
---

Văn bản Markdown đã được xử lý (sửa lỗi chính tả, ngữ pháp, cải thiện logic cục bộ, sửa lỗi cú pháp MD cơ bản, giữ nguyên cấu trúc heading):
"""
    # Lưu ý: chunk_delimiter_description_for_prompt chỉ là một chuỗi text trong prompt, không phải biến Python.
    # Nếu bạn muốn nó thực sự là giá trị từ config.CHUNK_DELIMITER thì bạn phải truyền biến đó vào hàm này.
    # Ví dụ: def refine_text_spellcheck_basic_md(..., chunk_delimiter_to_avoid: str)
    # Và trong prompt: (như "{chunk_delimiter_to_avoid.strip()}") vào văn bản.
    # Tuy nhiên, để đơn giản và tránh lỗi, sử dụng một chuỗi mô tả cố định như trên là được.

    api_params = {
        "contents": prompt,
        "generation_config": genai.types.GenerationConfig(
            temperature=0.15  # Giữ ở mức thấp hoặc điều chỉnh nhẹ
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

    response = api_manager.execute_generative_call(
        model_name_to_use=model_name,
        api_params_for_method=api_params,
        call_type="Text Refining (Spellcheck & Basic MD & Logic Flow)",  # Cập nhật call_type
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
                f"    Lỗi (Refiner): Response từ Gemini không có nội dung. Feedback: {response.prompt_feedback}"
            )
        else:
            print(
                f"    Lỗi (Refiner): Response từ Gemini không có nội dung hoặc response là None."
            )
        return None

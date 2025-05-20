import google.generativeai as genai
import os
import time
from dotenv import load_dotenv

# --- CẤU HÌNH ---
MODEL_NAME = "gemini-1.5-flash-latest"  # Hoặc "gemini-1.0-pro", "gemini-1.5-pro-latest"
CHUNK_DELIMITER = "\n\n---CHUNK_DELIMITER---\n\n"  # Dấu phân cách chunk
RETRY_WAIT_SECONDS = 60
MAX_RETRIES_PER_KEY_CYCLE = (
    2  # Số lần thử lại với tất cả các key trước khi dừng hẳn (tùy chọn)
)
OUTPUT_FILE_SUFFIX = "_processed.md"  # Đuôi file cho kết quả đã xử lý


# --- HÀM TẢI API KEYS ---
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
        print(
            "LỖI: Không tìm thấy GEMINI_API_KEY nào trong file .env. Định dạng cần là GEMINI_API_KEY_1, GEMINI_API_KEY_2,..."
        )
    return api_keys


# --- HÀM XỬ LÝ VĂN BẢN VỚI GEMINI (SỬA LỖI, CHUẨN HÓA MARKDOWN, CHUNKING) ---
def process_markdown_with_gemini(
    markdown_content,
    api_keys_list,
    current_key_index_ref,
    total_keys_exhausted_cycles_ref,
    language="tiếng Việt, English",
):
    """
    Gửi nội dung Markdown đến Gemini API để sửa lỗi chính tả, chuẩn hóa Markdown (bao gồm cấu trúc heading)
    và chèn dấu phân cách chunk.
    """
    if not api_keys_list:
        print("LỖI: Không có API key nào được cung cấp.")
        return None

    # Giới hạn độ dài input cho Gemini để tránh lỗi vượt token hoặc chi phí không cần thiết
    # Gemini 1.5 Flash có context window lớn, nhưng việc xử lý văn bản quá dài trong một lượt
    # có thể làm giảm chất lượng hoặc tăng khả năng bị timeout/lỗi.
    # Cân nhắc chia nhỏ file lớn hơn ở tầng ngoài nếu cần.
    MAX_INPUT_CHARS = (
        300000  # Ví dụ: giới hạn 300k ký tự (khoảng < 100k token, tùy nội dung)
    )
    if len(markdown_content) > MAX_INPUT_CHARS:
        print(
            f"    CẢNH BÁO: Nội dung Markdown quá dài ({len(markdown_content)} chars)."
        )
        print(f"    Sẽ cắt bớt còn {MAX_INPUT_CHARS} ký tự đầu tiên để xử lý.")
        markdown_content = markdown_content[:MAX_INPUT_CHARS]

    prompt = f"""
Bạn là một chuyên gia biên tập và cấu trúc văn bản Markdown cực kỳ cẩn thận và thông minh.
Nhiệm vụ của bạn là đọc văn bản Markdown bằng {language} dưới đây và thực hiện các công việc sau theo đúng trình tự, trả về một văn bản Markdown DUY NHẤT đã hoàn thiện:

**QUY TRÌNH XỬ LÝ BẮT BUỘC:**

1.  **Bước 1: Sửa lỗi chính tả và ngữ pháp:**
    * Sửa tất cả các lỗi chính tả {language} một cách chính xác.
    * Đảm bảo ngữ pháp, dấu câu và cách dùng từ phù hợp với văn phong của tài liệu gốc.

2.  **Bước 2: Chuẩn hóa định dạng Markdown và Cấu trúc Tiêu đề (Headings):**
    * **Sửa lỗi cú pháp Markdown:** Đảm bảo cú pháp Markdown cơ bản là đúng chuẩn (ví dụ: danh sách -, *, 1.; khối mã ```; trích dẫn >). Sửa các lỗi định dạng nếu có (danh sách không đúng cách, bảng bị lệch).
    * **QUAN TRỌNG - Chuẩn hóa Cấu trúc Tiêu đề:**
        * **Tiêu đề chính (Node gốc):** Nếu có một tiêu đề chính bao quát toàn bộ tài liệu (thường xuất hiện ở đầu tiên và là cấp cao nhất), hãy đảm bảo nó được định dạng là H1 (sử dụng một dấu `#`). Ví dụ: `# Tên Đơn Đăng Ký Chính`.
        * **Các Phần/Mục lớn:** Các phần lớn của tài liệu, ví dụ như "Phần I", "Phần II", "Mục A", "Điều 1", phải được định dạng là H2 (sử dụng hai dấu `##`). Ví dụ: `## Phần I: Thông Tin Chung`. **TUYỆT ĐỐI KHÔNG sử dụng H1 (`#`) cho các phần như "Phần II", "Phần III" này nếu chúng không phải là tiêu đề chính của toàn bộ tài liệu.**
        * **Các Mục con:** Các mục con trực tiếp của H2 nên là H3 (sử dụng ba dấu `###`). Ví dụ: `### 1.1. Chi tiết mục con`.
        * **Các cấp độ sâu hơn:** Tiếp tục sử dụng H4 (`####`), H5 (`#####`) cho các cấp độ tiêu đề sâu hơn nếu cần.
        * Mục tiêu là tạo ra một cấu trúc tiêu đề Markdown phân cấp rõ ràng và logic.
    * **Định dạng khác:** Chuẩn hóa khoảng trắng xung quanh các yếu tố Markdown (ví dụ: luôn có một dòng trống trước và sau các khối tiêu đề, danh sách, bảng, khối mã). Loại bỏ các ký tự hoặc định dạng lạ, nhiễu từ OCR không phải là Markdown hợp lệ hoặc không mang ý nghĩa.

3.  **Bước 3: Chia văn bản thành các đoạn (chunks):**
    * Sau khi văn bản đã được sửa lỗi chính tả và quan trọng nhất là đã được **chuẩn hóa cấu trúc tiêu đề và định dạng Markdown ở Bước 2**, hãy chèn một dấu phân cách đặc biệt là "{CHUNK_DELIMITER.strip()}" vào những vị trí hợp lý.
    * Mỗi chunk nên là một đơn vị thông tin tương đối độc lập và có ngữ nghĩa.
    * Cố gắng không phá vỡ các bảng, danh sách dài, hoặc khối mã Markdown giữa chừng khi chèn delimiter. Nếu một bảng hoặc danh sách dài, bạn có thể chèn delimiter trước hoặc sau toàn bộ cấu trúc đó.
    * Ưu tiên chèn delimiter **sau** một khối tiêu đề hoàn chỉnh (ví dụ: sau nội dung của một mục H2 hoặc H3), hoặc giữa các đoạn văn (paragraphs) có sự chuyển ý rõ ràng hoặc ít liên kết chặt chẽ về mặt ngữ nghĩa.
    * Dấu phân cách "{CHUNK_DELIMITER.strip()}" phải nằm trên một dòng riêng của chính nó.

**YÊU CẦU ĐẦU RA TUYỆT ĐỐI:**
* **Giữ nguyên tuyệt đối ý nghĩa và cấu trúc logic cơ bản của nội dung gốc.** Các chỉnh sửa chỉ nhằm mục đích cải thiện chất lượng chính tả, định dạng Markdown theo quy tắc trên, và thêm dấu phân cách chunk.
* Không thêm, bớt hoặc thay đổi nội dung thông tin của văn bản ngoài việc sửa lỗi và chèn delimiter.
* **Chỉ trả về nội dung văn bản Markdown đã được xử lý hoàn chỉnh theo đúng 3 bước trên.** Không thêm bất kỳ lời giải thích, bình luận, lời chào, phần mở đầu hay phần kết luận nào khác ngoài văn bản Markdown đã được xử lý.

Văn bản Markdown cần xử lý:
---
{markdown_content}
---

Văn bản Markdown đã được xử lý hoàn chỉnh:
"""

    while True:
        current_api_key = api_keys_list[current_key_index_ref[0]]
        print(f"    Đang sử dụng API Key #{current_key_index_ref[0] + 1} để xử lý...")

        try:
            genai.configure(api_key=current_api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            generation_config = genai.types.GenerationConfig(
                temperature=0.05,
                # max_output_tokens=16384 # Model Flash có thể không hỗ trợ output lớn như Pro
            )
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
                processed_text = response.text
                print(f"    Xử lý thành công với Key #{current_key_index_ref[0] + 1}.")
                total_keys_exhausted_cycles_ref[0] = 0
                return processed_text
            else:
                error_message = f"    Lỗi: Response từ Gemini không có nội dung (Key #{current_key_index_ref[0] + 1})."
                if response.prompt_feedback:
                    error_message += f" Prompt Feedback: {response.prompt_feedback}"
                print(error_message)
                raise genai.types.generation_types.BlockedPromptException(
                    "Response trống từ Gemini"
                )

        except (
            genai.types.generation_types.BlockedPromptException,
            genai.types.generation_types.StopCandidateException,
        ) as e:
            print(
                f"    Cảnh báo/Lỗi (Blocked/Stop) với Key #{current_key_index_ref[0] + 1}: {type(e).__name__} - {e}"
            )
            current_key_index_ref[0] = (current_key_index_ref[0] + 1) % len(
                api_keys_list
            )
            if current_key_index_ref[0] == 0:
                total_keys_exhausted_cycles_ref[0] += 1
                print(
                    f"    Tất cả API key đã được thử {total_keys_exhausted_cycles_ref[0]} lần và gặp vấn đề (Blocked/Stop)."
                )
                if total_keys_exhausted_cycles_ref[0] >= MAX_RETRIES_PER_KEY_CYCLE:
                    print(
                        f"    Đã thử {MAX_RETRIES_PER_KEY_CYCLE} chu kỳ với tất cả các key. Bỏ qua file này."
                    )
                    return None
                print(
                    f"    Đang đợi {RETRY_WAIT_SECONDS} giây trước khi thử lại chu kỳ mới..."
                )
                time.sleep(RETRY_WAIT_SECONDS)

        except Exception as e:
            if (
                "429" in str(e)
                or "resource_exhausted" in str(e).lower()
                or "quota" in str(e).lower()
            ):
                print(
                    f"    Lỗi Quota với Key #{current_key_index_ref[0] + 1}. Chi tiết: {type(e).__name__} - {e}"
                )
            else:
                print(
                    f"    Lỗi không xác định với Key #{current_key_index_ref[0] + 1}: {type(e).__name__} - {e}"
                )
                import traceback

                traceback.print_exc()

            current_key_index_ref[0] = (current_key_index_ref[0] + 1) % len(
                api_keys_list
            )
            if current_key_index_ref[0] == 0:
                total_keys_exhausted_cycles_ref[0] += 1
                print(
                    f"    Tất cả API key đã gặp lỗi/quota {total_keys_exhausted_cycles_ref[0]} lần trong chu kỳ này."
                )
                if total_keys_exhausted_cycles_ref[0] >= MAX_RETRIES_PER_KEY_CYCLE:
                    print(
                        f"    Đã thử {MAX_RETRIES_PER_KEY_CYCLE} chu kỳ với tất cả các key. Bỏ qua file này."
                    )
                    return None
                print(
                    f"    Đang đợi {RETRY_WAIT_SECONDS} giây trước khi thử lại chu kỳ mới..."
                )
                time.sleep(RETRY_WAIT_SECONDS)


# --- HÀM XỬ LÝ HÀNG LOẠT ---
def process_markdown_files_in_folder(
    input_dir, output_dir, api_keys, text_language="tiếng Việt, English"
):
    """
    Đọc tất cả các file .md trong input_dir, xử lý bằng Gemini, và lưu vào output_dir.
    """
    if not api_keys:
        print("Không có API key để thực hiện. Dừng chương trình.")
        return

    if not os.path.isdir(input_dir):
        print(f"LỖI: Thư mục đầu vào '{input_dir}' không tồn tại.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Đã tạo thư mục đầu ra: '{output_dir}'")
        except OSError as e:
            print(f"LỖI: Không thể tạo thư mục đầu ra '{output_dir}': {e}")
            return

    current_key_idx_ref = [0]
    total_exhausted_cycles_ref = [0]

    md_files_found = 0
    successful_processing_count = 0
    failed_processing_files = []

    print(f"\nBắt đầu quét thư mục '{input_dir}' để tìm file .md...")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(
            (".md", ".txt")
        ):  # Có thể xử lý cả .txt nếu nội dung là markdown
            md_files_found += 1
            input_file_path = os.path.join(input_dir, filename)

            base_filename, file_ext = os.path.splitext(filename)
            output_processed_path = os.path.join(
                output_dir,
                base_filename + OUTPUT_FILE_SUFFIX,  # Ví dụ: file_goc_processed.md
            )

            print(f"\nĐang xử lý file: {filename} (File {md_files_found})...")
            start_time = time.time()

            try:
                with open(input_file_path, "r", encoding="utf-8") as f_in:
                    original_content = f_in.read()

                if not original_content.strip():
                    print(f"    Cảnh báo: File '{filename}' rỗng. Bỏ qua.")
                    failed_processing_files.append(f"{filename} (rỗng)")
                    continue

                processed_content = process_markdown_with_gemini(
                    original_content,
                    api_keys,
                    current_key_idx_ref,
                    total_exhausted_cycles_ref,
                    language=text_language,
                )

                if processed_content:
                    with open(output_processed_path, "w", encoding="utf-8") as f_out:
                        f_out.write(processed_content)
                    successful_processing_count += 1
                    elapsed_time = time.time() - start_time
                    print(
                        f"  Xử lý thành công '{filename}'. Đã lưu vào '{output_processed_path}'. Thời gian: {elapsed_time:.2f} giây."
                    )
                else:
                    failed_processing_files.append(filename)
                    print(
                        f"  Xử lý thất bại cho '{filename}' sau nhiều lần thử hoặc do lỗi."
                    )

            except Exception as e_file_processing:
                print(
                    f"  Lỗi nghiêm trọng khi xử lý file '{filename}': {e_file_processing}"
                )
                failed_processing_files.append(f"{filename} (lỗi đọc/ghi)")
                import traceback

                traceback.print_exc()

    print("\n--- HOÀN THÀNH QUÁ TRÌNH XỬ LÝ MARKDOWN BẰNG GEMINI ---")
    print(f"Tổng số file .md (hoặc .txt) được tìm thấy: {md_files_found}")
    print(f"Số file được xử lý thành công: {successful_processing_count}")
    if failed_processing_files:
        print(f"Số file xử lý thất bại hoặc bị bỏ qua: {len(failed_processing_files)}")
        print("Danh sách file thất bại/bỏ qua:")
        for failed_file in failed_processing_files:
            print(f"  - {failed_file}")
    elif md_files_found > 0:
        print("Tất cả các file đã được xử lý (hoặc đã thử xử lý).")
    else:
        print("Không tìm thấy file .md hoặc .txt nào trong thư mục đầu vào.")


# --- KHỐI THỰC THI CHÍNH ---
if __name__ == "__main__":
    # --- CẤU HÌNH ---
    # THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP
    # Thư mục chứa file .md từ Mistral OCR (hoặc file .txt chứa markdown)
    input_markdown_dir = (
        "E:/kienlong/banking_ai_platform/data/processed_data/mistral_ocr_output_md"
    )
    # Thư mục lưu kết quả sau khi Gemini xử lý (sửa lỗi, chuẩn hóa MD, chèn delimiter chunk)
    output_processed_dir = "E:/kienlong/banking_ai_platform/data/processed_data/gemini_processed_markdown_v2"  # Đổi tên thư mục output mới
    document_language = "tiếng Việt, English"  # Ngôn ngữ của tài liệu để sửa chính tả

    # --- KẾT THÚC CẤU HÌNH ---

    gemini_keys = load_api_keys()
    if not gemini_keys:
        print(
            "Không tìm thấy API key nào của Gemini trong file .env. Vui lòng kiểm tra lại."
        )
        exit()

    print(f"Đã tải {len(gemini_keys)} API key của Gemini.")
    if genai.__version__:  # Kiểm tra xem thư viện đã được import đúng chưa
        print(f"Phiên bản google-generativeai: {genai.__version__}")
    else:
        print("Không thể xác định phiên bản google-generativeai.")

    valid_config = True
    if not input_markdown_dir:
        print("LỖI: 'input_markdown_dir' chưa được cấu hình (trống).")
        valid_config = False
    elif not os.path.isdir(input_markdown_dir):
        print(
            f"LỖI: Đường dẫn đầu vào '{input_markdown_dir}' không phải là một thư mục hoặc không tồn tại."
        )
        valid_config = False

    if not output_processed_dir:
        print("LỖI: 'output_processed_dir' chưa được cấu hình (trống).")
        valid_config = False

    if valid_config:
        print(f"Thư mục file Markdown đầu vào: {input_markdown_dir}")
        print(f"Thư mục lưu kết quả đã xử lý: {output_processed_dir}")
        print(f"Sử dụng mô hình Gemini: {MODEL_NAME}")
        print(f"Ngôn ngữ tài liệu cho sửa lỗi chính tả: {document_language}")
        print(f'Dấu phân cách chunk sẽ được chèn: "{CHUNK_DELIMITER.strip()}"')

        confirmation = (
            input("Bạn có muốn tiếp tục với các cài đặt trên? (yes/no): ")
            .strip()
            .lower()
        )
        if confirmation == "yes" or confirmation == "y":
            print("\nBắt đầu quá trình xử lý văn bản Markdown bằng Gemini API...")
            process_markdown_files_in_folder(
                input_markdown_dir,
                output_processed_dir,
                gemini_keys,
                text_language=document_language,
            )
        else:
            print("Đã hủy quá trình xử lý.")
    else:
        print("Vui lòng kiểm tra lại cấu hình đường dẫn.")

import fitz  # PyMuPDF
import os
import time


def extract_text_from_single_pdf(pdf_path, output_txt_path):
    """
    Trích xuất toàn bộ văn bản từ một file PDF và lưu vào file .txt.
    Sử dụng phương pháp trích xuất văn bản chuẩn (không phải OCR).

    Args:
        pdf_path (str): Đường dẫn đến file PDF đầu vào.
        output_txt_path (str): Đường dẫn đến file .txt để lưu kết quả.

    Returns:
        bool: True nếu trích xuất và lưu thành công, False nếu có lỗi.
    """
    full_page_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text(
                "text"
            )  # "text" là tùy chọn mặc định, có thể bỏ qua
            full_page_text += (
                page_text + "\n"
            )  # Thêm dấu xuống dòng giữa các trang (tùy chọn)
        doc.close()

        with open(output_txt_path, "w", encoding="utf-8") as f_out:
            f_out.write(full_page_text)
        return True
    except Exception as e:
        print(f"  Lỗi khi xử lý file '{os.path.basename(pdf_path)}': {e}")
        # Tạo file lỗi để ghi nhận
        error_log_path = output_txt_path + ".error.txt"
        with open(error_log_path, "w", encoding="utf-8") as f_err:
            f_err.write(
                f"Không thể trích xuất văn bản từ file PDF: {pdf_path}\nLỗi: {e}"
            )
        return False


def process_all_pdfs_in_folder(input_folder, output_folder):
    """
    Trích xuất văn bản từ tất cả các file PDF trong một thư mục
    và lưu kết quả vào các file .txt tương ứng trong một thư mục khác.

    Args:
        input_folder (str): Đường dẫn đến thư mục chứa các file PDF.
        output_folder (str): Đường dẫn đến thư mục để lưu các file .txt kết quả.
    """
    if not os.path.isdir(input_folder):
        print(
            f"LỖI: Thư mục đầu vào '{input_folder}' không tồn tại hoặc không phải là thư mục."
        )
        return

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Đã tạo thư mục đầu ra: '{output_folder}'")
        except OSError as e:
            print(f"LỖI: Không thể tạo thư mục đầu ra '{output_folder}': {e}")
            return

    pdf_files_found = 0
    successful_extraction_count = 0
    failed_extraction_files = []

    print(f"\nBắt đầu quét thư mục '{input_folder}' để tìm file PDF...")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_files_found += 1
            input_pdf_path = os.path.join(input_folder, filename)

            # Tạo tên file output, giữ nguyên tên gốc nhưng đổi phần mở rộng thành .txt
            base_filename_without_ext = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(
                output_folder, base_filename_without_ext + ".txt"
            )

            print(f"\nĐang xử lý file: {filename} (File {pdf_files_found})...")
            start_time = time.time()

            if extract_text_from_single_pdf(input_pdf_path, output_txt_path):
                successful_extraction_count += 1
                elapsed_time = time.time() - start_time
                print(
                    f"  Trích xuất thành công '{filename}'. Đã lưu vào '{output_txt_path}'. Thời gian: {elapsed_time:.2f} giây."
                )
            else:
                failed_extraction_files.append(filename)
                print(
                    f"  Trích xuất thất bại '{filename}'. Xem file .error.txt (nếu có) để biết chi tiết."
                )

    print("\n--- HOÀN THÀNH QUÁ TRÌNH TRÍCH XUẤT VĂN BẢN ---")
    print(f"Phiên bản PyMuPDF (fitz): {fitz.__doc__}")
    print(f"Tổng số file PDF được tìm thấy: {pdf_files_found}")
    print(f"Số file PDF được trích xuất thành công: {successful_extraction_count}")
    if failed_extraction_files:
        print(f"Số file PDF trích xuất thất bại: {len(failed_extraction_files)}")
        print("Danh sách file thất bại:")
        for failed_file in failed_extraction_files:
            print(f"  - {failed_file}")
    elif pdf_files_found > 0:
        print("Tất cả các file PDF đã được xử lý thành công.")
    else:
        print("Không tìm thấy file PDF nào trong thư mục đầu vào.")


# --- Khối thực thi chính ---
if __name__ == "__main__":
    # --- CẤU HÌNH ---
    # THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI MÁY CỦA BẠN
    input_pdf_folder = "E:/kienlong/banking_ai_platform/data/raw_data"
    output_text_folder = "E:/kienlong/banking_ai_platform/data/processed_data/text_extracted"  # Đổi tên thư mục output để tránh ghi đè nếu có

    # --- KẾT THÚC CẤU HÌNH ---

    print(f"Phiên bản PyMuPDF (fitz) đang sử dụng: {fitz.__doc__}")

    valid_config = True
    if not input_pdf_folder:
        print("LỖI: 'input_pdf_folder' chưa được cấu hình (trống) trong mã nguồn.")
        valid_config = False
    elif not os.path.isdir(input_pdf_folder):
        print(
            f"LỖI: Đường dẫn đầu vào '{input_pdf_folder}' không phải là một thư mục hoặc không tồn tại."
        )
        valid_config = False

    if not output_text_folder:
        print("LỖI: 'output_text_folder' chưa được cấu hình (trống) trong mã nguồn.")
        valid_config = False

    if valid_config:
        print(f"Thư mục PDF đầu vào: {input_pdf_folder}")
        print(f"Thư mục lưu text trích xuất: {output_text_folder}")

        confirmation = (
            input("Bạn có muốn tiếp tục với các cài đặt trên? (yes/no): ")
            .strip()
            .lower()
        )
        if confirmation == "yes" or confirmation == "y":
            print("\nBắt đầu quá trình trích xuất văn bản hàng loạt...")
            process_all_pdfs_in_folder(input_pdf_folder, output_text_folder)
        else:
            print("Đã hủy quá trình trích xuất văn bản.")

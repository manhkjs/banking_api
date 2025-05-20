# src/data_processing/ocr_service.py
import os
import time
from mistralai import Mistral  # Import thư viện Mistral


def ocr_single_pdf_with_mistral(
    pdf_path: str, output_md_path: str, mistral_client: Mistral, ocr_model_name: str
) -> bool:
    """
    Thực hiện OCR trên một file PDF bằng Mistral AI và lưu kết quả (Markdown).
    Sử dụng phương thức tải file lên, lấy signed URL, rồi xử lý OCR.

    Args:
        pdf_path (str): Đường dẫn đến file PDF đầu vào.
        output_md_path (str): Đường dẫn đến file .md để lưu kết quả Markdown.
        mistral_client (Mistral): Đối tượng client của Mistral AI đã được khởi tạo.
        ocr_model_name (str): Tên model OCR của Mistral (ví dụ: "mistral-ocr-latest").

    Returns:
        bool: True nếu OCR và lưu thành công, False nếu có lỗi.
    """
    base_filename = os.path.basename(pdf_path)
    print(f"    Bắt đầu OCR cho file: {base_filename}...")
    uploaded_file_id = None  # Để theo dõi ID file đã tải lên cho việc xóa nếu có lỗi

    try:
        # Bước 1: Tải file PDF lên Mistral
        print(f"      Đang tải file '{base_filename}' lên Mistral...")
        with open(pdf_path, "rb") as f:
            uploaded_pdf_response = mistral_client.files.upload(
                file={"file_name": base_filename, "content": f}, purpose="ocr"
            )

        if not uploaded_pdf_response or not uploaded_pdf_response.id:
            print(f"      Lỗi: Không tải được file '{base_filename}' lên Mistral.")
            return False
        uploaded_file_id = uploaded_pdf_response.id  # Lưu ID để xóa sau
        print(f"      Tải file thành công. File ID: {uploaded_file_id}")

        # Có thể cần đợi một chút để file được xử lý trên server Mistral
        # time.sleep(3) # Ví dụ: 3 giây

        # Bước 2: Lấy signed URL cho file đã tải lên
        print(f"      Đang lấy signed URL cho file ID: {uploaded_file_id}...")
        signed_url_response = mistral_client.files.get_signed_url(
            file_id=uploaded_file_id
        )
        if not signed_url_response or not signed_url_response.url:
            print(
                f"      Lỗi: Không lấy được signed URL cho file ID: {uploaded_file_id}"
            )
            return False  # Không cần xóa file ở đây, sẽ xóa ở khối finally nếu uploaded_file_id có giá trị
        print(f"      Lấy signed URL thành công.")

        # Bước 3: Thực hiện OCR bằng signed URL
        print(f"      Đang thực hiện OCR từ signed URL...")
        ocr_response = mistral_client.ocr.process(
            model=ocr_model_name,
            document={"type": "document_url", "document_url": signed_url_response.url},
        )

        # Bước 4: Xử lý kết quả và lưu file
        all_pages_markdown_content = []
        if (
            hasattr(ocr_response, "pages")
            and isinstance(ocr_response.pages, list)
            and ocr_response.pages
        ):
            for page_object in ocr_response.pages:
                if hasattr(page_object, "markdown") and isinstance(
                    page_object.markdown, str
                ):
                    all_pages_markdown_content.append(page_object.markdown)
                else:
                    print(
                        f"      Cảnh báo: Trang (index: {getattr(page_object, 'index', 'N/A')}) trong file '{base_filename}' không có 'markdown' hợp lệ."
                    )

            if not all_pages_markdown_content:
                print(
                    f"      Lỗi: Không tìm thấy nội dung markdown trong các trang của file '{base_filename}'."
                )
                return False  # Sẽ được xóa ở finally

            final_markdown_content = "\n\n\n\n".join(all_pages_markdown_content)
        else:
            print(
                f"      Lỗi: OCR response cho file '{base_filename}' không có thuộc tính 'pages' hợp lệ hoặc 'pages' rỗng."
            )
            print(
                f"      Chi tiết ocr_response: {vars(ocr_response) if hasattr(ocr_response, '__dict__') else ocr_response}"
            )
            return False  # Sẽ được xóa ở finally

        with open(output_md_path, "w", encoding="utf-8") as f_out:
            f_out.write(final_markdown_content)
        print(f"    OCR thành công. Đã lưu vào: {output_md_path}")
        return True

    except Exception as e:
        print(
            f"    Lỗi nghiêm trọng khi OCR file '{base_filename}' bằng Mistral: {type(e).__name__} - {e}"
        )
        # import traceback; traceback.print_exc() # Bật để debug sâu
        return False
    finally:
        # Luôn cố gắng xóa file đã tải lên Mistral dù thành công hay thất bại (nếu đã có ID)
        if uploaded_file_id:
            try:
                print(
                    f"      Đang dọn dẹp file trên Mistral (ID: {uploaded_file_id})..."
                )
                mistral_client.files.delete(file_id=uploaded_file_id)
                print(
                    f"      Xóa file trên Mistral (ID: {uploaded_file_id}) thành công."
                )
            except Exception as e_del:
                print(
                    f"      Lưu ý: Lỗi khi xóa file trên Mistral (ID: {uploaded_file_id}): {e_del}"
                )


def run_ocr_pipeline_on_folder(
    input_pdf_folder, output_md_folder, mistral_client, ocr_model_name
):
    """
    OCR tất cả các file PDF trong một thư mục bằng Mistral AI
    và lưu kết quả (Markdown) vào các file .md tương ứng trong một thư mục khác.
    """
    if not os.path.isdir(input_pdf_folder):
        print(
            f"LỖI: Thư mục đầu vào '{input_pdf_folder}' không tồn tại hoặc không phải là thư mục."
        )
        return False

    if not os.path.exists(output_md_folder):
        try:
            os.makedirs(output_md_folder)
            print(f"Đã tạo thư mục đầu ra: '{output_md_folder}'")
        except OSError as e:
            print(f"LỖI: Không thể tạo thư mục đầu ra '{output_md_folder}': {e}")
            return False

    pdf_files_found = 0
    successful_ocr_count = 0
    failed_ocr_files = []

    print(f"\nBắt đầu quét thư mục '{input_pdf_folder}' để tìm file PDF cho OCR...")

    for filename in os.listdir(input_pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_files_found += 1
            input_pdf_path = os.path.join(input_pdf_folder, filename)
            base_filename_without_ext = os.path.splitext(filename)[0]
            output_md_file_path = os.path.join(
                output_md_folder, base_filename_without_ext + ".md"
            )

            print(f"\n  Đang xử lý file PDF: {filename} (File {pdf_files_found})...")
            start_time = time.time()

            if ocr_single_pdf_with_mistral(
                input_pdf_path, output_md_file_path, mistral_client, ocr_model_name
            ):
                successful_ocr_count += 1
            else:
                failed_ocr_files.append(filename)

            elapsed_time = time.time() - start_time
            print(
                f"    Hoàn tất xử lý '{filename}'. Thời gian: {elapsed_time:.2f} giây."
            )
            time.sleep(1)  # Khoảng dừng nhỏ giữa các file để tránh rate limit

    print("\n--- HOÀN THÀNH GIAI ĐOẠN OCR BẰNG MISTRAL ---")
    print(f"Tổng số file PDF được tìm thấy: {pdf_files_found}")
    print(f"Số file PDF được OCR thành công: {successful_ocr_count}")
    if failed_ocr_files:
        print(f"Số file PDF OCR thất bại: {len(failed_ocr_files)}")
        print("Danh sách file thất bại:")
        for failed_file in failed_ocr_files:
            print(f"  - {failed_file}")
    elif pdf_files_found > 0:
        print("Tất cả các file PDF đã được xử lý (hoặc đã thử xử lý).")
    else:
        print("Không tìm thấy file PDF nào trong thư mục đầu vào.")
    return successful_ocr_count > 0 or (pdf_files_found > 0 and not failed_ocr_files)


from mistralai import Mistral  # Import Mistral
import base64
from typing import Optional


def ocr_image_content(
    image_bytes: bytes,
    mistral_client: Mistral,
    model_name: str,
    # filename: str = "uploaded_image" # Filename không thực sự cần thiết cho API này
) -> Optional[str]:
    """Thực hiện OCR trên một file ảnh (dưới dạng bytes) bằng Mistral."""
    if not mistral_client:
        print("Lỗi (OCR Service): Mistral client không được cung cấp.")
        return None
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        # Giả sử ảnh là PNG, bạn có thể cần logic để xác định mime type
        image_data_url = f"data:image/png;base64,{base64_image}"

        ocr_response_obj = mistral_client.ocr.process(
            model=model_name,
            document={"type": "image_url", "document_url": image_data_url},
        )

        markdown_content = ""
        if (
            hasattr(ocr_response_obj, "pages")
            and isinstance(ocr_response_obj.pages, list)
            and ocr_response_obj.pages
        ):
            for page_obj in ocr_response_obj.pages:
                if hasattr(page_obj, "markdown") and isinstance(page_obj.markdown, str):
                    markdown_content += (
                        page_obj.markdown + "\n\n"
                    )  # Nối các "trang" nếu ảnh lớn
        return markdown_content.strip() if markdown_content else None
    except Exception as e:
        print(f"Lỗi khi OCR ảnh bằng Mistral (bytes): {e}")
        return None


def ocr_image_from_url(
    image_url: str, mistral_client: Mistral, model_name: str
) -> Optional[str]:
    """Thực hiện OCR trên ảnh từ URL bằng Mistral."""
    if not mistral_client:
        print("Lỗi (OCR Service): Mistral client không được cung cấp.")
        return None
    try:
        ocr_response_obj = mistral_client.ocr.process(
            model=model_name, document={"type": "image_url", "image_url": image_url}
        )
        markdown_content = ""
        if (
            hasattr(ocr_response_obj, "pages")
            and isinstance(ocr_response_obj.pages, list)
            and ocr_response_obj.pages
        ):
            for page_obj in ocr_response_obj.pages:
                if hasattr(page_obj, "markdown") and isinstance(page_obj.markdown, str):
                    markdown_content += page_obj.markdown + "\n\n"
        return markdown_content.strip() if markdown_content else None
    except Exception as e:
        print(f"Lỗi khi OCR ảnh từ URL bằng Mistral: {e}")
        return None

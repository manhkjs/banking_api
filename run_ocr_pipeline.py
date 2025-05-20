# run_ocr_pipeline.py
import os
from dotenv import load_dotenv
from mistralai import Mistral

# Import các hàm và cấu hình từ các module khác
import config  # File config.py ở thư mục gốc
from src.data_processing.ocr_service import (
    run_ocr_pipeline_on_folder,
)  # Import hàm xử lý chính


def main():
    print("Bắt đầu Pipeline OCR tài liệu PDF...")

    # Tải API key của Mistral từ .env
    # load_dotenv() # Đã được gọi trong config.py nếu bạn đặt ở đó, hoặc gọi lại ở đây
    # Hoặc nếu config.py đã load_dotenv() thì không cần gọi lại
    if not os.getenv(config.MISTRAL_API_KEY_ENV_VAR):  # Kiểm tra key đã được load chưa
        load_dotenv(dotenv_path=os.path.join(config.PROJECT_ROOT, ".env"))

    mistral_api_key = os.getenv(config.MISTRAL_API_KEY_ENV_VAR)

    if not mistral_api_key:
        print(
            f"LỖI: Biến môi trường '{config.MISTRAL_API_KEY_ENV_VAR}' chưa được thiết lập trong file .env."
        )
        print("Vui lòng đặt API key của Mistral trước khi chạy.")
        return

    try:
        mistral_client = Mistral(api_key=mistral_api_key)
        # Thử một lệnh nhỏ để xác nhận client hoạt động (ví dụ: list models nếu API hỗ trợ)
        # Hoặc chỉ cần tiếp tục, lỗi sẽ xuất hiện khi gọi API nếu client không đúng
        print("Đã khởi tạo Mistral client thành công.")
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo Mistral client: {e}")
        return

    # Lấy đường dẫn thư mục từ config.py
    input_dir = config.PDF_INPUT_FOLDER
    output_dir = config.OCR_OUTPUT_MD_FOLDER
    ocr_model = config.MISTRAL_OCR_MODEL_NAME

    # Kiểm tra sự tồn tại của thư mục input
    if not os.path.isdir(input_dir):
        print(
            f"LỖI: Thư mục đầu vào '{input_dir}' không tồn tại hoặc không phải là thư mục."
        )
        print("Vui lòng tạo thư mục và đặt file PDF cần OCR vào đó.")
        return

    # Tạo thư mục output nếu chưa có (run_ocr_pipeline_on_folder cũng làm điều này, nhưng kiểm tra ở đây cũng tốt)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Thư mục PDF đầu vào: {input_dir}")
    print(f"Thư mục lưu kết quả OCR (Markdown): {output_dir}")
    print(f"Sử dụng model OCR Mistral: {ocr_model}")

    confirmation = (
        input("Bạn có muốn bắt đầu quá trình OCR không? (yes/no): ").strip().lower()
    )
    if confirmation == "yes" or confirmation == "y":
        run_ocr_pipeline_on_folder(input_dir, output_dir, mistral_client, ocr_model)
    else:
        print("Đã hủy quá trình OCR.")


if __name__ == "__main__":
    # Tạo các file __init__.py nếu chưa có để Python nhận diện là package
    project_root_for_init = os.path.dirname(os.path.abspath(__file__))
    packages_to_initialize = [
        "src",
        "src/data_processing",
        # Thêm các package khác nếu bạn đã tạo và muốn đảm bảo __init__.py
    ]
    for pkg_path_str in packages_to_initialize:
        # pkg_dir_abs = os.path.join(project_root_for_init, *pkg_path_str.split('/'))
        # Sửa lại để xử lý đúng nếu project_root_for_init là thư mục gốc và pkg_path_str bắt đầu bằng "src"
        # Ví dụ, nếu project_root_for_init là /kienlongbank_rag_project và pkg_path_str là "src/data_processing"
        # thì *pkg_path_str.split('/') sẽ là ('src', 'data_processing')
        # os.path.join sẽ nối chúng lại thành /kienlongbank_rag_project/src/data_processing
        path_parts = pkg_path_str.split("/")
        pkg_dir_abs = os.path.join(project_root_for_init, *path_parts)

        if not os.path.exists(pkg_dir_abs):
            os.makedirs(pkg_dir_abs, exist_ok=True)

        init_file = os.path.join(pkg_dir_abs, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass  # Tạo file rỗng

    main()

# src/utils/api_key_manager.py
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv

# Giả sử config.py nằm ở thư mục gốc của dự án
# và utils nằm trong src, thì đường dẫn .env là 2 cấp lên rồi vào .env
# PROJECT_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DOTENV_PATH = os.path.join(PROJECT_ROOT_FOR_ENV, '.env')


class GeminiApiKeyManager:
    def __init__(
        self,
        retry_wait_seconds,
        max_retries_per_key_cycle,
        api_key_prefix="GEMINI_API_KEY_",
    ):
        # Xác định đường dẫn đến file .env từ thư mục gốc của dự án
        # Giả sử thư mục gốc của dự án là thư mục cha của 'src'
        current_script_path = os.path.abspath(
            __file__
        )  # E:\kienlong\banking_ai_platform\src\utils\api_key_manager.py
        src_dir = os.path.dirname(
            os.path.dirname(current_script_path)
        )  # E:\kienlong\banking_ai_platform\src
        project_root = os.path.dirname(src_dir)  # E:\kienlong\banking_ai_platform
        dotenv_path = os.path.join(project_root, ".env")

        # print(f"ApiKeyManager: Đang tìm file .env tại: {dotenv_path}")
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            # print("ApiKeyManager: Đã gọi load_dotenv().")
        else:
            print(
                f"CẢNH BÁO (ApiKeyManager): Không tìm thấy file .env tại {dotenv_path}. API keys có thể không được tải nếu chưa set ở môi trường."
            )

        self.api_key_prefix = api_key_prefix
        self.api_keys = self._load_keys()

        if not self.api_keys:
            raise ValueError(
                f"LỖI (ApiKeyManager): Không tìm thấy API key nào với tiền tố '{self.api_key_prefix}' trong file .env hoặc biến môi trường."
            )

        self.current_key_index = 0
        self.total_keys_exhausted_cycles = 0
        self.retry_wait_seconds = retry_wait_seconds
        self.max_retries_per_key_cycle = max_retries_per_key_cycle
        # print(f"GeminiApiKeyManager: Đã tải {len(self.api_keys)} API keys (tiền tố: {self.api_key_prefix}).")
        # if hasattr(genai, '__version__'):
        #      print(f"Phiên bản google-generativeai đang được sử dụng bởi ApiKeyManager: {genai.__version__}")

    def _load_keys(self):
        keys = []
        i = 1
        # print(f"  ApiKeyManager: Bắt đầu vòng lặp lấy API keys với tiền tố '{self.api_key_prefix}'...")
        while True:
            key_name = f"{self.api_key_prefix}{i}"
            key = os.getenv(key_name)
            if key:
                keys.append(key)
                # print(f"    Tìm thấy key: {key_name}")
                i += 1
            else:
                if i == 1:  # Nếu ngay cả key đầu tiên cũng không có
                    print(f"    Không tìm thấy key đầu tiên: {key_name}")
                # print(f"    Dừng vòng lặp tìm key ở {key_name} vì không có giá trị.")
                break
        return keys

    def _execute_with_retry(self, api_call_logic_func, call_type):
        """Hàm nội bộ để thực hiện logic gọi API với retry và xoay vòng key."""
        keys_tried_in_current_cycle = 0
        while True:
            current_api_key = self.api_keys[self.current_key_index]
            # print(f"    Sử dụng API Key #{self.current_key_index + 1} cho {call_type}...")
            try:
                genai.configure(api_key=current_api_key)
                response = api_call_logic_func()  # Thực thi logic gọi API cụ thể
                # print(f"    {call_type} thành công với Key #{self.current_key_index + 1}.")
                self.total_keys_exhausted_cycles = 0  # Reset khi thành công
                return response

            except (
                genai.types.generation_types.BlockedPromptException,
                genai.types.generation_types.StopCandidateException,
            ) as e_block:
                print(
                    f"    Cảnh báo/Lỗi (Blocked/Stop) với Key #{self.current_key_index + 1} ({call_type}): {type(e_block).__name__} - {e_block.args if hasattr(e_block, 'args') else e_block}"
                )
            except Exception as e_api:
                error_str = str(e_api).lower()
                if (
                    "429" in error_str
                    or "resource_exhausted" in error_str
                    or "quota" in error_str
                ):
                    print(
                        f"    Lỗi Quota với Key #{self.current_key_index + 1} ({call_type})."
                    )
                else:
                    print(
                        f"    Lỗi không xác định với Key #{self.current_key_index + 1} ({call_type}): {type(e_api).__name__} - {e_api}"
                    )

            keys_tried_in_current_cycle += 1
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

            if keys_tried_in_current_cycle >= len(self.api_keys):
                self.total_keys_exhausted_cycles += 1
                print(
                    f"    Tất cả {len(self.api_keys)} API key đã gặp lỗi/quota {self.total_keys_exhausted_cycles} lần cho {call_type}."
                )
                if self.total_keys_exhausted_cycles >= self.max_retries_per_key_cycle:
                    print(
                        f"    Đã thử {self.max_retries_per_key_cycle} chu kỳ cho {call_type}. Bỏ qua yêu cầu này."
                    )
                    return None
                print(
                    f"    Đang đợi {self.retry_wait_seconds} giây trước khi thử lại chu kỳ mới cho {call_type}..."
                )
                time.sleep(self.retry_wait_seconds)
                keys_tried_in_current_cycle = 0

    def execute_generative_call(
        self, model_name_to_use, api_params_for_method, call_type="Generation"
    ):
        """
        Thực hiện một lệnh gọi đến phương thức generate_content của GenerativeModel.
        api_params_for_method là dict chứa các tham số cho generate_content (ví dụ: contents, generation_config,...).
        """

        def api_logic():
            model = genai.GenerativeModel(model_name_to_use)
            return model.generate_content(**api_params_for_method)

        # print(f"    (ApiKeyManager) Chuẩn bị gọi execute_generative_call cho model {model_name_to_use}...")
        return self._execute_with_retry(api_logic, call_type)

    def call_embedding_model(
        self, model_name, content_to_embed, task_type, call_type="Embedding"
    ):
        """Thực hiện gọi API embedding của Gemini."""

        def api_logic():
            return genai.embed_content(
                model=model_name,
                content=content_to_embed,  # content có thể là string hoặc list of strings
                task_type=task_type,
            )

        # print(f"    (ApiKeyManager) Chuẩn bị gọi call_embedding_model cho model {model_name}...")
        return self._execute_with_retry(api_logic, call_type)

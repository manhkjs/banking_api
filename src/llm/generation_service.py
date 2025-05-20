# src/llm/generation_service.py
import google.generativeai as genai  # Cần cho GenerationConfig

# from src.utils.api_key_manager import GeminiApiKeyManager # Nhận instance của manager
# import config # Để lấy model name, bank info, prompt guidelines


def generate_chatbot_response(
    user_query: str,
    compiled_context: str,
    gemini_generation_model_name: str,  # Tên model từ config
    api_manager,  # Instance của GeminiApiKeyManager
    bank_homepage_url: str,  # từ config
    bank_contact_info: str,  # từ config
    generation_prompt_guidelines: str,  # từ config
) -> str:
    """
    Xây dựng prompt và gọi Gemini để sinh câu trả lời.
    """
    print("  Đang tạo prompt cho Gemini generation...")

    fallback_text = (
        f"Rất tiếc, hiện tại tôi chưa thể cung cấp thông tin chính xác và đầy đủ nhất về nội dung này. "
        f"Để được hỗ trợ cụ thể và cập nhật, bạn vui lòng truy cập trang web chính thức của Kienlongbank tại {bank_homepage_url} "
        f"hoặc liên hệ trực tiếp với chúng tôi qua {bank_contact_info} nhé!"
    )

    # Sử dụng f-string để chèn fallback_text vào guidelines nếu nó là một placeholder
    # Hoặc bạn có thể truyền fallback_text như một biến riêng vào prompt chính
    final_guidelines = generation_prompt_guidelines.replace(
        "{fallback_text_for_prompt}", fallback_text
    )
    # Và đảm bảo các placeholder khác như {BANK_HOMEPAGE_URL}, {BANK_CONTACT_INFO} cũng được thay thế trong guidelines
    final_guidelines = final_guidelines.replace(
        "{BANK_HOMEPAGE_URL}", bank_homepage_url
    )
    final_guidelines = final_guidelines.replace(
        "{BANK_CONTACT_INFO}", bank_contact_info
    )

    generation_prompt = f"""
Bạn là Trợ lý ảo của Ngân hàng Kienlongbank – một chuyên viên chăm sóc khách hàng thân thiện, chuyên nghiệp và tận tâm.
Nhiệm vụ của bạn là hỗ trợ người dùng một cách dễ hiểu, tự nhiên, giống như một nhân viên ngân hàng thật sự đang trò chuyện trực tiếp với khách hàng.

CÂU HỎI CỦA NGƯỜI DÙNG:
{user_query}

THÔNG TIN THAM KHẢO TỪ CƠ SỞ DỮ LIỆU NỘI BỘ (phần này có thể trống hoặc thông tin có thể không hoàn toàn liên quan đến câu hỏi):
---
{compiled_context if compiled_context else "(Không có thông tin cụ thể nào được tìm thấy trong cơ sở dữ liệu nội bộ cho câu hỏi này.)"}
---

NGUYÊN TẮC VÀ HƯỚNG DẪN TRẢ LỜI BẮT BUỘC CHO BẠN:
{final_guidelines}

TRẢ LỜI:
"""
    print("  Đang sinh câu trả lời từ Gemini...")
    generation_params = {
        "contents": generation_prompt,
        "generation_config": genai.types.GenerationConfig(
            temperature=0.5, max_output_tokens=2048
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

    # Gọi qua ApiKeyManager, truyền tên model và các params
    llm_response = api_manager.execute_generative_call(  # Giả sử phương thức này trong ApiKeyManager
        model_name_to_use=gemini_generation_model_name,
        api_params_for_method=generation_params,
        call_type="Generating Answer",
    )

    if llm_response and llm_response.parts:
        return llm_response.text
    elif (
        llm_response
        and hasattr(llm_response, "prompt_feedback")
        and llm_response.prompt_feedback
    ):
        print(f"    LLM response bị chặn: {llm_response.prompt_feedback}")
        return (
            f"Xin lỗi, yêu cầu của bạn không thể được xử lý vào lúc này. "
            f"Bạn có thể thử diễn đạt lại câu hỏi hoặc tham khảo trang web {bank_homepage_url} / liên hệ {bank_contact_info}."
        )
    else:
        # Trả về fallback nếu có lỗi nghiêm trọng hoặc không có response parts
        return fallback_text  # Hoặc một thông báo lỗi chung hơn

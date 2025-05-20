from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, status
from typing import Optional
from mistralai import Mistral  # Cho type hinting

from src.api import models as api_models  # Pydantic models
from src.api import dependencies  # Dependencies
import config  # Config chung
from src.data_processing import ocr_service  # Service OCR

router = APIRouter(prefix="/ocr", tags=["OCR Operations"])


@router.post("/image/upload", response_model=api_models.OcrResponse)
async def ocr_image_upload_endpoint(
    file: UploadFile = File(..., description="File ảnh cần OCR (PNG, JPG, WEBP, etc.)"),
    mistral_client: Optional[Mistral] = Depends(dependencies.get_mistral_client),
):
    if not mistral_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dịch vụ OCR (Mistral) chưa sẵn sàng.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File không hợp lệ. Vui lòng tải lên một file ảnh.",
        )

    try:
        image_bytes = await file.read()
        extracted_text = ocr_service.ocr_image_content(
            image_bytes, mistral_client, config.MISTRAL_OCR_MODEL_NAME
        )
        if (
            extracted_text is None
        ):  # Bao gồm cả trường hợp trả về chuỗi rỗng cũng coi là lỗi
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Xử lý OCR thất bại hoặc không có nội dung text.",
            )
        return api_models.OcrResponse(
            extracted_text=extracted_text, source_type="upload", file_name=file.filename
        )
    except HTTPException as http_exc:  # Re-raise HTTPException
        raise http_exc
    except Exception as e:
        # import traceback; traceback.print_exc() # Để debug
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình OCR: {str(e)}",
        )


@router.post("/image/url", response_model=api_models.OcrResponse)
async def ocr_image_url_endpoint(
    request: api_models.OcrUrlRequest,
    mistral_client: Optional[Mistral] = Depends(dependencies.get_mistral_client),
):
    if not mistral_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dịch vụ OCR (Mistral) chưa sẵn sàng.",
        )
    try:
        extracted_text = ocr_service.ocr_image_from_url(
            request.image_url, mistral_client, config.MISTRAL_OCR_MODEL_NAME
        )
        if (
            extracted_text is None
        ):  # Bao gồm cả trường hợp trả về chuỗi rỗng cũng coi là lỗi
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Xử lý OCR thất bại hoặc không có nội dung text.",
            )
        return api_models.OcrResponse(extracted_text=extracted_text, source_type="url")
    except HTTPException as http_exc:  # Re-raise HTTPException
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình OCR từ URL: {str(e)}",
        )

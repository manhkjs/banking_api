# src/api/endpoints/chat_router.py
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict
import uuid
import networkx as nx
from qdrant_client import QdrantClient

from src.api import models as api_models
from src.api import dependencies
import config
from src.utils.api_key_manager import GeminiApiKeyManager
from src.embedding.embed_querry import embed_query_gemini
from src.retrieval.retrieval_service import retrieve_and_compile_context
from src.llm.generation_service import generate_chatbot_response
from src.reranking.reranker import Reranker


router = APIRouter(prefix="/chat", tags=["Chatbot RAG"])

# Biến lưu trữ lịch sử hội thoại (đơn giản, trong bộ nhớ, tương tự như trước)
# Cần một giải pháp tốt hơn cho production
conversation_histories: Dict[str, List[Dict[str, str]]] = {}


@router.post("", response_model=api_models.ChatbotResponse)  # Đổi tên model response
async def chat_endpoint(
    payload: api_models.ChatQuery,  # Sử dụng ChatQuery model
    api_manager: GeminiApiKeyManager = Depends(dependencies.get_gemini_api_manager),
    knowledge_graph: nx.DiGraph = Depends(dependencies.get_knowledge_graph),
    qdrant_cli: QdrantClient = Depends(dependencies.get_qdrant_client),
    reranker_instance: Optional[Reranker] = Depends(dependencies.get_reranker),
):
    print("!!!!!! DEBUG: ĐÃ VÀO ĐƯỢC HÀM chat_endpoint !!!!!!")  # <--- THÊM DÒNG NÀY
    print(f"!!!!!! DEBUG: Payload nhận được: {payload.model_dump_json(indent=2)}")
    print(f"API Endpoint /chat: Nhận được câu hỏi: '{payload.query}'")

    # --- Quản lý Conversation ID và Lịch sử ---
    conversation_id = payload.conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    current_session_history = conversation_histories.get(conversation_id, [])
    formatted_history_for_prompt = ""
    if current_session_history:
        history_to_format = current_session_history[
            -(getattr(config, "MAX_CONVERSATION_HISTORY_TURNS", 3) * 2) :
        ]
        history_parts = []
        for (
            turn
        ) in (
            history_to_format
        ):  # Giả sử turn là {"role": "user/assistant", "content": "..."}
            role_label = (
                "Khách hàng" if turn["role"] == "user" else "Trợ lý Kienlongbank"
            )
            history_parts.append(f"{role_label}: {turn['content']}")
        formatted_history_for_prompt = "\n".join(history_parts)

    # 1. Embedding câu hỏi
    query_vector = embed_query_gemini(
        user_query=payload.query,
        api_manager=api_manager,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        task_type=config.EMBEDDING_TASK_TYPE_QUERY,
    )
    if not query_vector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi embedding câu hỏi.",
        )

    # 2. Truy xuất ngữ cảnh
    initial_retrieval_limit_chat = config.QDRANT_SEARCH_LIMIT
    final_top_n_chat = config.RERANK_TOP_N

    # Điều chỉnh logic lấy limit cho chat nếu cần (tương tự search)
    if config.RERANKER_ACTIVE and reranker_instance and reranker_instance.model:
        if initial_retrieval_limit_chat < final_top_n_chat:
            initial_retrieval_limit_chat = final_top_n_chat * 2
    else:
        initial_retrieval_limit_chat = final_top_n_chat

    compiled_context, context_parts_for_display = retrieve_and_compile_context(
        original_query=payload.query,
        query_vector=query_vector,
        qdrant_cli=qdrant_cli,
        knowledge_graph=knowledge_graph,
        reranker=reranker_instance if config.RERANKER_ACTIVE else None,
        qdrant_collection_name=config.QDRANT_COLLECTION_NAME,
        qdrant_search_limit=initial_retrieval_limit_chat,
        reranker_active=config.RERANKER_ACTIVE,
        rerank_top_n=final_top_n_chat,
    )

    # Log context (tùy chọn)
    # ...

    # 3. Sinh câu trả lời
    answer = generate_chatbot_response(
        user_query=payload.query,
        compiled_context=compiled_context,
        # conversation_history_str=formatted_history_for_prompt,
        gemini_generation_model_name=config.GENERATION_MODEL_NAME,
        api_manager=api_manager,
        bank_homepage_url=config.BANK_HOMEPAGE_URL,
        bank_contact_info=config.BANK_CONTACT_INFO,
        generation_prompt_guidelines=config.GENERATION_PROMPT_GUIDELINES,
    )

    if not answer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi sinh câu trả lời từ LLM.",
        )

    # Cập nhật lịch sử hội thoại
    conversation_histories.setdefault(conversation_id, []).append(
        {"role": "user", "content": payload.query}
    )
    conversation_histories[conversation_id].append(
        {"role": "assistant", "content": answer}
    )
    max_hist_len = getattr(config, "MAX_CONVERSATION_HISTORY_TURNS", 3) * 2
    if len(conversation_histories[conversation_id]) > max_hist_len:
        conversation_histories[conversation_id] = conversation_histories[
            conversation_id
        ][-max_hist_len:]

    return api_models.ChatbotResponse(  # Sử dụng ChatbotResponse model
        answer=answer,
        retrieved_sources=[
            api_models.RetrievedSource(**item) for item in context_parts_for_display
        ],
        conversation_id=conversation_id,
    )

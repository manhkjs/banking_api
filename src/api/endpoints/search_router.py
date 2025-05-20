# src/api/endpoints/search_router.py
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
import networkx as nx
from qdrant_client import QdrantClient

from src.api import models as api_models
from src.api import dependencies
import config
from src.utils.api_key_manager import GeminiApiKeyManager
from src.embedding.embed_querry import embed_query_gemini
from src.retrieval.retrieval_service import retrieve_and_compile_context
from src.reranking.reranker import Reranker


router = APIRouter(prefix="/search", tags=["Document Search"])


@router.post("/documents", response_model=List[api_models.RetrievedSource])
async def search_documents_endpoint(
    payload: api_models.DocumentSearchQuery,
    api_manager: GeminiApiKeyManager = Depends(dependencies.get_gemini_api_manager),
    knowledge_graph: nx.DiGraph = Depends(dependencies.get_knowledge_graph),
    qdrant_cli: QdrantClient = Depends(dependencies.get_qdrant_client),
    reranker_instance: Optional[Reranker] = Depends(
        dependencies.get_reranker
    ),  # Sửa tên hàm dependency
):
    print(
        f"API Endpoint /search/documents: Query: '{payload.query}', top_k: {payload.top_k}"
    )

    query_vector = embed_query_gemini(
        user_query=payload.query,
        api_manager=api_manager,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        task_type=config.EMBEDDING_TASK_TYPE_QUERY,
    )
    if not query_vector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi embedding câu hỏi tìm kiếm.",
        )

    # qdrant_search_limit sẽ là số lượng lấy từ Qdrant ban đầu
    # rerank_top_n (trong config) sẽ là số lượng cuối cùng sau khi rerank
    # payload.top_k là số lượng client muốn nhận cuối cùng

    # Nếu reranking active, lấy nhiều hơn từ Qdrant để reranker có dữ liệu làm việc
    initial_retrieval_limit = config.QDRANT_SEARCH_LIMIT
    final_top_n_after_rerank = (
        payload.top_k
    )  # Client muốn nhận bao nhiêu kết quả cuối cùng

    if config.RERANKER_ACTIVE and reranker_instance and reranker_instance.model:
        # Đảm bảo initial_retrieval_limit đủ lớn cho reranker, và final_top_n không lớn hơn nó
        if initial_retrieval_limit < final_top_n_after_rerank:
            initial_retrieval_limit = final_top_n_after_rerank * 2  # Ví dụ: lấy gấp đôi
        print(
            f"  Tìm kiếm ban đầu với limit: {initial_retrieval_limit}, sau rerank sẽ lấy top: {final_top_n_after_rerank}"
        )
    else:
        initial_retrieval_limit = (
            final_top_n_after_rerank  # Không rerank, lấy đúng số lượng client yêu cầu
        )
        print(f"  Tìm kiếm với limit: {initial_retrieval_limit} (không rerank)")

    _compiled_llm_context, context_parts_for_display = retrieve_and_compile_context(
        original_query=payload.query,
        query_vector=query_vector,
        qdrant_cli=qdrant_cli,
        knowledge_graph=knowledge_graph,
        reranker=(
            reranker_instance if config.RERANKER_ACTIVE else None
        ),  # Chỉ truyền nếu active
        qdrant_collection_name=config.QDRANT_COLLECTION_NAME,
        qdrant_search_limit=initial_retrieval_limit,
        reranker_active=config.RERANKER_ACTIVE,
        rerank_top_n=final_top_n_after_rerank,
    )

    if not context_parts_for_display:
        return []

    # Chuyển đổi sang Pydantic model (đảm bảo các key khớp)
    response_sources = [
        api_models.RetrievedSource(**item) for item in context_parts_for_display
    ]
    return response_sources

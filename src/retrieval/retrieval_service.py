# src/retrieval/retrieval_service.py
import networkx as nx
from src.vector_store.qdrant_service import search_qdrant_collection
from src.reranking.reranker import Reranker  # << IMPORT MỚI
from typing import List, Dict, Any, Optional  # << IMPORT TYPE HINTING

# import config # Các hằng số sẽ được truyền vào từ chatbot_cli.py


def retrieve_and_compile_context(
    original_query: str,  # << THÊM original_query
    query_vector: list[float],
    qdrant_cli,
    knowledge_graph: nx.DiGraph,
    reranker: Optional[Reranker],  # << THÊM reranker instance (có thể là None)
    qdrant_collection_name: str,
    qdrant_search_limit: int,
    reranker_active: bool,  # << THÊM cờ kích hoạt reranker
    rerank_top_n: int,  # << THÊM số lượng top N sau rerank
):
    """
    Truy xuất, (tùy chọn) rerank, và tổng hợp ngữ cảnh.
    """
    print(
        f"  Đang tìm kiếm trên Qdrant (collection: {qdrant_collection_name}, top {qdrant_search_limit} kết quả)..."
    )
    search_hits = search_qdrant_collection(
        qdrant_cli, qdrant_collection_name, query_vector, qdrant_search_limit
    )

    if not search_hits:
        print("  Không có kết quả tìm kiếm nào từ Qdrant cho câu hỏi này.")
        return "", []  # Trả về context rỗng và list rỗng

    # Chuẩn bị danh sách các document ứng viên từ Qdrant hits
    candidate_documents = []
    for hit in search_hits:
        payload = hit.payload
        original_text_content = payload.get("original_text", "").strip()
        if original_text_content:  # Chỉ thêm nếu có nội dung text
            candidate_documents.append(
                {
                    "qdrant_id": str(
                        hit.id
                    ),  # Chuyển ID Qdrant sang str nếu nó là UUID object
                    "score": float(hit.score),  # Điểm từ Qdrant
                    "original_text": original_text_content,
                    "graph_node_id": payload.get("graph_node_id"),
                    "document_name": payload.get(
                        "document_name", "Tài liệu không xác định"
                    ),
                    "node_type": payload.get("node_type", "Nội dung"),
                    # Mang theo các payload khác nếu cần thiết cho bước sau
                }
            )

    if not candidate_documents:
        print(
            "  Không có tài liệu hợp lệ nào từ Qdrant (sau khi lọc text rỗng) để xử lý tiếp."
        )
        return "", []

    # Thực hiện Reranking nếu được kích hoạt và reranker đã được khởi tạo
    final_documents_for_context = []
    if reranker_active and reranker and reranker.model:
        print(f"  Thực hiện reranking cho {len(candidate_documents)} ứng viên...")
        # text_key="original_text" vì đó là trường chứa nội dung để reranker so sánh
        reranked_docs = reranker.rerank(
            original_query,
            candidate_documents,
            text_key="original_text",
            top_n=rerank_top_n,
        )
        final_documents_for_context = reranked_docs
        print(
            f"  Đã chọn top {len(final_documents_for_context)} tài liệu sau khi rerank."
        )
    else:
        if reranker_active and (not reranker or not reranker.model):
            print(
                "  CẢNH BÁO: Reranking được kích hoạt nhưng model reranker không khả dụng. Sử dụng kết quả gốc từ Qdrant."
            )
        # Nếu không rerank, lấy top_n từ kết quả Qdrant (đã được sắp xếp theo score từ Qdrant)
        final_documents_for_context = candidate_documents[:rerank_top_n]
        # rerank_top_n ở đây đóng vai trò là số lượng context cuối cùng muốn lấy

    # Xây dựng ngữ cảnh từ final_documents_for_context
    print(
        f"  Đang xây dựng ngữ cảnh từ {len(final_documents_for_context)} kết quả cuối cùng và KG..."
    )
    compiled_context_for_llm = ""
    context_parts_for_display = []
    unique_texts_for_context = set()

    for doc_data in final_documents_for_context:
        original_text = doc_data.get("original_text", "")  # Đã strip ở trên
        graph_node_id = doc_data.get("graph_node_id")
        doc_name = doc_data.get("document_name")
        node_type = doc_data.get("node_type")
        # Lấy điểm: ưu tiên _rerank_score, nếu không có thì lấy score từ Qdrant
        final_score = doc_data.get("_rerank_score", doc_data.get("score", 0.0))

        if not original_text or original_text in unique_texts_for_context:
            continue
        unique_texts_for_context.add(original_text)

        llm_detail = f'Trích dẫn từ tài liệu \'{doc_name}\' (Loại: {node_type}, Điểm liên quan: {final_score:.4f}):\n"""\n{original_text}\n"""'
        display_detail = {
            "source": doc_name,
            "type": node_type,
            "score": final_score,
            "content_snippet": (
                original_text[:300] + "..."
                if len(original_text) > 300
                else original_text
            ),
        }

        if knowledge_graph.has_node(graph_node_id):
            kg_node_data = knowledge_graph.nodes[graph_node_id]
            if node_type == "Chunk":
                doc_id_of_chunk = kg_node_data.get("source_document_id")
                if doc_id_of_chunk and knowledge_graph.has_node(doc_id_of_chunk):
                    doc_kg_data = knowledge_graph.nodes[doc_id_of_chunk]
                    doc_summary = doc_kg_data.get("summary")
                    if (
                        doc_summary
                        and doc_summary not in original_text
                        and len(doc_summary) < 300
                    ):
                        llm_detail += (
                            f"\n(Tóm tắt của tài liệu '{doc_name}': {doc_summary})"
                        )
                        display_detail["document_summary"] = doc_summary
            elif (
                node_type == "DocumentSummary"
            ):  # Giả sử DocumentSummary là loại node từ KG
                keywords = kg_node_data.get("keywords")
                if keywords:
                    llm_detail += f"\n(Các từ khóa liên quan của tài liệu: {keywords})"
                    display_detail["document_keywords"] = keywords

        context_parts_for_display.append(display_detail)
        compiled_context_for_llm += llm_detail + "\n\n---\n\n"

    compiled_context_for_llm = compiled_context_for_llm.strip().strip("---")

    return compiled_context_for_llm, context_parts_for_display

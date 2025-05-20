# src/reranking/reranker_service.py
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any
import time  # Để đo thời gian (tùy chọn)


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Khởi tạo Reranker với một model cross-encoder cụ thể.

        Args:
            model_name (str): Tên của model cross-encoder từ Hugging Face Hub.
                              Ví dụ: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            device (str): Thiết bị để chạy model ('cpu', 'cuda' nếu có).
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        try:
            print(
                f"Thông tin (Reranker): Đang tải model '{self.model_name}' trên '{self.device}'..."
            )
            start_time = time.time()
            self.model = CrossEncoder(self.model_name, device=self.device)
            load_time = time.time() - start_time
            print(
                f"Thông tin (Reranker): Model '{self.model_name}' đã tải thành công sau {load_time:.2f} giây."
            )
        except Exception as e:
            print(
                f"LỖI (Reranker): Không thể tải model reranker '{self.model_name}'. Lỗi: {e}"
            )
            print("Reranking sẽ không hoạt động.")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "original_text",
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Sắp xếp lại danh sách các document dựa trên độ liên quan của chúng với query.

        Args:
            query (str): Câu hỏi gốc của người dùng.
            documents (List[Dict[str, Any]]): Danh sách các dictionary, mỗi dict đại diện
                                             cho một document/chunk và phải chứa trường `text_key`.
            text_key (str): Tên trường trong mỗi dict document chứa nội dung văn bản cần rerank.
            top_n (int, optional): Số lượng document hàng đầu cần trả về sau khi rerank.
                                   Nếu None, trả về tất cả các document đã rerank.

        Returns:
            List[Dict[str, Any]]: Danh sách các document đã được sắp xếp lại theo điểm rerank.
        """
        if not self.model:
            print(
                "CẢNH BÁO (Reranker): Model chưa được tải. Trả về danh sách document gốc."
            )
            return documents[:top_n] if top_n is not None else documents
        if not documents:
            return []
        if not query:
            print("CẢNH BÁO (Reranker): Query rỗng. Trả về danh sách document gốc.")
            return documents[:top_n] if top_n is not None else documents

        # Tạo các cặp (query, document_text)
        pairs = []
        valid_documents_for_scoring = (
            []
        )  # Lưu các document có text hợp lệ để ghép lại sau

        for doc in documents:
            doc_text = doc.get(text_key)
            if isinstance(doc_text, str) and doc_text.strip():
                pairs.append([query, doc_text])
                valid_documents_for_scoring.append(doc)
            else:
                print(
                    f"CẢNH BÁO (Reranker): Bỏ qua document không có text hợp lệ ở key '{text_key}': {str(doc)[:100]}..."
                )

        if not pairs:
            print(
                "CẢNH BÁO (Reranker): Không có cặp (query, document_text) hợp lệ để rerank."
            )
            return (
                documents[:top_n] if top_n is not None else documents
            )  # Hoặc trả về []

        print(
            f"  Thông tin (Reranker): Đang tính điểm rerank cho {len(pairs)} cặp (query, document)..."
        )
        try:
            scores = self.model.predict(
                pairs, show_progress_bar=False
            )  # show_progress_bar=True nếu muốn thấy tiến trình
        except Exception as e_predict:
            print(f"  LỖI (Reranker): Lỗi khi tính điểm rerank: {e_predict}")
            # Trong trường hợp lỗi, trả về danh sách gốc (chưa rerank) để không làm gián đoạn pipeline
            return documents[:top_n] if top_n is not None else documents

        # Kết hợp scores với các document hợp lệ ban đầu
        # Chỉ những document có trong valid_documents_for_scoring mới có score
        scored_documents = []
        for i in range(len(scores)):
            doc_data = valid_documents_for_scoring[
                i
            ].copy()  # Tạo bản copy để không thay đổi dict gốc
            doc_data["_rerank_score"] = float(scores[i])
            scored_documents.append(doc_data)

        # Sắp xếp các document theo điểm rerank giảm dần
        reranked_documents = sorted(
            scored_documents, key=lambda x: x["_rerank_score"], reverse=True
        )

        print(
            f"  Thông tin (Reranker): Đã sắp xếp lại {len(reranked_documents)} tài liệu."
        )

        if top_n is not None:
            return reranked_documents[:top_n]
        return reranked_documents

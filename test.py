import networkx as nx
import os

# Import config để lấy đường dẫn file KG
import config  # Giả sử config.py nằm cùng cấp hoặc trong PYTHONPATH

# Import hàm tải KG từ module bạn đã tạo
# Đảm bảo cấu trúc thư mục và __init__.py cho phép import này
# Nếu run_kg_builder_pipeline.py và inspect_kg_max_chunks.py cùng ở thư mục gốc,
# và src là một package, cách import sẽ là:
from src.knowledge_graph.kg_loader_service import load_nx_graph_from_file


def find_document_with_most_chunks(graph: nx.DiGraph):
    """
    Tìm nút Document có nhiều nút Chunk con nhất trong đồ thị.

    Args:
        graph (nx.DiGraph): Đối tượng đồ thị NetworkX.

    Returns:
        tuple: (node_id_cua_document, so_luong_chunks) hoặc (None, 0) nếu không tìm thấy.
    """
    if not graph:
        return None, 0

    max_chunks = -1
    doc_with_max_chunks_id = None
    doc_node_attributes = None

    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "Document":
            current_chunk_count = 0
            # Đếm số lượng chunk con trực tiếp
            # G.successors(node_id) trả về các nút mà node_id trỏ tới
            for successor_id in graph.successors(node_id):
                # Kiểm tra xem cạnh có phải là HAS_CHUNK và nút kế tiếp có phải là Chunk không
                # Trong DiGraph, G[node_id][successor_id] trả về thuộc tính của cạnh đầu tiên (nếu không phải MultiDiGraph)
                edge_data = graph.get_edge_data(node_id, successor_id)
                # Nếu có nhiều loại cạnh từ Document, bạn cần kiểm tra cụ thể edge_data.get('type')
                # Với cấu trúc hiện tại, chúng ta giả định tất cả các successor của Document là Chunk qua HAS_CHUNK
                if graph.nodes[successor_id].get("type") == "Chunk" and (
                    edge_data and edge_data.get("type") == "HAS_CHUNK"
                ):  # Đảm bảo cạnh đúng là HAS_CHUNK
                    current_chunk_count += 1

            if current_chunk_count > max_chunks:
                max_chunks = current_chunk_count
                doc_with_max_chunks_id = node_id
                doc_node_attributes = data

    return doc_with_max_chunks_id, max_chunks, doc_node_attributes


def print_document_and_chunks_info(
    graph: nx.DiGraph, doc_id: str, num_chunks_to_show=5
):
    """
    In thông tin của một Document node và một vài Chunk node con của nó.
    """
    if not graph.has_node(doc_id):
        print(f"Lỗi: Không tìm thấy Document Node với ID '{doc_id}' trong đồ thị.")
        return

    doc_data = graph.nodes[doc_id]
    print(f"\n--- Thông tin Document Node ---")
    print(f"ID: {doc_id}")
    for key, value in doc_data.items():
        if key == "summary" and len(value) > 150:
            print(f"  {key.capitalize()}: {value[:150].strip()}...")
        else:
            print(f"  {key.capitalize()}: {value}")

    print(
        f"\n--- Các Chunk con của Document '{doc_data.get('name', doc_id)}' (hiển thị tối đa {num_chunks_to_show} chunks) ---"
    )
    chunk_count = 0
    for successor_id in graph.successors(doc_id):
        if chunk_count >= num_chunks_to_show:
            print(f"    ... và còn nhiều chunk khác.")
            break

        successor_data = graph.nodes[successor_id]
        edge_data = graph.get_edge_data(doc_id, successor_id)

        if successor_data.get("type") == "Chunk" and (
            edge_data and edge_data.get("type") == "HAS_CHUNK"
        ):
            chunk_count += 1
            print(f"\n  Chunk #{successor_data.get('order_in_doc', 'N/A')}:")
            print(f"    ID Chunk: {successor_id}")
            chunk_text = successor_data.get("text_content", "")
            print(
                f"    Nội dung (200 ký tự đầu): {chunk_text[:200].strip().replace(os.linesep, ' ')}..."
            )

    if chunk_count == 0:
        print(
            "    Document này không có chunk con nào được liên kết bằng cạnh HAS_CHUNK."
        )


if __name__ == "__main__":
    print("Bắt đầu kiểm tra Knowledge Graph...")

    # 1. Tải Knowledge Graph từ file đã lưu
    # Sử dụng đường dẫn từ config.py
    graph_file = config.KG_GRAPHML_OUTPUT_FILE
    # Hoặc nếu bạn muốn hardcode để test nhanh:
    # graph_file = "document_knowledge_graph_1.graphml" # Đảm bảo file này ở cùng thư mục với script

    knowledge_graph = load_nx_graph_from_file(graph_file)

    if knowledge_graph:
        # 2. Tìm Document Node có nhiều Chunk con nhất
        doc_id, num_chunks, doc_attributes = find_document_with_most_chunks(
            knowledge_graph
        )

        if doc_id:
            print(f"\n--- KẾT QUẢ PHÂN TÍCH ---")
            print(f"Document Node có nhiều Chunk con nhất là:")
            print(f"  ID: {doc_id}")
            print(f"  Tên file gốc: {doc_attributes.get('original_filename', 'N/A')}")
            print(f"  Tên tài liệu (name): {doc_attributes.get('name', 'N/A')}")
            print(f"  Số lượng Chunk con: {num_chunks}")

            # 3. In thông tin chi tiết hơn về Document đó và một vài chunk của nó
            print_document_and_chunks_info(
                knowledge_graph, doc_id, num_chunks_to_show=5
            )

            # 4. Thống kê các loại nút (tùy chọn, như script trước)
            print("\n--- THỐNG KÊ CÁC LOẠI NÚT TRONG ĐỒ THỊ ---")
            node_types = {}
            for _, data in knowledge_graph.nodes(data=True):
                node_type = data.get("type", "Không xác định")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            for n_type, count in node_types.items():
                print(f"Loại nút '{n_type}': {count} nút")

        else:
            print(
                "Không tìm thấy Document Node nào trong đồ thị hoặc không có Document Node nào có Chunk con."
            )
    else:
        print("Không thể thực hiện phân tích do không tải được đồ thị.")

    print("\nHoàn thành kiểm tra.")

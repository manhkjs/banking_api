# src/knowledge_graph/kg_loader_service.py
import os
import networkx as nx


def load_nx_graph_from_file(graph_file_path: str) -> nx.DiGraph | None:
    """
    Tải đối tượng đồ thị NetworkX từ file GraphML.
    """
    if not os.path.exists(graph_file_path):
        print(f"LỖI (kg_loader): File đồ thị '{graph_file_path}' không tồn tại.")
        return None
    try:
        print(f"Đang tải đồ thị từ file: {graph_file_path}...")
        graph = nx.read_graphml(graph_file_path)
        print(
            f"Tải đồ thị thành công với {graph.number_of_nodes()} nút và {graph.number_of_edges()} cạnh."
        )
        return graph
    except Exception as e:
        print(f"Lỗi khi tải đồ thị từ file '{graph_file_path}': {e}")
        return None

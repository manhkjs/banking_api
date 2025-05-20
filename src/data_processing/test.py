import networkx as nx
import os
import matplotlib.pyplot as plt  # Thêm import này ở đầu nếu chưa có

# --- CẤU HÌNH TÊN FILE ĐỒ THỊ ---
# Đảm bảo tên file này khớp với tên file bạn đã lưu đồ thị ở script trước
# (build_document_knowledge_graph.py)
GRAPH_FILE_TO_LOAD = "document_knowledge_graph.graphml"


# --- HÀM TẢI ĐỒ THỊ TỪ FILE ---
def load_knowledge_graph(graph_file_path):
    """
    Tải đối tượng đồ thị NetworkX từ file GraphML.

    Args:
        graph_file_path (str): Đường dẫn đến file .graphml.

    Returns:
        nx.DiGraph or None: Đối tượng đồ thị đã tải, hoặc None nếu có lỗi.
    """
    if not os.path.exists(graph_file_path):
        print(f"LỖI: File đồ thị '{graph_file_path}' không tìm thấy.")
        print(
            "Vui lòng đảm bảo bạn đã chạy script xây dựng đồ thị trước và file đã được lưu đúng vị trí."
        )
        return None

    try:
        print(f"Đang tải đồ thị từ file: {graph_file_path}...")
        graph = nx.read_graphml(graph_file_path)
        print("Tải đồ thị thành công.")
        # Chuyển đổi ID nút về đúng kiểu nếu cần (read_graphml có thể đọc ID dạng chuỗi)
        # Tuy nhiên, nếu ID ban đầu đã là chuỗi thì không cần thiết.
        # Ví dụ, nếu bạn biết ID nút phải là số nguyên:
        # graph = nx.convert_node_labels_to_integers(graph, first_label=0)
        return graph
    except Exception as e:
        print(f"Lỗi khi tải đồ thị từ file '{graph_file_path}': {e}")
        import traceback

        traceback.print_exc()
        return None


# --- KHỐI THỰC THI CHÍNH ---
if __name__ == "__main__":
    # 1. Tải đồ thị từ file
    knowledge_graph = load_knowledge_graph(GRAPH_FILE_TO_LOAD)

    # 2. Kiểm tra và In thông tin đồ thị (Như code bạn đã cung cấp)
    if knowledge_graph and knowledge_graph.number_of_nodes() > 0:
        print("\n--- THÔNG TIN CƠ BẢN VỀ ĐỒ THỊ ---")
        print(f"Số lượng nút (nodes): {knowledge_graph.number_of_nodes()}")
        print(f"Số lượng cạnh (edges): {knowledge_graph.number_of_edges()}")

        # In ra một vài nút đầu tiên với dữ liệu của chúng
        print("\n--- MỘT VÀI NÚT ĐẦU TIÊN (kèm dữ liệu) ---")
        nodes_to_show = 5
        count = 0
        for node_id, data in knowledge_graph.nodes(data=True):
            if count >= nodes_to_show:
                break
            print(f"Nút ID: {node_id}, Dữ liệu: {data}")
            count += 1

        # In ra một vài cạnh đầu tiên với dữ liệu của chúng
        print("\n--- MỘT VÀI CẠNH ĐẦU TIÊN (kèm dữ liệu) ---")
        edges_to_show = 5
        count = 0
        for u, v, data in knowledge_graph.edges(data=True):
            if count >= edges_to_show:
                break
            print(f"Cạnh từ '{u}' đến '{v}', Dữ liệu: {data}")
            count += 1

        # Kiểm tra thông tin của một nút cụ thể
        first_doc_node = None
        for node_id, data in knowledge_graph.nodes(data=True):
            if data.get("type") == "Document":
                first_doc_node = node_id
                break

        if first_doc_node:
            print(f"\n--- THÔNG TIN NÚT CỤ THỂ ({first_doc_node}) ---")
            # NetworkX lưu trữ dữ liệu nút trong một dict-like object
            print(knowledge_graph.nodes[first_doc_node])

            print(f"\n--- CÁC NÚT LÂN CẬN (SUCCESSORS) CỦA {first_doc_node} ---")
            # knowledge_graph là DiGraph nên có successors
            if isinstance(knowledge_graph, nx.DiGraph):
                for successor in knowledge_graph.successors(first_doc_node):
                    print(
                        f"  -> {successor} (Loại: {knowledge_graph.nodes[successor].get('type')})"
                    )
                    if knowledge_graph.nodes[successor].get("type") == "Chunk":
                        chunk_text = knowledge_graph.nodes[successor].get(
                            "text_content", ""
                        )
                        print(
                            f"     Nội dung chunk (100 ký tự đầu): {chunk_text[:100]}..."
                        )
            else:  # Nếu là Graph không có hướng, dùng neighbors
                for neighbor in knowledge_graph.neighbors(first_doc_node):
                    print(
                        f"  - {neighbor} (Loại: {knowledge_graph.nodes[neighbor].get('type')})"
                    )

        print("\n--- THỐNG KÊ SỐ LƯỢNG CÁC LOẠI NÚT ---")
        node_types = {}
        for _, data in knowledge_graph.nodes(data=True):
            node_type = data.get("type", "Không xác định")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        for (
            n_type,
            count_nodes,
        ) in node_types.items():  # Đổi tên biến count thành count_nodes
            print(f"Loại nút '{n_type}': {count_nodes} nút")

    else:
        print("Đồ thị rỗng hoặc không thể tải.")

    # 3. Trực quan hóa đồ thị bằng Matplotlib (Như code bạn đã cung cấp)
    if knowledge_graph and knowledge_graph.number_of_nodes() > 0:
        # Giới hạn số lượng nút để vẽ nếu đồ thị quá lớn cho matplotlib
        if knowledge_graph.number_of_nodes() > 100:  # Ví dụ: chỉ vẽ nếu ít hơn 100 nút
            print(
                "\nĐồ thị quá lớn để vẽ bằng Matplotlib một cách hiệu quả. Bỏ qua bước vẽ."
            )
            print(f"Hãy sử dụng công cụ như Gephi để mở file '{GRAPH_FILE_TO_LOAD}'.")
        else:
            print(
                "\nĐang vẽ đồ thị bằng Matplotlib (có thể mất chút thời gian và không tối ưu cho đồ thị lớn)..."
            )
            plt.figure(figsize=(18, 12))  # Tăng kích thước

            # Sử dụng layout khác nếu shell_layout không phù hợp
            # pos = nx.spring_layout(knowledge_graph, k=0.8, iterations=30)
            try:
                # Kamada-Kawai thường cho kết quả tốt hơn cho đồ thị kích thước vừa phải
                pos = nx.kamada_kawai_layout(knowledge_graph)
            except (
                nx.NetworkXException
            ):  # Fallback nếu kamada_kawai lỗi (ví dụ đồ thị không liên thông)
                print("Layout Kamada-Kawai thất bại, thử Spring layout...")
                pos = nx.spring_layout(knowledge_graph, k=0.8, iterations=30)

            labels = {
                node_id: data.get(
                    "name", data.get("type", str(node_id)[:15])
                )  # Cắt bớt ID nếu quá dài
                for node_id, data in knowledge_graph.nodes(data=True)
            }

            # Lấy màu cho các loại nút khác nhau
            color_map = []
            for node in knowledge_graph.nodes(data=True):
                if node[1].get("type") == "Document":
                    color_map.append("red")
                elif node[1].get("type") == "Chunk":
                    color_map.append("skyblue")
                elif node[1].get("type") == "Heading":
                    color_map.append("lightgreen")
                elif node[1].get("type") == "Field":
                    color_map.append("yellow")
                else:
                    color_map.append("grey")

            nx.draw(
                knowledge_graph,
                pos,
                with_labels=True,
                labels=labels,
                node_size=1500,  # Tăng kích thước nút
                node_color=color_map,
                font_size=7,  # Giảm kích thước font
                font_weight="normal",
                arrows=isinstance(
                    knowledge_graph, nx.DiGraph
                ),  # Chỉ vẽ mũi tên nếu là DiGraph
                arrowstyle="-|>",
                arrowsize=12,
                width=0.5,  # Độ dày của cạnh
            )

            edge_labels = nx.get_edge_attributes(knowledge_graph, "type")
            nx.draw_networkx_edge_labels(
                knowledge_graph, pos, edge_labels=edge_labels, font_size=6
            )

            plt.title("Trực quan hóa Knowledge Graph (Sử dụng Matplotlib)")
            plt.show()
    else:
        print("Không thể vẽ đồ thị rỗng hoặc đồ thị chưa được tải.")

import heapq
import networkx as nx
import matplotlib.pyplot as plt


def dijkstra(graph, start):
    # 初始化距离字典
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # 使用优先队列来实现最小堆
    pq = [(0, start)]

    while pq:
        # 弹出当前距离最小的节点
        current_distance, current_node = heapq.heappop(pq)

        # 如果当前节点已经处理过，跳过
        if current_distance > distances[current_node]:
            continue

        # 遍历当前节点的邻居
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # 如果新的路径长度比当前记录的短，更新距离字典和优先队列
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances


def draw_graph(graph, distances, shortest_path, start_node):
    G = nx.DiGraph()

    # 添加节点
    for node in graph:
        G.add_node(node)

    # 添加边
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            if (node, neighbor) in shortest_path or (neighbor, node) in shortest_path:
                G.add_edge(node, neighbor, weight=weight, color='red')
            else:
                G.add_edge(node, neighbor, weight=weight, color='black')

    # 绘制图形
    pos = nx.spring_layout(G)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color=edge_colors, arrows=True)

    labels = {}
    for i in distances:
        if isinstance(distances[i], dict):
            for j in distances[i]:
                if (i, j) in G.edges():
                    labels[(i, j)] = str(distances[i][j])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # 标记起点
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='red', node_size=500)

    plt.show()

if __name__ == "__main__":
    # 示例图的邻接表表示
    graph = {
        'A': {'B': 2, 'D': 4},
        'B': {'C': 5, 'E': 7},
        'C': {'E': 6},
        'D': {'E': 2},
        'E': {'C': 1}
    }

    # 执行Dijkstra算法
    start_node = 'A'
    distances = dijkstra(graph, start_node)

    # 找到最短路径
    shortest_path = []
    for node in graph:
        if node != start_node:
            path = nx.shortest_path(nx.Graph(graph), start_node, node, weight='weight')
            for i in range(len(path) - 1):
                shortest_path.append((path[i], path[i + 1]))

    # 绘制图示
    draw_graph(graph, distances, shortest_path, start_node)

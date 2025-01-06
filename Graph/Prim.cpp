#include <iostream>
#include <vector>
#include <queue>

using namespace std;

// Cấu trúc đại diện cho một cạnh (edge)
struct Edge {
    int vertex;    // Đỉnh kết nối
    int weight;    // Trọng số của cạnh
};

// So sánh hai cạnh dựa trên trọng số để sử dụng priority_queue
struct Compare {
    bool operator()(Edge const &a, Edge const &b) {
        return a.weight > b.weight; // Cạnh có trọng số nhỏ hơn sẽ được ưu tiên
    }
};

// Hàm Prim để tìm cây khung nhỏ nhất
void primMST(const vector<vector<Edge>> &graph, int startVertex) {
    int n = graph.size(); // Số lượng đỉnh trong đồ thị

    // Mảng để đánh dấu các đỉnh đã được thêm vào cây khung nhỏ nhất
    vector<bool> visited(n, false);

    // Mảng lưu trọng số nhỏ nhất để kết nối mỗi đỉnh
    vector<int> minWeight(n, INT_MAX);

    // Mảng lưu đỉnh cha để theo dõi cây khung nhỏ nhất
    vector<int> parent(n, -1);

    // Priority Queue để lưu các cạnh theo trọng số
    priority_queue<Edge, vector<Edge>, Compare> pq;

    // Bắt đầu từ đỉnh `startVertex`
    pq.push({startVertex, 0});
    minWeight[startVertex] = 0;

    while (!pq.empty()) {
        // Lấy đỉnh có trọng số nhỏ nhất từ hàng đợi ưu tiên
        int u = pq.top().vertex;
        pq.pop();

        // Nếu đỉnh đã được xử lý, bỏ qua
        if (visited[u]) continue;

        // Đánh dấu đỉnh đã được thêm vào cây khung
        visited[u] = true;

        // Duyệt qua các cạnh kề của đỉnh hiện tại
        for (const Edge &neighbor : graph[u]) {
            int v = neighbor.vertex;
            int weight = neighbor.weight;

            // Nếu đỉnh `v` chưa nằm trong MST và trọng số nhỏ hơn trọng số hiện tại
            if (!visited[v] && weight < minWeight[v]) {
                minWeight[v] = weight;
                parent[v] = u;
                pq.push({v, weight});
            }
        }
    }

    // In cây khung nhỏ nhất
    cout << "Cây khung nhỏ nhất (MST):" << endl;
    for (int i = 1; i < n; i++) {
        if (parent[i] != -1) {
            cout << "Edge: " << parent[i] << " - " << i << " | Weight: " << minWeight[i] << endl;
        }
    }
}

int main() {
    // Đồ thị dạng danh sách kề (Adjacency List)
    vector<vector<Edge>> graph = {
        {{1, 2}, {3, 6}}, // Đỉnh 0
        {{0, 2}, {2, 3}, {3, 8}, {4, 5}}, // Đỉnh 1
        {{1, 3}, {4, 7}}, // Đỉnh 2
        {{0, 6}, {1, 8}}, // Đỉnh 3
        {{1, 5}, {2, 7}}  // Đỉnh 4
    };

    int startVertex = 0; // Đỉnh bắt đầu

    // Gọi hàm Prim
    primMST(graph, startVertex);

    return 0;
}


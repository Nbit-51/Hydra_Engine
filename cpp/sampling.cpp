#include <vector>
#include <algorithm>
#include <queue>
#include <cstring>

struct TokenScore { 
    int id; 
    float score; 
    
    // Reverse comparison for min-heap (so smallest score is on top)
    bool operator<(const TokenScore& other) const {
        return score > other.score;
    }
};

// Optimized top-k using heap for better cache locality
std::vector<int> top_k_filter(const std::vector<float>& logits, int k) {
    const int n = static_cast<int>(logits.size());
    
    // Clamp k to valid range
    k = std::min(k, n);
    if (k <= 0) return {};
    
    // Use min-heap to maintain top-k elements
    // Only stores k elements instead of n (huge memory savings)
    std::priority_queue<TokenScore> heap;
    
    // Initialize heap with first k elements
    for (int i = 0; i < k; ++i) {
        heap.push({i, logits[i]});
    }
    
    // Process remaining elements
    float min_score = heap.top().score;
    for (int i = k; i < n; ++i) {
        if (logits[i] > min_score) {
            heap.pop();
            heap.push({i, logits[i]});
            min_score = heap.top().score;
        }
    }
    
    // Extract results (will be in ascending order)
    std::vector<int> top_indices;
    top_indices.reserve(k);
    
    while (!heap.empty()) {
        top_indices.push_back(heap.top().id);
        heap.pop();
    }
    
    // Reverse to get descending order
    std::reverse(top_indices.begin(), top_indices.end());
    
    return top_indices;
}

// Alternative: Fast approximate top-k using partitioning (O(n) average)
std::vector<int> top_k_filter_fast(const std::vector<float>& logits, int k) {
    const int n = static_cast<int>(logits.size());
    k = std::min(k, n);
    
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    
    // Partial sort only up to k (O(n) average case)
    std::nth_element(
        indices.begin(), 
        indices.begin() + k, 
        indices.end(),
        [&logits](int a, int b) { return logits[a] > logits[b]; }
    );
    
    // Extract and sort top-k
    std::vector<int> result(indices.begin(), indices.begin() + k);
    std::sort(result.begin(), result.end(), 
              [&logits](int a, int b) { return logits[a] > logits[b]; });
    
    return result;
}

#include <vector>
#include <algorithm>

struct TokenScore { int id; float score; };
bool compareTokens(TokenScore a, TokenScore b) { return (a.score > b.score); }

std::vector<int> top_k_filter(std::vector<float> logits, int k) {
    int n = logits.size();
    std::vector<TokenScore> tokens(n);
    for (int i = 0; i < n; i++) tokens[i] = {i, logits[i]};
    std::partial_sort(tokens.begin(), tokens.begin() + k, tokens.end(), compareTokens);
    std::vector<int> top_indices;
    for (int i = 0; i < k; i++) top_indices.push_back(tokens[i].id);
    return top_indices;
}

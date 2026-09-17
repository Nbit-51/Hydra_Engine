#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>

struct Vocab {
    std::vector<std::string> pieces;
    uint32_t eos_id = 0;
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t n = 0;
        f.read(reinterpret_cast<char*>(&n), 4);
        f.read(reinterpret_cast<char*>(&eos_id), 4);
        pieces.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t len = 0;
            f.read(reinterpret_cast<char*>(&len), 4);
            pieces[i].resize(len);
            if (len) f.read(&pieces[i][0], len);
        }
        return (bool)f;
    }
};

static size_t utf8_pending(const std::string& s) {
    for (size_t i = 1; i <= 3 && i <= s.size(); ++i) {
        unsigned char c = (unsigned char)s[s.size() - i];
        if ((c & 0x80) == 0) return 0;
        if ((c & 0xC0) == 0xC0) {
            int need = (c & 0xF0) == 0xF0 ? 4 : ((c & 0xE0) == 0xE0 ? 3 : 2);
            return (int)i < need ? i : 0;
        }
    }
    return 0;
}

struct Streamer {
    std::string pending;
    void feed(const std::string& piece) {
        pending += piece;
        size_t keep = utf8_pending(pending);
        size_t out_len = pending.size() - keep;
        if (out_len) {
            std::cout.write(pending.data(), out_len);
            std::cout.flush();
            pending = pending.substr(out_len);
        }
    }
    void end() {
        std::cout.write(pending.data(), pending.size());
        std::cout.flush();
    }
};

static std::vector<int64_t> parse_ids(const std::string& s) {
    std::vector<int64_t> ids;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) ids.push_back(std::stoll(tok));
    }
    return ids;
}

static int64_t sample(torch::Tensor logits, float temp = 0.7, int top_k = 40) {
    auto next_logits = logits / temp;
    auto topk = torch::topk(next_logits, top_k, -1);
    auto values = std::get<0>(topk).squeeze(0);
    auto indices = std::get<1>(topk).squeeze(0);
    auto probs = torch::softmax(values, -1);
    auto next_token_idx = torch::multinomial(probs, 1);
    auto chosen_token = indices.gather(0, next_token_idx);
    return chosen_token.item<int64_t>();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " model.pt vocab.bin \"id1,id2,...\" [max_tokens]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string vocab_path = argv[2];
    auto prompt_ids = parse_ids(argv[3]);
    int max_tokens = argc > 4 ? std::atoi(argv[4]) : 128;

    torch::Device device(torch::kCUDA);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    torch::NoGradGuard no_grad;

    Vocab vocab;
    if (!vocab.load(vocab_path)) { std::cerr << "cannot load vocab\n"; return 1; }

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path, device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return 1;
    }

    try { model.run_method("reset_cache"); } catch (...) {}

    auto opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    Streamer streamer;
    for (auto id : prompt_ids)
        if (id >= 0 && (size_t)id < vocab.pieces.size()) streamer.feed(vocab.pieces[id]);

    // ---------- PREFILL ----------
    std::vector<int64_t> ids = prompt_ids.empty() ? std::vector<int64_t>{1} : prompt_ids;
    int64_t n = (int64_t)ids.size();
    auto input_ids = torch::from_blob(ids.data(), {1, n}, torch::kLong).clone().to(device);
    auto position_ids = torch::arange(n, torch::kLong).to(device).view({1, n});

    std::vector<torch::jit::IValue> pf;
    pf.push_back(input_ids);
    pf.push_back(position_ids);
    pf.push_back(n);
    auto logits = model.forward(pf).toTensor();
    cudaDeviceSynchronize();

    int64_t cur = sample(logits.select(1, n - 1));
    int64_t pos = n;

    // ---------- DECODE ----------
    auto d_ids = torch::zeros({1, 1}, opts);
    auto d_pos = torch::zeros({1, 1}, opts);
    std::vector<torch::jit::IValue> di;
    di.push_back(d_ids);
    di.push_back(d_pos);
    di.push_back(pos + 1);  // placeholder, overwritten each iteration below

    // CRITICAL: Warmup iterations to trigger JIT compilation and cuBLAS autotuning
    d_ids.fill_(cur);
    d_pos.fill_(pos);
    for (int i = 0; i < 3; ++i) {
        model.forward(di);
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    int generated = 0;
    for (int i = 0; i < max_tokens; ++i) {
        if (cur == (int64_t)vocab.eos_id) break;
        if (cur >= 0 && (size_t)cur < vocab.pieces.size()) streamer.feed(vocab.pieces[cur]);

        d_ids.fill_(cur);
        d_pos.fill_(pos);
        di[2] = pos + 1;
        logits = model.forward(di).toTensor();

        cur = sample(logits.select(1, 0));
        pos++;
        generated++;
    }
    streamer.end();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "\n\n[decode] " << generated << " tokens in " << ms << " ms ("
              << 1000.0 * generated / ms << " tok/s)" << std::endl;
    return 0;
}

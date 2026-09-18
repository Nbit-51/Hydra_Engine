#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

enum class CudaGraphMode { Auto, Off, Required, IsolatedAttempt };

struct DecodeGraph {
    int64_t cache_bucket = 0;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    torch::Tensor logits;
};

static CudaGraphMode parse_graph_mode(const std::string& value) {
    if (value == "auto") return CudaGraphMode::Auto;
    if (value == "off") return CudaGraphMode::Off;
    if (value == "required") return CudaGraphMode::Required;
    if (value == "isolated-attempt") return CudaGraphMode::IsolatedAttempt;
    throw std::invalid_argument("cuda_graphs must be auto, off, or required");
}

static int run_isolated_graph_attempt(int argc, char** argv) {
    // A failed LibTorch capture may poison CUDA allocator state. Let a child
    // own the graph attempt so the parent remains clean enough to fall back.
    std::vector<std::string> owned_args{
        argv[0], argv[1], argv[2], argv[3], argc > 4 ? argv[4] : "128", "isolated-attempt"
    };
    if (argc > 6) owned_args.emplace_back(argv[6]);

    std::vector<char*> child_args;
    child_args.reserve(owned_args.size() + 1);
    for (auto& arg : owned_args) child_args.push_back(arg.data());
    child_args.push_back(nullptr);

#ifdef _WIN32
    return static_cast<int>(_spawnv(_P_WAIT, child_args[0], child_args.data()));
#else
    const pid_t pid = fork();
    if (pid == 0) {
        execvp(child_args[0], child_args.data());
        _exit(127);
    }
    if (pid < 0) return -1;

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
#endif
}

static int64_t decode_bucket(int64_t length) {
    int64_t bucket = 8;
    while (bucket < length) bucket <<= 1;
    return bucket;
}

static std::unique_ptr<DecodeGraph> capture_decode_graph(
    torch::jit::script::Module& model,
    const torch::Tensor& input_ids,
    const torch::Tensor& position_ids,
    int64_t cache_bucket,
    std::string& error
) {
    auto state = std::make_unique<DecodeGraph>();
    state->cache_bucket = cache_bucket;
    state->graph = std::make_unique<at::cuda::CUDAGraph>();

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(4);
    inputs.push_back(input_ids);
    inputs.push_back(position_ids);
    inputs.push_back(cache_bucket);
    inputs.push_back(true);

    auto capture_stream = c10::cuda::getStreamFromPool(false);
    try {
        {
            c10::cuda::CUDAStreamGuard guard(capture_stream);
            for (int i = 0; i < 3; ++i) model.forward(inputs);
        }
        capture_stream.synchronize();

        {
            c10::cuda::CUDAStreamGuard guard(capture_stream);
            state->graph->capture_begin();
            state->logits = model.forward(inputs).toTensor();
            state->graph->capture_end();
        }
        capture_stream.synchronize();
        return state;
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}

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
        std::cerr << "usage: " << argv[0]
                  << " model.pt vocab.bin \"id1,id2,...\" [max_tokens]"
                  << " [cuda_graphs=auto|off|required] [seed]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string vocab_path = argv[2];
    auto prompt_ids = parse_ids(argv[3]);
    int max_tokens = 128;
    int64_t seed = 0;
    bool has_seed = false;
    try {
        if (argc > 4) max_tokens = std::stoi(argv[4]);
        if (max_tokens <= 0) throw std::invalid_argument("max_tokens must be positive");
        if (argc > 6) {
            seed = std::stoll(argv[6]);
            has_seed = true;
        }
    } catch (const std::exception& e) {
        std::cerr << "FATAL: invalid numeric argument: " << e.what() << std::endl;
        return 1;
    }

    CudaGraphMode graph_mode = CudaGraphMode::Auto;
    try {
        graph_mode = parse_graph_mode(argc > 5 ? argv[5] : "auto");
    } catch (const std::invalid_argument& e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return 1;
    }
    if (graph_mode == CudaGraphMode::Auto) {
        const int attempt_status = run_isolated_graph_attempt(argc, argv);
        if (attempt_status == 0) return 0;
        if (attempt_status != 2) {
            std::cerr << "FATAL: isolated CUDA Graph attempt exited with status "
                      << attempt_status << std::endl;
            return attempt_status > 0 ? attempt_status : 1;
        }
        std::cerr << "[cuda-graph] isolated capture failed; retrying with standard LibTorch decode"
                  << std::endl;
        graph_mode = CudaGraphMode::Off;
    }
    if (has_seed) torch::manual_seed(seed);

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

    // ---------- PREFILL ----------
    std::vector<int64_t> ids = prompt_ids.empty() ? std::vector<int64_t>{1} : prompt_ids;
    int64_t n = (int64_t)ids.size();
    auto input_ids = torch::from_blob(ids.data(), {1, n}, torch::kLong).clone().to(device);
    auto position_ids = torch::arange(n, torch::kLong).to(device).view({1, n});

    std::vector<torch::jit::IValue> pf;
    pf.push_back(input_ids);
    pf.push_back(position_ids);
    pf.push_back(n);
    pf.push_back(false);

    auto reset_and_prefill = [&]() -> torch::Tensor {
        model.run_method("reset_cache");
        auto result = model.forward(pf).toTensor();
        cudaDeviceSynchronize();
        return result;
    };

    torch::Tensor logits;
    try {
        logits = reset_and_prefill();
    } catch (const c10::Error& e) {
        std::cerr << "FATAL: exported model does not provide the current Hydra decode API: "
                  << e.what() << std::endl;
        std::cerr << "Re-run export_to_cpp.py before launching this binary." << std::endl;
        return 1;
    }
    int64_t cur = sample(logits.select(1, n - 1));
    int64_t pos = n;

    // ---------- DECODE ----------
    auto d_ids = torch::zeros({1, 1}, opts);
    auto d_pos = torch::zeros({1, 1}, opts);
    std::vector<torch::jit::IValue> di;
    di.push_back(d_ids);
    di.push_back(d_pos);
    di.push_back(pos + 1);
    di.push_back(false);

    d_ids.fill_(cur);
    d_pos.fill_(pos);

    std::unordered_map<int64_t, std::unique_ptr<DecodeGraph>> graphs;
    bool graphs_enabled = graph_mode != CudaGraphMode::Off;
    if (graphs_enabled) {
        std::vector<int64_t> buckets;
        std::unordered_set<int64_t> seen;
        for (int i = 0; i < max_tokens; ++i) {
            const int64_t bucket = decode_bucket(n + i + 1);
            if (seen.insert(bucket).second) buckets.push_back(bucket);
        }

        for (const int64_t bucket : buckets) {
            std::string capture_error;
            auto graph = capture_decode_graph(model, d_ids, d_pos, bucket, capture_error);
            if (!graph) {
                std::cerr << "[cuda-graph] capture failed for cache bucket " << bucket
                          << ": " << capture_error << std::endl;
                graphs.clear();
                graphs_enabled = false;
                break;
            }
            graphs.emplace(bucket, std::move(graph));
        }

        if (!graphs_enabled && graph_mode != CudaGraphMode::Off) {
            if (graph_mode == CudaGraphMode::Required) {
                std::cerr << "FATAL: CUDA Graph capture was required but could not be enabled."
                          << std::endl;
            }
            std::cerr.flush();
            std::_Exit(2);
        }
    }

    // Capture warmup mutates the static cache slots. Rebuild a clean prefill
    // state before timed generation, regardless of capture success.
    logits = reset_and_prefill();
    cur = sample(logits.select(1, n - 1));
    pos = n;
    d_ids.fill_(cur);
    d_pos.fill_(pos);

    if (graphs_enabled) {
        std::cout << "[cuda-graph] enabled with " << graphs.size()
                  << " decode bucket(s)" << std::endl;
    } else {
        std::cout << "[cuda-graph] disabled; using standard LibTorch decode" << std::endl;
        for (int i = 0; i < 3; ++i) model.forward(di);
    }
    cudaDeviceSynchronize();

    for (auto id : prompt_ids)
        if (id >= 0 && (size_t)id < vocab.pieces.size()) streamer.feed(vocab.pieces[id]);

    auto t0 = std::chrono::high_resolution_clock::now();
    int generated = 0;
    for (int i = 0; i < max_tokens; ++i) {
        if (cur == (int64_t)vocab.eos_id) break;
        if (cur >= 0 && (size_t)cur < vocab.pieces.size()) streamer.feed(vocab.pieces[cur]);

        d_ids.fill_(cur);
        d_pos.fill_(pos);
        if (graphs_enabled) {
            const int64_t bucket = decode_bucket(pos + 1);
            auto graph_it = graphs.find(bucket);
            if (graph_it == graphs.end()) {
                std::cerr << "FATAL: no captured CUDA Graph for cache bucket " << bucket
                          << std::endl;
                return 2;
            }
            graph_it->second->graph->replay();
            logits = graph_it->second->logits;
        } else {
            di[2] = pos + 1;
            logits = model.forward(di).toTensor();
        }

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

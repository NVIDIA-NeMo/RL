#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

__global__ void dp_chunk_kernel(
    const int64_t* __restrict__ ids1,
    const int64_t* __restrict__ ids2,
    const int64_t* __restrict__ joined1,
    const int64_t* __restrict__ joined2,
    float* __restrict__ dp,
    int16_t* __restrict__ trace,
    int n1,
    int n2,
    int max_comb_len,
    float exact_match_score,
    float gap_penalty,
    float combination_score_multiplier
) {
    const int tid = threadIdx.x;
    const int dp_cols = n2 + 1;
    const int join_cols = max_comb_len + 1;
    const int64_t invalid = static_cast<int64_t>(-1);

    for (int i = tid; i <= n1; i += blockDim.x) {
        dp[i * dp_cols + 0] = static_cast<float>(i) * gap_penalty;
        trace[i * dp_cols + 0] = 2; // up
    }
    for (int j = tid; j <= n2; j += blockDim.x) {
        dp[0 * dp_cols + j] = static_cast<float>(j) * gap_penalty;
        trace[0 * dp_cols + j] = 3; // left
    }
    if (tid == 0) {
        trace[0] = 0;
    }
    __syncthreads();

    for (int diag = 2; diag <= n1 + n2; ++diag) {
        const int j_start = max(1, diag - n1);
        const int j_end = min(n2, diag - 1);
        const int cells = j_end - j_start + 1;

        for (int t = tid; t < cells; t += blockDim.x) {
            const int j = j_start + t;
            const int i = diag - j;

            const int64_t id_i = ids1[i - 1];
            const int64_t id_j = ids2[j - 1];

            float best = dp[(i - 1) * dp_cols + (j - 1)] + ((id_i == id_j) ? exact_match_score : -exact_match_score);
            int16_t best_move = 1; // diag

            float s_up = dp[(i - 1) * dp_cols + j] + gap_penalty;
            if (s_up > best) {
                best = s_up;
                best_move = 2;
            }

            float s_left = dp[i * dp_cols + (j - 1)] + gap_penalty;
            if (s_left > best) {
                best = s_left;
                best_move = 3;
            }

            const int k_max_s2 = min(j, max_comb_len);
            for (int k = 2; k <= k_max_s2; ++k) {
                const int64_t joined = joined2[j * join_cols + k];
                if (joined != invalid && id_i == joined) {
                    float s = dp[(i - 1) * dp_cols + (j - k)] + combination_score_multiplier * static_cast<float>(k);
                    if (s > best) {
                        best = s;
                        best_move = static_cast<int16_t>(10 + k); // comb_s1_over_s2_k
                    }
                }
            }

            const int k_max_s1 = min(i, max_comb_len);
            for (int k = 2; k <= k_max_s1; ++k) {
                const int64_t joined = joined1[i * join_cols + k];
                if (joined != invalid && id_j == joined) {
                    float s = dp[(i - k) * dp_cols + (j - 1)] + combination_score_multiplier * static_cast<float>(k);
                    if (s > best) {
                        best = s;
                        best_move = static_cast<int16_t>(20 + k); // comb_s2_over_s1_k
                    }
                }
            }

            dp[i * dp_cols + j] = best;
            trace[i * dp_cols + j] = best_move;
        }
        __syncthreads();
    }
}

inline void check_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

} // namespace

std::vector<torch::Tensor> dp_chunk_cuda(
    torch::Tensor ids1,
    torch::Tensor ids2,
    torch::Tensor joined1,
    torch::Tensor joined2,
    double exact_match_score,
    double combination_score_multiplier,
    double gap_penalty,
    int64_t max_comb_len
) {
    check_cuda(ids1, "ids1");
    check_cuda(ids2, "ids2");
    check_cuda(joined1, "joined1");
    check_cuda(joined2, "joined2");

    TORCH_CHECK(ids1.dtype() == torch::kInt64, "ids1 must be int64");
    TORCH_CHECK(ids2.dtype() == torch::kInt64, "ids2 must be int64");
    TORCH_CHECK(joined1.dtype() == torch::kInt64, "joined1 must be int64");
    TORCH_CHECK(joined2.dtype() == torch::kInt64, "joined2 must be int64");

    TORCH_CHECK(ids1.dim() == 1, "ids1 must be [n1]");
    TORCH_CHECK(ids2.dim() == 1, "ids2 must be [n2]");
    TORCH_CHECK(joined1.dim() == 2, "joined1 must be [n1+1, max_comb_len+1]");
    TORCH_CHECK(joined2.dim() == 2, "joined2 must be [n2+1, max_comb_len+1]");

    const int n1 = static_cast<int>(ids1.size(0));
    const int n2 = static_cast<int>(ids2.size(0));
    TORCH_CHECK(joined1.size(0) == n1 + 1, "joined1 first dim must be n1+1");
    TORCH_CHECK(joined2.size(0) == n2 + 1, "joined2 first dim must be n2+1");
    TORCH_CHECK(joined1.size(1) == max_comb_len + 1, "joined1 second dim mismatch");
    TORCH_CHECK(joined2.size(1) == max_comb_len + 1, "joined2 second dim mismatch");

    auto dp = torch::empty({n1 + 1, n2 + 1}, ids1.options().dtype(torch::kFloat32));
    auto trace = torch::empty({n1 + 1, n2 + 1}, ids1.options().dtype(torch::kInt16));

    const int threads = 256;
    dp_chunk_kernel<<<1, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        ids1.data_ptr<int64_t>(),
        ids2.data_ptr<int64_t>(),
        joined1.data_ptr<int64_t>(),
        joined2.data_ptr<int64_t>(),
        dp.data_ptr<float>(),
        trace.data_ptr<int16_t>(),
        n1,
        n2,
        static_cast<int>(max_comb_len),
        static_cast<float>(exact_match_score),
        static_cast<float>(gap_penalty),
        static_cast<float>(combination_score_multiplier)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto score = dp.index({n1, n2}).unsqueeze(0);
    return {trace, score};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dp_chunk_cuda", &dp_chunk_cuda, "TokenAlign DP chunk CUDA");
}


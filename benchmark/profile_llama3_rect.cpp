// Nsight Systems capture target for the Llama-3 sublayers with every
// plaintext-ciphertext product on Kang's Algorithm 5 and every
// ciphertext-ciphertext product on Kang's Algorithm 4.
//
//   llama3_rect_profile
//
// WHY THIS EXISTS
// ---------------
// profile_llama3_nobatch.cpp measures Algorithm 5's PROJECTIONS and says so
// plainly: the non-linear layers and the two encrypted products of attention are
// not in it, "because on this path they need the slot form and the slot form
// needs the block index moved off the Y axis, which is a second homomorphic
// transform that is not written yet".
//
// It is written now, in llama3_rect.cu, so this target measures what that file
// unlocked: the whole sublayers, not merely their matrix halves.
//
// WHAT IT MEASURES
// ----------------
// One stage per run, chosen with HEONGPU_RECT_STAGE, each inside its own NVTX
// range and with the profiler started only around the measured region:
//
//   project   one d_model -> d_model projection, Algorithm 5
//   bridge    rect -> batch -> rect, the transform this path needed
//   ccmm      Q K^T for k/2 heads in one Algorithm 4 call
//   rmsnorm   the whole RMSNorm sublayer, bridges included
//   ffn       the whole SwiGLU sublayer
//   attention the whole attention sublayer
//   block     one pre-norm transformer block
//
// WHAT WILL NOT FIT, STATED UP FRONT
// ----------------------------------
// Algorithm 5's summation is a CMT read at k = 2, so it needs N/2 - 1 Galois
// keys -- 2047 at HEonGPU's minimum ring, and the count depends on the ring
// degree ALONE. The key set therefore scales with the chain and nothing else,
// and the deeper stages here run out of card before they run out of levels. The
// dry run reports the budget before anything is allocated; use it.
//
// As in the sibling targets the plaintext values are random and the fitted
// intervals are nominal: nothing about the cost of a CKKS circuit depends on the
// numbers in it.

#include <heongpu/heongpu.hpp>

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

constexpr auto S = heongpu::Scheme::CKKS;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
using Rect = heongpu::llama::Llama3RectOperator;

int EnvInt(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return fallback;
    return std::atoi(v);
}

std::string EnvStr(const char* name, const char* fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return std::string(fallback);
    return std::string(v);
}

struct Range
{
    explicit Range(const std::string& name) { nvtxRangePushA(name.c_str()); }
    ~Range() { nvtxRangePop(); }
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
};

/// Wall time of a region. Most host time here is blocking cudaMemcpy rather
/// than cudaDeviceSynchronize, so no single row of the Nsight API table is the
/// answer and events bracketing the region are.
struct RegionTimer
{
    cudaEvent_t start, stop;
    RegionTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    float ms()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float t = 0.0f;
        cudaEventElapsedTime(&t, start, stop);
        return t;
    }
    ~RegionTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

void ReportMemory(const char* stage)
{
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        return;
    std::cout << "[rect] memory after " << stage << ": "
              << (static_cast<double>(total_bytes - free_bytes) / kGiB)
              << " GiB used of " << (static_cast<double>(total_bytes) / kGiB)
              << " GiB" << std::endl;
}

double SwitchingKeyBytes(int n, int limbs, int special)
{
    const int groups = (limbs + special - 1) / special;
    return 2.0 * groups * (limbs + special) * n * 8.0;
}

std::vector<double> random_matrix(std::size_t n, double amplitude,
                                  std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> pick(-amplitude, amplitude);
    std::vector<double> out(n);
    for (double& v : out)
        v = pick(rng);
    return out;
}

} // namespace

int main(int argc, char* argv[])
{
    (void) argc;
    (void) argv;

    const std::string stage = EnvStr("HEONGPU_RECT_STAGE", "project");
    const int log_n = EnvInt("HEONGPU_RECT_LOGN", 12);
    const int d = EnvInt("HEONGPU_RECT_D", 128);
    const int limbs = EnvInt("HEONGPU_RECT_LIMBS", 10);
    const int special = EnvInt("HEONGPU_RECT_SPECIAL_PRIMES", 5);
    const int prime_bits = EnvInt("HEONGPU_RECT_PRIME_BITS", 40);
    const bool dry_run = EnvInt("HEONGPU_RECT_DRY_RUN", 0) != 0;

    const int n = 1 << log_n;
    const int k = n / d;
    const int step = k / 2;
    const int half = n / 2;

    // Every width here is in whole groups of N/2 channels, because that is what
    // one Algorithm 5 call covers and what one Algorithm 4 call's batch axis
    // holds. A partly filled group would carry heads that do not exist.
    const int channel_groups = EnvInt("HEONGPU_RECT_CHANNEL_GROUPS", 1);
    const int hidden_groups = EnvInt("HEONGPU_RECT_HIDDEN_GROUPS", 1);
    const int channels = channel_groups * half;
    const int hidden = hidden_groups * half;
    const int heads = channel_groups * step;

    const double key_bytes = SwitchingKeyBytes(n, limbs, special);
    const int rot_count = half - 1;

    std::cout << "[rect] stage           : " << stage << std::endl;
    std::cout << "[rect] ring            : logN " << log_n << " (N = " << n
              << "), d = " << d << ", k = " << k << ", blocks k/2 = " << step
              << std::endl;
    std::cout << "[rect] chain           : " << limbs << " limbs of "
              << prime_bits << " bits, " << special << " special primes"
              << std::endl;
    std::cout << "[rect] width           : d_model " << channels << " ("
              << channel_groups << " group(s)), hidden " << hidden << ", "
              << heads << " heads of " << d << std::endl;
    std::cout << "[rect] activation      : " << (channel_groups * d)
              << " ciphertexts, against " << channels
              << " on the Algorithm 1 path" << std::endl;
    std::cout << "[rect] rotation keys   : " << rot_count << " indices, "
              << (rot_count * key_bytes / kGiB) << " GiB" << std::endl;
    std::cout << "[rect] Alg-5 plaintext : "
              << (static_cast<double>(limbs) * d * half * k * 8.0 / kGiB)
              << " GiB per upload" << std::endl;

    if (dry_run)
    {
        std::cout << "[rect] dry run, nothing allocated" << std::endl;
        return 0;
    }

    // -----------------------------------------------------------------------
    // Context and keys
    // -----------------------------------------------------------------------
    std::vector<int> q_bits(limbs, prime_bits);
    q_bits.front() = 60;
    std::vector<int> p_bits(special, 60);

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(static_cast<std::size_t>(n));
    context->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    context->generate();
    ReportMemory("context");

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEEncoder<S> encoder(context);

    heongpu::BatchMatrixLayout layout(n, d);
    const double scale = std::pow(2.0, prime_bits);
    heongpu::llama::Llama3BatchOperator batch(context, encoder, layout, scale);
    Rect op(context, encoder, layout, scale);

    std::vector<int> shifts = op.rotation_indices();
    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);
    ReportMemory("keys");

    // -----------------------------------------------------------------------
    // One activation and the weights the chosen stage needs
    // -----------------------------------------------------------------------
    std::mt19937_64 rng(20260806u);
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));

    heongpu::llama::RectActivation x = op.encrypt(
        random_matrix(static_cast<std::size_t>(d) * channels, 1.0, rng),
        channels, encryptor, scale);
    ReportMemory("activation");

    // -----------------------------------------------------------------------
    // The measured region
    // -----------------------------------------------------------------------
    std::cout << "[rect] --- measured region ---" << std::endl;
    cudaProfilerStart();
    RegionTimer whole;

    if (stage == "project")
    {
        const std::vector<double> w = random_matrix(
            static_cast<std::size_t>(channels) * channels, amp, rng);
        Range _r("stage.project");
        heongpu::llama::RectActivation out =
            op.project(x, w, channels, channels, "profile", galois);
        cudaDeviceSynchronize();
    }
    else if (stage == "bridge")
    {
        Range _r("stage.bridge");
        heongpu::llama::BatchActivation b = op.to_batch(x, 0, galois);
        std::vector<heongpu::llama::BatchActivation> groups;
        groups.push_back(std::move(b));
        for (int g = 1; g < channel_groups; ++g)
            groups.push_back(op.to_batch(x, g, galois));
        heongpu::llama::RectActivation back =
            op.from_batch(groups, channels, galois);
        cudaDeviceSynchronize();
    }
    else if (stage == "ccmm")
    {
        Range _r("stage.ccmm");
        heongpu::llama::BatchActivation q = op.to_batch(x, 0, galois);
        heongpu::llama::BatchActivation kb = op.to_batch(x, 0, galois);
        heongpu::llama::BatchActivation kt =
            batch.transpose(kb, "key", galois);
        heongpu::llama::BatchActivation s =
            op.matmul(q, kt, "score", galois, relin);
        cudaDeviceSynchronize();
    }
    else if (stage == "rmsnorm")
    {
        Rect::RectRMSNormConfig config;
        config.eps = 1e-5;
        config.sum_lo = channels * 0.20;
        config.sum_hi = channels * 0.50;
        const std::vector<double> weight =
            random_matrix(static_cast<std::size_t>(channels), 0.5, rng);
        Range _r("stage.rmsnorm");
        heongpu::llama::RectActivation out =
            op.rms_norm(x, weight, config, galois, relin);
        cudaDeviceSynchronize();
    }
    else if (stage == "ffn")
    {
        Rect::RectFeedForwardWeights w;
        w.gate = random_matrix(
            static_cast<std::size_t>(channels) * hidden, amp, rng);
        w.up = random_matrix(
            static_cast<std::size_t>(channels) * hidden, amp, rng);
        w.down = random_matrix(
            static_cast<std::size_t>(hidden) * channels, amp, rng);
        Rect::RectFeedForwardConfig config;
        config.in_channels = channels;
        config.hidden_channels = hidden;
        config.hidden_block_groups =
            EnvInt("HEONGPU_RECT_HIDDEN_BLOCK_GROUPS", 1);
        Range _r("stage.ffn");
        heongpu::llama::RectActivation out =
            op.feed_forward(x, w, config, galois, relin);
        cudaDeviceSynchronize();
    }
    else if (stage == "attention" || stage == "block")
    {
        Rect::RectAttentionWeights aw;
        aw.query = random_matrix(
            static_cast<std::size_t>(channels) * channels, amp, rng);
        aw.key = aw.query;
        aw.value = aw.query;
        aw.output = aw.query;

        Rect::RectAttentionConfig ac;
        ac.in_channels = channels;
        ac.heads = heads;
        ac.kv_heads = heads;
        ac.causal = true;
        ac.score_shift = 0.0;
        ac.softmax.bound = 8.0;
        ac.softmax.iterations = 2;
        ac.softmax.exp_degree = 31;
        ac.softmax.inverse_degree = 15;
        ac.softmax.inverse_newton = 2;

        if (stage == "attention")
        {
            Range _r("stage.attention");
            heongpu::llama::RectActivation out =
                op.attention(x, aw, ac, galois, relin);
            cudaDeviceSynchronize();
        }
        else
        {
            Rect::RectTransformerBlockWeights bw;
            bw.attention_norm =
                random_matrix(static_cast<std::size_t>(channels), 0.5, rng);
            bw.feed_forward_norm = bw.attention_norm;
            bw.attention = aw;
            bw.feed_forward.gate = random_matrix(
                static_cast<std::size_t>(channels) * hidden, amp, rng);
            bw.feed_forward.up = bw.feed_forward.gate;
            bw.feed_forward.down = random_matrix(
                static_cast<std::size_t>(hidden) * channels, amp, rng);

            Rect::RectTransformerBlockConfig bc;
            bc.attention_norm.sum_lo = channels * 0.20;
            bc.attention_norm.sum_hi = channels * 0.50;
            bc.feed_forward_norm = bc.attention_norm;
            bc.attention = ac;
            bc.feed_forward.in_channels = channels;
            bc.feed_forward.hidden_channels = hidden;
            bc.feed_forward.hidden_block_groups =
                EnvInt("HEONGPU_RECT_HIDDEN_BLOCK_GROUPS", 1);

            Range _r("stage.block");
            heongpu::llama::RectActivation out =
                op.transformer_block(x, bw, bc, galois, relin);
            cudaDeviceSynchronize();
        }
    }
    else
    {
        std::cerr << "[rect] unknown stage '" << stage << "'" << std::endl;
        return 2;
    }

    const double total = whole.ms();
    cudaProfilerStop();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[rect] " << stage << ": " << total << " ms" << std::endl;
    ReportMemory("run");
    return 0;
}

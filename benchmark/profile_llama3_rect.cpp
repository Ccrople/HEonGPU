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
// The block stage writes the six steps out rather than calling
// transformer_block, so that each one reports its wall time AND its level cost.
// The second number is the one that decides whether the chain is long enough,
// and it appears in no Nsight report.
//
// Shape and depth are both env-driven, since the two trade against each other
// here and neither is worth a rebuild: HEONGPU_RECT_{CHANNEL_GROUPS,
// HIDDEN_GROUPS, HIDDEN_BLOCK_GROUPS, KV_HEADS} set the width, and
// HEONGPU_RECT_{NORM_DEGREE, NORM_NEWTON, SOFTMAX_BOUND, SOFTMAX_ITERS,
// EXP_DEGREE, INVERSE_DEGREE, INVERSE_NEWTON, SILU_DEGREE} set what the
// non-linearities spend.
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

double EnvDouble(const char* name, double fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return fallback;
    return std::atof(v);
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

int DepthOf(const heongpu::llama::RectActivation& a)
{
    return a.empty() ? -1 : a.column.front().depth();
}

/// One line per step of the block: what it cost in wall time and what it cost
/// in levels. The second number is the one that decides whether the next step
/// runs at all, and it is not visible in an Nsight report.
void ReportStep(const char* name, double ms, int before, int after, int limbs)
{
    std::cout << "[rect] step " << std::setw(22) << std::left << name
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(10) << ms << " ms   depth " << std::setw(3)
              << before << " -> " << std::setw(3) << after << "  (" << (limbs - after)
              << " limbs left)" << std::endl;
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
    // A kv head still has to be a whole channel block, so the ratio is capped
    // by the block count and not by anything the model says.
    const int kv_heads = std::max(1, EnvInt("HEONGPU_RECT_KV_HEADS", heads));

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
              << heads << " heads of " << d << " over " << kv_heads
              << " kv heads" << std::endl;
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

    // HEONGPU_RECT_FUSED=1 takes every crossing in one level via the composed
    // BSGS map, so the same stages measure the trade against the staged path.
    const bool fused_crossings = EnvInt("HEONGPU_RECT_FUSED", 0) != 0;
    op.set_fused_crossings(fused_crossings);
    op.set_fused_plain_capacity(
        static_cast<std::size_t>(EnvInt("HEONGPU_RECT_FUSED_SETS", 8)));
    std::cout << "[rect] crossings       : "
              << (fused_crossings ? "fused, one level each"
                                  : "staged, 2/2/3/3 levels")
              << std::endl;

    // HEONGPU_RECT_HOIST=1 shares one key-switch decomposition across every
    // rotation train of a crossing and fuses each giant group's plaintext
    // products into one launch. Bit-identical results; only the work and the
    // launch count move.
    const bool hoisted_crossings = EnvInt("HEONGPU_RECT_HOIST", 0) != 0;
    op.set_hoisted_crossings(hoisted_crossings);
    std::cout << "[rect] hoisting        : "
              << (hoisted_crossings ? "on, one ModUp per rotation train"
                                    : "off, one ModUp per rotation")
              << std::endl;

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
        // HEONGPU_RECT_BRIDGE_REPS > 1 separates the one-time diagonal encode
        // from the recurring cost: rep 0 is cold, later reps run on the cached
        // plaintext sets, which is what a multi-block stack sees.
        Range _r("stage.bridge");
        const int reps = EnvInt("HEONGPU_RECT_BRIDGE_REPS", 1);
        for (int r = 0; r < reps; ++r)
        {
            RegionTimer rep;
            heongpu::llama::BatchActivation b = op.to_batch(x, 0, galois);
            std::vector<heongpu::llama::BatchActivation> groups;
            groups.push_back(std::move(b));
            for (int g = 1; g < channel_groups; ++g)
                groups.push_back(op.to_batch(x, g, galois));
            heongpu::llama::RectActivation back =
                op.from_batch(groups, channels, galois);
            cudaDeviceSynchronize();
            std::cout << "[rect] bridge rep " << r << ": " << rep.ms()
                      << " ms" << std::endl;
        }
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
        // Grouped-query attention: the key and value projections are narrower
        // by heads / kv_heads, which is where a real Llama-3 saves its work.
        const int kv_channels = kv_heads * d;
        Rect::RectAttentionWeights aw;
        aw.query = random_matrix(
            static_cast<std::size_t>(channels) * channels, amp, rng);
        aw.key = random_matrix(
            static_cast<std::size_t>(channels) * kv_channels, amp, rng);
        aw.value = aw.key;
        aw.output = aw.query;

        Rect::RectAttentionConfig ac;
        ac.in_channels = channels;
        ac.heads = heads;
        ac.kv_heads = kv_heads;
        ac.causal = EnvInt("HEONGPU_RECT_CAUSAL", 1) != 0;
        ac.score_shift = 0.0;
        ac.softmax.bound = EnvDouble("HEONGPU_RECT_SOFTMAX_BOUND", 8.0);
        ac.softmax.iterations = EnvInt("HEONGPU_RECT_SOFTMAX_ITERS", 2);
        ac.softmax.exp_degree = EnvInt("HEONGPU_RECT_EXP_DEGREE", 31);
        ac.softmax.inverse_degree = EnvInt("HEONGPU_RECT_INVERSE_DEGREE", 15);
        ac.softmax.inverse_newton = EnvInt("HEONGPU_RECT_INVERSE_NEWTON", 2);

        if (stage == "attention")
        {
            Range _r("stage.attention");
            heongpu::llama::RectActivation out =
                op.attention(x, aw, ac, galois, relin);
            cudaDeviceSynchronize();
            std::cout << "[rect] attention depth " << DepthOf(x) << " -> "
                      << DepthOf(out) << std::endl;
        }
        else
        {
            // The block is written out here rather than called through
            // Llama3RectOperator::transformer_block so that each step can be
            // timed and its LEVEL cost printed. The operations, their order and
            // their NVTX names are the same; the report cannot tell the two
            // apart, and the level trace is what says whether a chain is long
            // enough before an NTT fails deep inside the second half.
            const std::vector<double> norm_w =
                random_matrix(static_cast<std::size_t>(channels), 0.5, rng);
            Rect::RectRMSNormConfig nc;
            nc.eps = 1e-5;
            nc.sum_lo = channels * 0.20;
            nc.sum_hi = channels * 0.50;
            nc.degree = EnvInt("HEONGPU_RECT_NORM_DEGREE", 31);
            nc.newton_iterations = EnvInt("HEONGPU_RECT_NORM_NEWTON", 2);

            Rect::RectFeedForwardWeights fw;
            fw.gate = random_matrix(
                static_cast<std::size_t>(channels) * hidden, amp, rng);
            fw.up = fw.gate;
            fw.down = random_matrix(
                static_cast<std::size_t>(hidden) * channels, amp, rng);
            Rect::RectFeedForwardConfig fc;
            fc.in_channels = channels;
            fc.hidden_channels = hidden;
            fc.silu_degree = EnvInt("HEONGPU_RECT_SILU_DEGREE", 31);
            fc.hidden_block_groups =
                EnvInt("HEONGPU_RECT_HIDDEN_BLOCK_GROUPS", 1);

            Range _r("transformer_block");
            heongpu::llama::RectActivation stream;
            stream.rows = x.rows;
            stream.groups = x.groups;
            stream.channels = x.channels;
            stream.column = x.column;

            {
                Range _rh("transformer_block.attention_half");
                int before = DepthOf(stream);
                RegionTimer t;
                heongpu::llama::RectActivation normed =
                    op.rms_norm(stream, norm_w, nc, galois, relin);
                cudaDeviceSynchronize();
                ReportStep("attention_norm", t.ms(), before, DepthOf(normed),
                           limbs);

                before = DepthOf(normed);
                RegionTimer t2;
                heongpu::llama::RectActivation sub =
                    op.attention(normed, aw, ac, galois, relin);
                cudaDeviceSynchronize();
                ReportStep("attention", t2.ms(), before, DepthOf(sub), limbs);
                normed.column.clear();

                before = DepthOf(sub);
                RegionTimer t3;
                stream.column =
                    batch.arith().residual_add(stream.column, sub.column);
                cudaDeviceSynchronize();
                ReportStep("attention_residual", t3.ms(), before,
                           DepthOf(stream), limbs);
            }

            {
                Range _rh("transformer_block.feed_forward_half");
                int before = DepthOf(stream);
                RegionTimer t;
                heongpu::llama::RectActivation normed =
                    op.rms_norm(stream, norm_w, nc, galois, relin);
                cudaDeviceSynchronize();
                ReportStep("feed_forward_norm", t.ms(), before,
                           DepthOf(normed), limbs);

                before = DepthOf(normed);
                RegionTimer t2;
                heongpu::llama::RectActivation sub =
                    op.feed_forward(normed, fw, fc, galois, relin);
                cudaDeviceSynchronize();
                ReportStep("feed_forward", t2.ms(), before, DepthOf(sub),
                           limbs);
                normed.column.clear();

                before = DepthOf(sub);
                RegionTimer t3;
                stream.column =
                    batch.arith().residual_add(stream.column, sub.column);
                cudaDeviceSynchronize();
                ReportStep("feed_forward_residual", t3.ms(), before,
                           DepthOf(stream), limbs);
            }
            ReportMemory("block");
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

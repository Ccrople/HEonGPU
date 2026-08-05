// Nsight Systems capture target for the Llama-3 module on Kang's batch matrix
// multiplication -- the "oneMM" path.
//
// The counterpart of profile_llama3.cpp, which captures the same model on the
// Section 4.2 BSGS product. Both take their shape from the environment, both
// measure between cudaProfilerStart() and cudaProfilerStop(), and both leave
// context generation, key generation and a warm-up outside the measured
// region, so a report from one is directly comparable with a report from the
// other.
//
//   llama3_batch_profile <blocks>
//
// WHAT IS BEING MEASURED, AND WHY IT IS ONE BLOCK
// -----------------------------------------------
// A stack is T(B) = a + b B, and only b -- one transformer block -- is a
// property of the model rather than of its two ends. The batch path has no
// refresh in it yet (see the header: whether CKKS bootstrapping carries a
// matrix encryption unchanged is untested), so a stack cannot be run here
// honestly in any case. One block, at the real width, is the number that
// multiplies by thirty two.
//
// WHAT THE MATRIX ENCODING DOES TO THE COST OF A SHAPE
// ----------------------------------------------------
// It moves it, wholesale, from key material to ciphertexts, and that inverts
// which part of the model is expensive.
//
// The slot path spends one ciphertext per d x d block, so d_model = 4096 at
// d = 128 is 32 ciphertexts -- but every projection is a BSGS product whose
// rotations need a key apiece, and the published packing wants tens of
// gigabytes of them. The matrix encoding spends one ciphertext per CHANNEL, so
// the same activation is 4096 ciphertexts, but Algorithm 1 uses no rotations at
// all and the only Galois key left is the d - 1 shifts the bridge and the CMT
// share. That key list does not grow with the width of the model.
//
// So this executable reports both budgets before it allocates anything. At the
// published Llama-3-8B width the ciphertexts, not the keys, are what decides
// whether a shape fits, which is the opposite of the slot path's problem and
// the single most useful thing a dry run here can say.
//
// RING DEGREE
// -----------
// Unlike the slot path, d does NOT fix the ring. A slot-form ciphertext holds a
// whole d x d matrix and therefore needs d * d slots, which is what forces
// logN 15 at d = 128. A matrix encryption holds one column of d entries over
// the subring R_k, so it needs only d | N, and N sets the BATCH -- k/2 = N/(2d)
// independent instances carried at once -- rather than the shape. Raising N
// buys throughput per pass and costs memory per ciphertext; it does not change
// the model. HEONGPU_ONEMM_LOGN is therefore a genuine free parameter here, and
// the largest value that fits is the one to profile at.
//
// As in the slot-path target the plaintext values are random and the fitted
// intervals are nominal: nothing about the cost of a CKKS circuit depends on
// the numbers in it. The test suite is where correctness is established.

#include <heongpu/heongpu.hpp>

#include <cuda_profiler_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

constexpr auto S = heongpu::Scheme::CKKS;
namespace llama = heongpu::llama;

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

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

void ReportMemory(const char* stage)
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        return;
    std::cout << "[oneMM] memory after " << stage << ": "
              << (static_cast<double>(total_bytes - free_bytes) / kGiB)
              << " GiB used of " << (static_cast<double>(total_bytes) / kGiB)
              << " GiB" << std::endl;
}

/// Wall time of the measured region, for the same reason as the slot-path
/// target: most of the host time is blocking cudaMemcpy rather than
/// cudaDeviceSynchronize, so no single row of the Nsight API table is the
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

/// One block with no refresh in it consumes about sixty four levels, which is
/// what the transformer-block test pins at seventy limbs with every degree cut
/// to the bone. That is a property of the circuit and not of the width, so it
/// is the same here and it is the reason a ciphertext is expensive.
constexpr int kLimbs = 70;
constexpr int kPrimeBits = 40;
constexpr int kLogScale = 40;
constexpr int kDefaultSpecialPrimes = 7;
constexpr int kHammingWeight = 16;

/// The published Llama-3-8B configuration.
constexpr int kHidden = 4096;
constexpr int kIntermediate = 14336;
constexpr int kHeadDim = 128;
constexpr int kHeads = 32;
constexpr int kKVHeads = 8;
constexpr int kLayers = 32;

struct Shape
{
    int d = kHeadDim;              ///< Tokens per block; rank of R_N over R_k.
    int in_channels = kHidden;     ///< d_model.
    int hidden_channels = kIntermediate;
    int heads = kHeads;
    int kv_heads = kKVHeads;
    int head_dim = kHeadDim;
    int blocks = kLayers;
    int hidden_block = 128;        ///< SwiGLU streaming chunk.
    bool output_projection = true;
};

/// Ciphertext bytes: two polynomials of `limbs` limbs over Z_q, one 64-bit
/// word apiece. Ciphertexts descend the chain as they are used, so this is the
/// figure at the top and therefore an upper bound per resident ciphertext.
double CiphertextBytes(int n, int limbs)
{
    return 2.0 * limbs * n * 8.0;
}

/// A switching key holds two polynomials for each decomposition group at every
/// prime of the chain and the special primes:
///
///     2 * ceil(Q / P) * (Q + P) * N * 8 bytes
///
/// Adding special primes makes keys SMALLER, not larger -- the group count
/// falls faster than the prime count rises -- which is the only lever on key
/// memory that leaves the model alone.
double SwitchingKeyBytes(int n, int limbs, int special)
{
    const int groups = (limbs + special - 1) / special;
    return 2.0 * groups * (limbs + special) * n * 8.0;
}

/// Peak resident ciphertexts, which at these widths is what decides whether a
/// shape fits.
///
/// The attention half is the peak: the stream, the normalised copy, the query
/// at its full width and the key, the value and the transposed key at the
/// grouped-query width are all live at once. The SwiGLU half holds the stream
/// plus four ciphertexts per streamed hidden channel, which is why the chunk
/// exists.
long long PeakCiphertexts(const Shape& shape)
{
    const long long c = shape.in_channels;
    const long long q = static_cast<long long>(shape.heads) * shape.head_dim;
    const long long kv = static_cast<long long>(shape.kv_heads) * shape.head_dim;

    const long long attention = c + c + q + kv + kv + kv;
    const long long swiglu = c + c + 4LL * shape.hidden_block;
    return std::max(attention, swiglu);
}

std::vector<double> random_weight(int rows, int cols, double amplitude,
                                  std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> pick(-amplitude, amplitude);
    std::vector<double> out(static_cast<std::size_t>(rows) * cols);
    for (double& v : out)
        v = pick(rng);
    return out;
}

llama::Llama3BatchOperator::BatchTransformerBlockConfig
block_config(const Shape& shape)
{
    llama::Llama3BatchOperator::BatchTransformerBlockConfig config;

    // The degrees are the transformer-block test's, which are the ones cut far
    // enough for the chain to close without a refresh. A Chebyshev fit costs
    // the same whatever it is fitted over, so the intervals are nominal.
    config.attention_norm.eps = 1e-5;
    config.attention_norm.sum_lo = 0.1;
    config.attention_norm.sum_hi = 10.0 * shape.in_channels;
    config.attention_norm.degree = EnvInt("HEONGPU_ONEMM_NORM_DEGREE", 15);
    config.attention_norm.newton_iterations = 1;
    config.feed_forward_norm = config.attention_norm;

    config.attention.in_channels = shape.in_channels;
    config.attention.q_channels = shape.heads * shape.head_dim;
    config.attention.kv_channels = shape.kv_heads * shape.head_dim;
    config.attention.heads = shape.heads;
    config.attention.kv_heads = shape.kv_heads;
    config.attention.causal = true;
    config.attention.head_scale = 0.0; // 1/sqrt(head_dim).
    config.attention.score_shift = 0.0;
    config.attention.softmax.bound = 2.0;
    config.attention.softmax.iterations = 2;
    config.attention.softmax.exp_degree = EnvInt("HEONGPU_ONEMM_EXP_DEGREE", 7);
    config.attention.softmax.inverse_degree =
        EnvInt("HEONGPU_ONEMM_INV_DEGREE", 7);
    config.attention.softmax.inverse_newton = 1;

    config.feed_forward.in_channels = shape.in_channels;
    config.feed_forward.hidden_channels = shape.hidden_channels;
    config.feed_forward.silu_bound = 8.0;
    config.feed_forward.silu_degree = EnvInt("HEONGPU_ONEMM_SILU_DEGREE", 15);
    config.feed_forward.hidden_block = shape.hidden_block;

    return config;
}

llama::Llama3BatchOperator::BatchTransformerBlockWeights
block_weights(const Shape& shape, std::mt19937_64& rng)
{
    const int c = shape.in_channels;
    const int q = shape.heads * shape.head_dim;
    const int kv = shape.kv_heads * shape.head_dim;
    const int h = shape.hidden_channels;
    const double amp = 1.0 / std::sqrt(static_cast<double>(c));

    llama::Llama3BatchOperator::BatchTransformerBlockWeights weights;
    weights.attention_norm = random_weight(1, c, 0.5, rng);
    weights.feed_forward_norm = random_weight(1, c, 0.5, rng);
    weights.attention.query = random_weight(c, q, amp, rng);
    weights.attention.key = random_weight(c, kv, amp, rng);
    weights.attention.value = random_weight(c, kv, amp, rng);
    if (shape.output_projection)
        weights.attention.output = random_weight(q, c, amp, rng);
    weights.feed_forward.gate = random_weight(c, h, amp, rng);
    weights.feed_forward.up = random_weight(c, h, amp, rng);
    weights.feed_forward.down = random_weight(h, c, amp, rng);
    return weights;
}

} // namespace

int main(int argc, char* argv[])
{
    Shape shape;
    shape.blocks = (argc > 1) ? std::atoi(argv[1])
                              : EnvInt("HEONGPU_ONEMM_BLOCKS", 1);
    shape.d = EnvInt("HEONGPU_ONEMM_D", shape.d);
    shape.in_channels = EnvInt("HEONGPU_ONEMM_CHANNELS", shape.in_channels);
    shape.hidden_channels =
        EnvInt("HEONGPU_ONEMM_HIDDEN", shape.hidden_channels);
    shape.heads = EnvInt("HEONGPU_ONEMM_HEADS", shape.heads);
    shape.kv_heads = EnvInt("HEONGPU_ONEMM_KV_HEADS", shape.kv_heads);
    shape.head_dim = EnvInt("HEONGPU_ONEMM_HEAD_DIM", shape.head_dim);
    shape.hidden_block =
        EnvInt("HEONGPU_ONEMM_HIDDEN_BLOCK", shape.hidden_block);
    shape.output_projection =
        EnvInt("HEONGPU_ONEMM_OUTPUT_PROJECTION", 1) != 0;

    const int limbs = EnvInt("HEONGPU_ONEMM_LIMBS", kLimbs);
    const int special = EnvInt("HEONGPU_ONEMM_SPECIAL_PRIMES",
                               kDefaultSpecialPrimes);
    // d only has to divide N here; it does not set it. The default keeps one
    // ciphertext small enough that the full width fits, and the batch is
    // whatever that leaves.
    const int log_n = EnvInt("HEONGPU_ONEMM_LOGN", 11);
    const int poly_modulus_degree = 1 << log_n;

    if (shape.heads % std::max(1, shape.kv_heads) != 0)
    {
        std::cerr << "[oneMM] kv_heads must divide heads" << std::endl;
        return EXIT_FAILURE;
    }

    // What the shape costs, before anything is allocated.
    {
        const int k = poly_modulus_degree / shape.d;
        const long long peak = PeakCiphertexts(shape);
        const double ct = CiphertextBytes(poly_modulus_degree, limbs);
        const double key = SwitchingKeyBytes(poly_modulus_degree, limbs,
                                             special);
        // The bridge and the CMT share one list: the d - 1 non-zero multiples
        // of k/2. A rotation by -delta*(k/2) is the rotation by
        // (d - delta)*(k/2), because the slot index wraps at d*(k/2) = N/2, so
        // the negatives are already in the list and it is d - 1 long rather
        // than twice that. It does not grow with the width of the model.
        const long long key_count = shape.d - 1;

        const double host_weights =
            (2.0 * shape.in_channels *
                 static_cast<double>(shape.heads * shape.head_dim) +
             2.0 * shape.in_channels *
                 static_cast<double>(shape.kv_heads * shape.head_dim) +
             3.0 * shape.in_channels * shape.hidden_channels) *
            8.0;

        std::cout << "[oneMM] model: d_model=" << shape.in_channels
                  << " intermediate=" << shape.hidden_channels
                  << " heads=" << shape.heads << "/" << shape.kv_heads
                  << " head_dim=" << shape.head_dim << " tokens=" << shape.d
                  << " layers=" << shape.blocks << std::endl;
        std::cout << "[oneMM] ring: N=" << poly_modulus_degree << " d="
                  << shape.d << " k=" << k << " batch=" << (k / 2)
                  << " limbs=" << limbs << " special=" << special << std::endl;
        std::cout << "[oneMM] ciphertext: " << (ct / (1024.0 * 1024.0))
                  << " MiB; peak resident " << peak << " = "
                  << (peak * ct / kGiB) << " GiB" << std::endl;
        std::cout << "[oneMM] galois keys: " << key_count << " indices at "
                  << (key / (1024.0 * 1024.0)) << " MiB each = "
                  << (key_count * key / kGiB) << " GiB (relin key "
                  << (key / (1024.0 * 1024.0)) << " MiB)" << std::endl;
        std::cout << "[oneMM] estimated device total: "
                  << ((peak * ct + key_count * key + key) / kGiB) << " GiB"
                  << std::endl;
        std::cout << "[oneMM] host weights: " << (host_weights / kGiB)
                  << " GiB per block" << std::endl;
        std::cout << "[oneMM] swiglu chunk: " << shape.hidden_block
                  << " of " << shape.hidden_channels << " hidden channels, "
                  << ((shape.hidden_channels + shape.hidden_block - 1) /
                      shape.hidden_block)
                  << " passes" << std::endl;
    }

    if (EnvInt("HEONGPU_ONEMM_DRY_RUN", 0) != 0)
    {
        std::cout << "[oneMM] dry run, nothing allocated" << std::endl;
        return EXIT_SUCCESS;
    }

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    std::vector<int> logq{60};
    logq.insert(logq.end(), limbs - 1, kPrimeBits);
    std::vector<int> logp(special, 60);
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes(logq, logp);
    context->generate();

    const double scale = std::pow(2.0, kLogScale);
    const heongpu::BatchMatrixLayout layout(poly_modulus_degree, shape.d);

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context, kHammingWeight);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> public_key(context);
    keygen.generate_public_key(public_key, secret);

    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEEncoder<S> encoder(context);
    llama::Llama3BatchOperator ops(context, encoder, layout, scale);

    ReportMemory("context and encryption keys");

    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);
    ReportMemory("relinearisation key");

    // The whole Galois requirement of this path: the bridge's shifts and the
    // CMT's, which are the same set.
    std::vector<int> shifts = ops.rotation_indices();
    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    std::cout << "[oneMM] galois key generated: " << shifts.size()
              << " indices" << std::endl;
    ReportMemory("galois key");

    std::mt19937_64 rng(2026);

    const llama::Llama3BatchOperator::BatchTransformerBlockConfig config =
        block_config(shape);
    const llama::Llama3BatchOperator::BatchTransformerBlockWeights weights =
        block_weights(shape, rng);
    std::cout << "[oneMM] host weights built" << std::endl;

    // A matrix encryption carries exactly k/2 instances, and the profile has
    // to hand over all of them.
    auto fresh_input = [&]()
    {
        std::uniform_real_distribution<double> pick(-1.0, 1.0);
        std::vector<std::vector<double>> batch(layout.batch);
        for (auto& m : batch)
        {
            m.resize(static_cast<std::size_t>(shape.d) * shape.in_channels);
            for (double& v : m)
                v = pick(rng);
        }
        return ops.encrypt(batch, shape.d, shape.in_channels, encryptor,
                           scale);
    };

    llama::BatchActivation in = fresh_input();
    ReportMemory("encrypted input");

    // The warm-up touches every kernel the measured pass will, so the
    // measurement pays no module load or JIT. At the full width it is as
    // expensive as the measurement, so it is switchable; a cold capture's
    // first range then carries the load time.
    if (EnvInt("HEONGPU_ONEMM_WARMUP", 1) != 0)
    {
        Shape warm = shape;
        warm.in_channels = std::min(shape.in_channels, shape.d);
        warm.hidden_channels = std::min(shape.hidden_channels, shape.d);
        warm.heads = 1;
        warm.kv_heads = 1;
        warm.hidden_block = warm.hidden_channels;

        std::mt19937_64 warm_rng(7);
        const auto warm_config = block_config(warm);
        const auto warm_weights = block_weights(warm, warm_rng);

        std::uniform_real_distribution<double> pick(-1.0, 1.0);
        std::vector<std::vector<double>> batch(layout.batch);
        for (auto& m : batch)
        {
            m.resize(static_cast<std::size_t>(warm.d) * warm.in_channels);
            for (double& v : m)
                v = pick(rng);
        }
        llama::BatchActivation warm_in = ops.encrypt(
            batch, warm.d, warm.in_channels, encryptor, scale);
        llama::BatchActivation discard = ops.transformer_block(
            warm_in, warm_weights, warm_config, galois, relin);
        cudaDeviceSynchronize();
        std::cout << "[oneMM] warm-up done, " << discard.columns()
                  << " output columns" << std::endl;
        ReportMemory("warm-up");
    }

    cudaProfilerStart();
    RegionTimer timer;
    llama::BatchActivation out =
        ops.transformer_block(in, weights, config, galois, relin);
    cudaDeviceSynchronize();
    const float elapsed = timer.ms();
    cudaProfilerStop();

    ReportMemory("measured block");
    std::cout << "[oneMM] one transformer block: " << elapsed << " ms"
              << std::endl;
    std::cout << "[oneMM] " << out.columns() << " output columns, depth "
              << out.column.front().depth() << ", " << (limbs -
                  out.column.front().depth()) << " levels left" << std::endl;
    std::cout << "[oneMM] extrapolated " << shape.blocks << " blocks: "
              << (elapsed * shape.blocks / 1000.0) << " s (no refresh)"
              << std::endl;
    std::cout << "[oneMM] per instance: " << (elapsed / layout.batch)
              << " ms/block, batch " << layout.batch << std::endl;

    return EXIT_SUCCESS;
}

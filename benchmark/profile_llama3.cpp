// Whole-model Nsight Systems capture target for the Llama-3 module.
//
// One measured forward pass between cudaProfilerStart() and cudaProfilerStop(),
// so a report captured with
//
//   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop
//
// holds that pass alone: context generation, key generation and a warm-up pass
// all fall outside it. The warm-up is a whole forward over one block, which is
// every kernel the measured pass will use, so the measurement pays no module
// load or JIT.
//
// The point of the executable is the shape of the cost rather than any single
// number, so it takes the model's shape from the environment and the block
// count from the command line:
//
//   llama3_profile <blocks>
//
// Running it at one, two and three blocks fits
//
//   T(B) = a + b B
//
// where a is what the two ends of the model cost -- the embedding, the final
// norm and the head -- and b is what one transformer block costs. Only b
// matters for a model of thirty two, and the NVTX ranges inside it say what b
// is made of: "block.bootstrap_mid" and "block.bootstrap_seam" against
// "attention" and "feed_forward", and inside those the projections, the
// encrypted products and the polynomials.
//
// WHAT THE SHAPE DOES TO THE ANSWER
// ---------------------------------
// It decides it, so it is worth stating rather than defaulting quietly. A
// transformer block of C channel blocks costs 2C bootstraps -- one per block of
// the stream, at each of the two refreshes -- but its projections are block
// products and cost O(C^2) calls of Equation (5). So bootstrapping dominates a
// narrow model and the matrix work dominates a wide one, and reading either off
// a single shape says nothing about a model of a different width. The default
// here is four channel blocks against a SwiGLU widened by the Llama-3 ratio,
// which is wide enough for the projections to be visible and narrow enough to
// run in seconds; HEONGPU_LLAMA_CHANNEL_BLOCKS sweeps it.
//
// The chain, the degrees and the bootstrapping configuration are the ones
// Llama3StackEnv validates in the test suite, so the levels here are the levels
// that were measured to be sufficient, not a guess. The norm weights are left
// out exactly as they are there: an empty weight skips a rescale, so including
// them would deepen the circuit past what that fixture pins.
//
// The plaintext values are random and the fitted intervals are nominal. Nothing
// about the cost of a CKKS circuit depends on the numbers in it -- the level
// consumption is fixed by the shape and the degrees -- so this executable makes
// no attempt to be numerically meaningful. The test suite is where correctness
// is established.

#include <heongpu/heongpu.hpp>

#include <cuda_profiler_api.h>

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

bool EnvIs(const char* name, const char* value)
{
    const char* v = std::getenv(name);
    return v != nullptr && std::string(v) == value;
}

/// Free and total device memory, so a shape that does not fit says where it
/// stopped fitting rather than dying inside the allocator.
void ReportMemory(const char* stage)
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        return;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    std::cout << "[profile] memory after " << stage << ": "
              << (static_cast<double>(total_bytes - free_bytes) / gib)
              << " GiB used of "
              << (static_cast<double>(total_bytes) / gib) << " GiB"
              << std::endl;
}

/// Wall time of the measured region.
///
/// The same reasoning as the CKKS profile target: most of the host time here is
/// blocking cudaMemcpy rather than cudaDeviceSynchronize, so no single row of
/// the Nsight API table is the answer and events bracketing the region are.
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

/// The chain Llama3StackEnv runs a stack on: sixty limbs of fifty bits over a
/// sixty bit bottom, which leaves thirty five after a refresh against the
/// attention half's thirty two. See the fixture for why none of these is free.
/// The ring, which HEONGPU_LLAMA_LOGN raises.
///
/// Four thousand and ninety six is what the fixture uses, and it is chosen for
/// the level budget rather than for the GPU: at that degree a CKKS kernel here
/// is a couple of thousand threads, which is a small fraction of an A6000, so
/// the pass is bound by kernel launches rather than by arithmetic. Raising it
/// is how one finds out which of the numbers below are properties of the
/// circuit and which are properties of a ring too small to fill the device.
constexpr int kDefaultLogN = 12;
constexpr int kLimbs = 60;
/// Special primes, which set the key-switching decomposition: the chain is cut
/// into ceil(limbs / this) groups and a rotation key holds two polynomials per
/// group at every prime. See the key-size arithmetic in main().
constexpr int kSpecialPrimes = 3;
constexpr int kPrimeBits = 50;
constexpr int kLogScale = 50;
constexpr int kCtoSPiece = 3;
constexpr int kStoCPiece = 3;
constexpr int kTaylor = 11;
constexpr int kHammingWeight = 16;

struct Shape
{
    int d = 8;              ///< Rows and columns of one packed matrix.
    int channel_blocks = 4; ///< d_model = channel_blocks * d.
    int hidden_blocks = 14; ///< SwiGLU width; Llama-3 widens by 3.5.
    int heads = 4;
    int kv_heads = 2;       ///< Grouped-query attention.
    int token_blocks = 1;   ///< Sequence length = token_blocks * d.
    int vocab_blocks = 2;   ///< 0 leaves the embedding and the head out.
    int blocks = 1;         ///< Transformer blocks in the stack.
};

/// The published Llama-3-8B configuration, in the units this module packs in.
///
/// A ciphertext holds d channels of d tokens, so d is set by the head
/// dimension -- 128, which makes one head exactly one block and RoPE a
/// rotation inside it -- and every other width is that model dimension over
/// d. This costs d * d = 16384 slots and therefore logN 15, which is the price
/// of the real shape rather than a choice: at any smaller ring a head spans
/// several blocks and the block counts below all change.
///
///   hidden 4096 / 128 = 32 channel blocks
///   intermediate 14336 / 128 = 112 hidden blocks
///   32 query heads over 8 key/value heads
///   vocabulary 128256 / 128 = 1002 blocks at each end
///   32 transformer blocks
///
/// HEONGPU_LLAMA_* still overrides any single field afterwards, which is what
/// makes the parts of the full shape that do not fit on one card separable
/// from the parts that do.
constexpr int kLlama38BHidden = 4096;
constexpr int kLlama38BIntermediate = 14336;
constexpr int kLlama38BHeadDim = 128;
constexpr int kLlama38BHeads = 32;
constexpr int kLlama38BKVHeads = 8;
constexpr int kLlama38BLayers = 32;
constexpr int kLlama38BVocab = 128256;

Shape Llama38BShape()
{
    Shape shape;
    shape.d = kLlama38BHeadDim;
    shape.channel_blocks = kLlama38BHidden / kLlama38BHeadDim;
    shape.hidden_blocks = kLlama38BIntermediate / kLlama38BHeadDim;
    shape.heads = kLlama38BHeads;
    shape.kv_heads = kLlama38BKVHeads;
    shape.token_blocks = 1;
    shape.vocab_blocks =
        (kLlama38BVocab + kLlama38BHeadDim - 1) / kLlama38BHeadDim;
    shape.blocks = kLlama38BLayers;
    return shape;
}

/// Calls of Equation (5) in one transformer block, and refreshes.
///
/// A projection from I input blocks to O output blocks is O * I block
/// products, and grouped-query attention narrows the key and the value by the
/// number of query heads sharing one of them. The stream is C blocks and each
/// is refreshed at both of the block's two seams.
struct BlockCost
{
    long long pcmm = 0;
    long long bootstraps = 0;
};

BlockCost block_cost(const Shape& shape)
{
    const long long c = shape.channel_blocks;
    const long long h = shape.hidden_blocks;
    const long long kv = c / (shape.heads / shape.kv_heads);
    const long long t = shape.token_blocks;

    BlockCost cost;
    // query + output, key + value, then gate + up + down.
    cost.pcmm = (2 * c * c + 2 * kv * c + 3 * h * c) * t;
    cost.bootstraps = 2 * c * t;
    return cost;
}

std::vector<double> random_matrix(int d, double amplitude, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> pick(-amplitude, amplitude);
    std::vector<double> out(static_cast<std::size_t>(d) * d);
    for (double& v : out)
        v = pick(rng);
    return out;
}

llama::BlockMatrix random_block_matrix(int out_blocks, int in_blocks, int d,
                                       double amplitude, std::mt19937_64& rng)
{
    llama::BlockMatrix m(out_blocks, in_blocks);
    for (int i = 0; i < out_blocks; i++)
        for (int j = 0; j < in_blocks; j++)
            m.at(i, j) = random_matrix(d, amplitude, rng);
    return m;
}

/// One block's config, at the degrees the stack fixture validates.
///
/// The intervals are nominal, for the reason given at the top of the file: a
/// Chebyshev fit costs the same whatever it is fitted over.
llama::Llama3Operator::TransformerBlockConfig
block_config(const Shape& shape, const llama::MatrixLayout& layout)
{
    llama::Llama3Operator::TransformerBlockConfig config;

    const int channels = shape.d * shape.channel_blocks;
    for (llama::Llama3Operator::RMSNormConfig* norm :
         {&config.attention_norm, &config.feed_forward_norm})
    {
        norm->stride = layout.d * layout.batch;
        norm->count = layout.d;
        norm->channels = channels;
        norm->token_blocks = shape.token_blocks;
        norm->eps = 1e-5;
        norm->sum_lo = 0.1;
        norm->sum_hi = 10.0 * channels;
        norm->degree = 31;
        // Both norms skip the Newton step, as they do in the fixture: a
        // sublayer follows each of them and wants the level more.
        norm->newton_iterations = 0;
    }

    config.attention.layout = layout;
    config.attention.heads = shape.heads;
    config.attention.kv_heads = shape.kv_heads;
    config.attention.token_blocks = shape.token_blocks;
    config.attention.causal = true;
    config.attention.rope = false;
    config.attention.score_shift = 0.0;
    config.attention.softmax.bound = 2.0;
    config.attention.softmax.iterations = 1;
    config.attention.softmax.exp_degree = 15;
    config.attention.softmax.inverse_degree = 15;
    config.attention.softmax.inverse_newton = 1;

    config.feed_forward.layout = layout;
    config.feed_forward.token_blocks = shape.token_blocks;
    config.feed_forward.silu_bound = 8.0;
    config.feed_forward.silu_degree = 31;

    config.bootstrap = true;
    return config;
}

llama::Llama3Operator::TransformerBlockWeights
block_weights(const Shape& shape, std::mt19937_64& rng)
{
    const int d = shape.d;
    const double amplitude = 1.0 / std::sqrt(static_cast<double>(d));
    const int c = shape.channel_blocks;
    // Grouped-query attention narrows the key and the value by exactly the
    // number of query heads that share one of them.
    const int kv = c / (shape.heads / shape.kv_heads);

    llama::Llama3Operator::TransformerBlockWeights weights;
    weights.attention.query = random_block_matrix(c, c, d, amplitude, rng);
    weights.attention.key = random_block_matrix(kv, c, d, amplitude, rng);
    weights.attention.value = random_block_matrix(kv, c, d, amplitude, rng);
    // W_o costs the attention half a tau and a projection, which is two of the
    // levels a refresh hands back. The block tests leave it out and the
    // measured thirty two is without it; a real block has it, so it is on here
    // and switchable rather than assumed.
    if (EnvInt("HEONGPU_LLAMA_OUTPUT_PROJECTION", 1) != 0)
        weights.attention.output = random_block_matrix(c, c, d, amplitude, rng);
    weights.feed_forward.gate =
        random_block_matrix(shape.hidden_blocks, c, d, amplitude, rng);
    weights.feed_forward.up =
        random_block_matrix(shape.hidden_blocks, c, d, amplitude, rng);
    weights.feed_forward.down =
        random_block_matrix(c, shape.hidden_blocks, d, amplitude, rng);
    return weights;
}

} // namespace

int main(int argc, char* argv[])
{
    // The preset supplies the published Llama-3-8B widths as defaults; the
    // command line still names the block count and every HEONGPU_LLAMA_* below
    // still overrides one field of it.
    const bool full = EnvIs("HEONGPU_LLAMA_PRESET", "8b");
    Shape shape = full ? Llama38BShape() : Shape();

    shape.blocks = (argc > 1) ? std::atoi(argv[1])
                              : EnvInt("HEONGPU_LLAMA_BLOCKS", shape.blocks);
    shape.d = EnvInt("HEONGPU_LLAMA_D", shape.d);
    shape.channel_blocks =
        EnvInt("HEONGPU_LLAMA_CHANNEL_BLOCKS", shape.channel_blocks);
    shape.hidden_blocks =
        EnvInt("HEONGPU_LLAMA_HIDDEN_BLOCKS", shape.hidden_blocks);
    shape.heads = EnvInt("HEONGPU_LLAMA_HEADS", shape.heads);
    shape.kv_heads = EnvInt("HEONGPU_LLAMA_KV_HEADS", shape.kv_heads);
    shape.token_blocks =
        EnvInt("HEONGPU_LLAMA_TOKEN_BLOCKS", shape.token_blocks);
    shape.vocab_blocks =
        EnvInt("HEONGPU_LLAMA_VOCAB_BLOCKS", shape.vocab_blocks);

    if (shape.blocks < 1)
    {
        std::cerr << "usage: " << argv[0] << " <blocks>" << std::endl;
        return EXIT_FAILURE;
    }

    // A ciphertext is d by d, so the full shape sets the ring rather than the
    // other way round: d = 128 needs 16384 slots and therefore logN 15.
    int default_log_n = kDefaultLogN;
    if (full)
    {
        default_log_n = 1;
        while ((1 << (default_log_n - 1)) < shape.d * shape.d)
            default_log_n++;
    }
    const int log_n = EnvInt("HEONGPU_LLAMA_LOGN", default_log_n);
    const int poly_modulus_degree = 1 << log_n;

    // What the shape costs, before anything is allocated. A configuration that
    // will not fit or will not finish should say so here rather than after an
    // hour of key generation.
    {
        const BlockCost cost = block_cost(shape);
        const long long vocab_pcmm =
            2LL * shape.vocab_blocks * shape.channel_blocks * shape.token_blocks;
        std::cout << "[profile] model: hidden=" << (shape.d * shape.channel_blocks)
                  << " intermediate=" << (shape.d * shape.hidden_blocks)
                  << " heads=" << shape.heads << "/" << shape.kv_heads
                  << " head_dim=" << shape.d
                  << " seq=" << (shape.d * shape.token_blocks)
                  << " vocab=" << (shape.d * shape.vocab_blocks)
                  << " layers=" << shape.blocks << std::endl;
        // The weights are host doubles, one d by d block per call of Equation
        // (5), and at the full width they are the first thing to run out
        // rather than anything on the device.
        const double block_bytes =
            static_cast<double>(cost.pcmm / (shape.token_blocks > 0
                                                 ? shape.token_blocks
                                                 : 1)) *
            shape.d * shape.d * 8.0;
        const double vocab_bytes = 2.0 * shape.vocab_blocks *
                                   shape.channel_blocks * shape.d * shape.d *
                                   8.0;
        const double gib = 1024.0 * 1024.0 * 1024.0;
        std::cout << "[profile] host weights: "
                  << (block_bytes * shape.blocks + vocab_bytes) / gib
                  << " GiB (" << (block_bytes / gib) << " per block, "
                  << (vocab_bytes / gib) << " for the two ends)" << std::endl;
        std::cout << "[profile] per block: " << cost.pcmm << " pcmm, "
                  << cost.bootstraps << " bootstraps ("
                  << (cost.bootstraps > 0
                          ? static_cast<double>(cost.pcmm) / cost.bootstraps
                          : 0.0)
                  << ":1); whole pass: "
                  << (cost.pcmm * shape.blocks + vocab_pcmm) << " pcmm, "
                  << (cost.bootstraps * shape.blocks - shape.channel_blocks *
                                                           shape.token_blocks)
                  << " bootstraps" << std::endl;
    }

    // What the rotation keys will cost, before any of them exists.
    //
    // A key holds two polynomials for each of the decomposition groups at every
    // prime of the chain and the special primes, so one of them is
    //
    //   2 * ceil(Q / P) * (Q + P) * N * 8 bytes
    //
    // which at logN 15 and sixty limbs is 630 MiB apiece. The index list is a
    // property of the shape alone, so both numbers are available here and a
    // shape whose keys do not fit says so in a second rather than after five
    // minutes of key generation.
    {
        const int slots = poly_modulus_degree / 2;
        if (slots % (shape.d * shape.d) == 0)
        {
            const llama::MatrixLayout probe_layout(
                shape.d, slots / (shape.d * shape.d));
            llama::Llama3Operator::ModelConfig probe;
            probe.layout = probe_layout;
            probe.token_blocks = shape.token_blocks;
            probe.stack.bootstrap_between_blocks = true;
            for (int b = 0; b < shape.blocks; b++)
                probe.stack.blocks.push_back(block_config(shape, probe_layout));
            probe.final_norm = probe.stack.blocks.front().attention_norm;

            const std::size_t count =
                llama::Llama3Operator::model_rotation_indices(probe).size();
            const int groups = (kLimbs + kSpecialPrimes - 1) / kSpecialPrimes;
            const double key_bytes = 2.0 * groups * (kLimbs + kSpecialPrimes) *
                                     poly_modulus_degree * 8.0;
            const double gib = 1024.0 * 1024.0 * 1024.0;
            std::cout << "[profile] rotation keys: " << count << " indices at "
                      << (key_bytes / (1024.0 * 1024.0)) << " MiB each = "
                      << (count * key_bytes / gib)
                      << " GiB, before the bootstrapping key" << std::endl;
        }
    }

    if (EnvInt("HEONGPU_LLAMA_DRY_RUN", 0) != 0)
    {
        std::cout << "[profile] dry run, nothing allocated" << std::endl;
        return EXIT_SUCCESS;
    }

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    std::vector<int> logq{60};
    logq.insert(logq.end(), kLimbs - 1, kPrimeBits);
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes(logq, {60, 60, 60});
    context->generate();

    const double scale = std::pow(2.0, kLogScale);

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context, kHammingWeight);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> public_key(context);
    keygen.generate_public_key(public_key, secret);

    heongpu::HEEncryptor<S> encryptor(context, public_key);
    heongpu::HEEncoder<S> encoder(context);
    llama::Llama3Operator ops(context, encoder, scale);

    const int slots = encoder.slot_count();
    if (slots % (shape.d * shape.d) != 0)
    {
        std::cerr << "[profile] d * d must divide the slot count" << std::endl;
        return EXIT_FAILURE;
    }
    const llama::MatrixLayout layout(shape.d, slots / (shape.d * shape.d));

    ReportMemory("context and encryption keys");

    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);
    ReportMemory("relinearisation key");

    heongpu::BootstrappingConfig boot_config(kCtoSPiece, kStoCPiece, kTaylor,
                                             true);
    ops.generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);
    std::vector<int> boot_shifts = ops.bootstrapping_key_indexs();
    heongpu::Galoiskey<S> boot_key(context, boot_shifts);
    keygen.generate_galois_key(boot_key, secret);
    ReportMemory("bootstrapping key");

    std::mt19937_64 rng(2026);

    llama::Llama3Operator::ModelConfig config;
    config.layout = layout;
    config.token_blocks = shape.token_blocks;
    config.stack.bootstrap_between_blocks = true;
    for (int b = 0; b < shape.blocks; b++)
        config.stack.blocks.push_back(block_config(shape, layout));
    config.final_norm = config.stack.blocks.front().attention_norm;
    // Zero, as in the model test: the last block hands back what its SwiGLU
    // half left, and the head has to fit in that. A Newton step here needs a
    // refresh in front of it, which is what bootstrap_before_head is for.
    config.final_norm.newton_iterations =
        EnvInt("HEONGPU_LLAMA_FINAL_NEWTON", 0);
    config.bootstrap_before_head = EnvInt("HEONGPU_LLAMA_BOOT_HEAD", 0) != 0;

    llama::Llama3Operator::ModelWeights weights;
    for (int b = 0; b < shape.blocks; b++)
        weights.blocks.push_back(block_weights(shape, rng));
    if (shape.vocab_blocks > 0)
    {
        const double amplitude = 1.0 / std::sqrt(static_cast<double>(shape.d));
        weights.embedding = random_block_matrix(
            shape.channel_blocks, shape.vocab_blocks, shape.d, amplitude, rng);
        weights.head = random_block_matrix(
            shape.vocab_blocks, shape.channel_blocks, shape.d, amplitude, rng);
    }

    std::vector<int> shifts =
        llama::Llama3Operator::model_rotation_indices(config);
    heongpu::Galoiskey<S> galois_key(context, shifts);
    keygen.generate_galois_key(galois_key, secret);
    ReportMemory("model rotation key");

    // The entry blocks: the one-hot columns when there is an embedding table,
    // and the activation itself when there is not.
    const int entry_blocks =
        (shape.vocab_blocks > 0 ? shape.vocab_blocks : shape.channel_blocks) *
        shape.token_blocks;
    auto fresh_input = [&]()
    {
        std::vector<heongpu::Ciphertext<S>> in;
        in.reserve(entry_blocks);
        std::uniform_real_distribution<double> pick(-1.0, 1.0);
        for (int i = 0; i < entry_blocks; i++)
        {
            std::vector<double> values(slots);
            for (double& v : values)
                v = pick(rng);
            heongpu::Plaintext<S> plain(context);
            encoder.encode(plain, values, scale);
            heongpu::Ciphertext<S> cipher(context);
            encryptor.encrypt(cipher, plain);
            in.push_back(std::move(cipher));
        }
        return in;
    };

    std::cout << "[profile] N=" << poly_modulus_degree << " slots=" << slots
              << " limbs=" << kLimbs << " d=" << layout.d
              << " batch=" << layout.batch << std::endl;
    std::cout << "[profile] blocks=" << shape.blocks
              << " channel_blocks=" << shape.channel_blocks
              << " hidden_blocks=" << shape.hidden_blocks
              << " heads=" << shape.heads << " kv_heads=" << shape.kv_heads
              << " token_blocks=" << shape.token_blocks
              << " vocab_blocks=" << shape.vocab_blocks << std::endl;
    std::cout << "[profile] galois keys: " << shifts.size() << " model, "
              << boot_shifts.size() << " bootstrapping" << std::endl;

    // The warm-up is a whole forward over one block, which touches every kernel
    // the measured pass will. A forward consumes its input, so it gets its own.
    //
    // At the full width one block is minutes rather than milliseconds and the
    // warm-up doubles a single-block capture, so it is switchable. Turning it
    // off puts module load and JIT inside the measured region; the ranges that
    // pay for it are whichever ran first, so read a cold capture's first block
    // with that in mind.
    if (EnvInt("HEONGPU_LLAMA_WARMUP", 1) != 0)
    {
        llama::Llama3Operator::ModelConfig warm_config = config;
        warm_config.stack.blocks.resize(1);
        llama::Llama3Operator::ModelWeights warm_weights = weights;
        warm_weights.blocks.resize(1);

        std::vector<heongpu::Ciphertext<S>> warm_in = fresh_input();
        std::vector<heongpu::Ciphertext<S>> discard = ops.forward(
            warm_in, warm_weights, warm_config, galois_key, boot_key, relin);
        cudaDeviceSynchronize();
        std::cout << "[profile] warm-up done, one block, "
                  << discard.size() << " output blocks" << std::endl;
    }

    std::vector<heongpu::Ciphertext<S>> in = fresh_input();

    cudaProfilerStart();
    RegionTimer timer;
    std::vector<heongpu::Ciphertext<S>> out =
        ops.forward(in, weights, config, galois_key, boot_key, relin);
    cudaDeviceSynchronize();
    const float elapsed = timer.ms();
    cudaProfilerStop();

    std::cout << "[profile] forward over " << shape.blocks << " block(s): "
              << elapsed << " ms" << std::endl;
    std::cout << "[profile] " << out.size() << " output blocks, depth "
              << out.front().depth() << ", " << (kLimbs - out.front().depth())
              << " levels left" << std::endl;

    return EXIT_SUCCESS;
}

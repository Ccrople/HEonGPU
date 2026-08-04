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
    Shape shape;
    shape.blocks = (argc > 1) ? std::atoi(argv[1])
                              : EnvInt("HEONGPU_LLAMA_BLOCKS", 1);
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

    const int log_n = EnvInt("HEONGPU_LLAMA_LOGN", kDefaultLogN);
    const int poly_modulus_degree = 1 << log_n;

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

    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);

    heongpu::BootstrappingConfig boot_config(kCtoSPiece, kStoCPiece, kTaylor,
                                             true);
    ops.generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);
    std::vector<int> boot_shifts = ops.bootstrapping_key_indexs();
    heongpu::Galoiskey<S> boot_key(context, boot_shifts);
    keygen.generate_galois_key(boot_key, secret);

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

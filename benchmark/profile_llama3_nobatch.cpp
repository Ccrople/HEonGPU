// Nsight Systems capture target for the Llama-3 projection path with the batch
// axis spent on ONE input rather than on k/2 of them.
//
//   llama3_nobatch_profile
//
// WHY THIS EXISTS
// ---------------
// profile_llama3_batch.cpp measures Kang's Algorithm 1, where the k/2 packed
// matrices are k/2 INDEPENDENT INPUTS. Every projection is then free of key
// switching, and the batch is pure throughput: k/2 users' worth of work for one
// user's worth of latency. That is the right answer when there are k/2 users.
//
// It is the wrong answer for one. A single input leaves k/2 - 1 of the slots
// empty, so the same latency buys a k/2-th of the work, and the projection is
// still one ciphertext per CHANNEL -- the encoding that [[onemm-memory-wall]]
// found ran out of ciphertext memory before it ran out of anything else.
//
// Algorithm 5 spends the batch axis on the input instead. A d x (N/2) matrix is
// k/2 blocks of d x d, one per batch slot, and the block products are summed by
// Theorem 3. One input then occupies the whole ring: an activation of N/2
// channels is d ciphertexts rather than N/2 of them, a factor of k/2 fewer.
//
// What it costs is the summation, and the summation is a CMT. Algorithm 1 needs
// no key switching at all; Algorithm 5 needs N/2 of them per call, plus the
// N/2 - 1 Galois keys the CMT at k = 2 asks for. Whether that trade is worth
// taking is exactly what this executable measures.
//
// WHAT IS AND IS NOT MEASURED
// ---------------------------
// The seven projections of one transformer block, at the published Llama-3-8B
// width, each in its own NVTX range. That is the matrix path of a block and it
// is where Algorithm 5 differs from Algorithm 1; it is not a whole block. The
// non-linear layers -- RMSNorm, the SoftMax, SiLU -- and the two encrypted
// products of attention are NOT here, because on this path they need the slot
// form and the slot form needs the block index moved off the Y axis, which is a
// second homomorphic transform that is not written yet. Neither is
// bootstrapping. Anything this reports is a projection cost and should be read
// as one.
//
// As in the sibling targets the plaintext values are random and the fitted
// intervals are nominal: nothing about the cost of a CKKS circuit depends on
// the numbers in it.

#include <heongpu/heongpu.hpp>

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

constexpr auto S = heongpu::Scheme::CKKS;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

int EnvInt(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return fallback;
    return std::atoi(v);
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
    std::cout << "[nobatch] memory after " << stage << ": "
              << (static_cast<double>(total_bytes - free_bytes) / kGiB)
              << " GiB used of " << (static_cast<double>(total_bytes) / kGiB)
              << " GiB" << std::endl;
}

double SwitchingKeyBytes(int n, int limbs, int special)
{
    const int groups = (limbs + special - 1) / special;
    return 2.0 * groups * (limbs + special) * n * 8.0;
}

double CiphertextBytes(int n, int limbs) { return 2.0 * limbs * n * 8.0; }

/// The published Llama-3-8B configuration.
struct Shape
{
    int d = 128;
    int channels = 4096;   ///< d_model.
    int hidden = 14336;
    int heads = 32;
    int kv_heads = 8;
    int head_dim = 128;
    int layers = 32;
};

/// One projection of the block, named for the sublayer it belongs to so a
/// capture can answer "which layer" and not merely "which kernel".
struct Projection
{
    const char* name;
    int in_channels;
    int out_channels;
};

std::vector<Projection> projections(const Shape& shape)
{
    const int q = shape.heads * shape.head_dim;
    const int kv = shape.kv_heads * shape.head_dim;
    return {{"attention.q_proj", shape.channels, q},
            {"attention.k_proj", shape.channels, kv},
            {"attention.v_proj", shape.channels, kv},
            {"attention.o_proj", q, shape.channels},
            {"ffn.gate_proj", shape.channels, shape.hidden},
            {"ffn.up_proj", shape.channels, shape.hidden},
            {"ffn.down_proj", shape.hidden, shape.channels}};
}

std::vector<double> random_weight(std::size_t n, double amplitude,
                                  std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> pick(-amplitude, amplitude);
    std::vector<double> out(n);
    for (double& v : out)
        v = pick(rng);
    return out;
}

// ---------------------------------------------------------------------------
// Algorithm 5: the block index rides the Y axis at both ends
// ---------------------------------------------------------------------------
//
// Both arrangements below are host-side only, which is the whole point: the
// conversion that makes Algorithm 5 chainable costs no homomorphic work at all,
// only a different order of writing the coefficients down.

/// Activation: X is d(tokens) x half(channels); channel c = t*d + j lands at
/// coefficient i + d*t of column ciphertext j.
std::vector<int64_t> stream_coefficients(int d, int half, int k, double scale,
                                         std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> pick(-1.0, 1.0);
    const int blocks = half / d;
    std::vector<int64_t> coeffs(static_cast<std::size_t>(d) * d * k, 0);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
        {
            int64_t* e = coeffs.data() + (static_cast<std::size_t>(i) * d + j) * k;
            for (int t = 0; t < blocks; ++t)
                e[t] = static_cast<int64_t>(std::llround(pick(rng) * scale));
        }
    return coeffs;
}

/// Weight: block row t of W sits at Y^{k-t} with a minus sign, so that the
/// constant coefficient of the R_k product is already the contraction over t.
/// Y^k = -1 is what turns the reversal into an addition.
std::vector<int64_t> weight_coefficients(int d, int half, int k, double scale,
                                         std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> pick(-1.0, 1.0);
    const double amp = 1.0 / std::sqrt(static_cast<double>(half));
    const int blocks = half / d;
    std::vector<int64_t> coeffs(static_cast<std::size_t>(d) * half * k, 0);
    for (int j = 0; j < d; ++j)
        for (int c = 0; c < half; ++c)
        {
            int64_t* e =
                coeffs.data() + (static_cast<std::size_t>(j) * half + c) * k;
            e[0] = static_cast<int64_t>(std::llround(pick(rng) * amp * scale));
            for (int t = 1; t < blocks; ++t)
                e[k - t] = -static_cast<int64_t>(
                    std::llround(pick(rng) * amp * scale));
        }
    return coeffs;
}

} // namespace

int main(int argc, char* argv[])
{
    (void) argc;
    (void) argv;

    Shape shape;
    shape.d = EnvInt("HEONGPU_NOBATCH_D", shape.d);
    shape.channels = EnvInt("HEONGPU_NOBATCH_CHANNELS", shape.channels);
    shape.hidden = EnvInt("HEONGPU_NOBATCH_HIDDEN", shape.hidden);
    shape.heads = EnvInt("HEONGPU_NOBATCH_HEADS", shape.heads);
    shape.kv_heads = EnvInt("HEONGPU_NOBATCH_KV_HEADS", shape.kv_heads);
    shape.head_dim = EnvInt("HEONGPU_NOBATCH_HEAD_DIM", shape.head_dim);
    shape.layers = EnvInt("HEONGPU_NOBATCH_LAYERS", shape.layers);

    const int algorithm = EnvInt("HEONGPU_NOBATCH_ALG", 5);
    const int log_n = EnvInt("HEONGPU_NOBATCH_LOGN", 12);
    const int limbs = EnvInt("HEONGPU_NOBATCH_LIMBS", 35);
    const int special = EnvInt("HEONGPU_NOBATCH_SPECIAL_PRIMES", 7);
    const int prime_bits = EnvInt("HEONGPU_NOBATCH_PRIME_BITS", 40);
    const bool dry_run = EnvInt("HEONGPU_NOBATCH_DRY_RUN", 0) != 0;

    const int n = 1 << log_n;
    const int d = shape.d;
    const int k = n / d;
    const int batch = k / 2;
    const int half = n / 2;

    // A group is what one Algorithm 5 call consumes and produces: N/2 channels
    // in d ciphertexts. A width that is not a whole number of groups is padded
    // to one, and the padding is paid for, so it is reported rather than
    // hidden.
    auto groups = [&](int channels) { return (channels + half - 1) / half; };

    const std::vector<Projection> projs = projections(shape);

    long long rect_calls = 0, alg1_columns = 0;
    for (const Projection& p : projs)
    {
        rect_calls += static_cast<long long>(groups(p.in_channels)) *
                      groups(p.out_channels);
        alg1_columns += p.out_channels;
    }

    const int rot_count = (algorithm == 5) ? (half - 1) : (d - 1);
    const double key_bytes = SwitchingKeyBytes(n, limbs, special);

    std::cout << "[nobatch] algorithm      : " << algorithm << " ("
              << (algorithm == 5 ? "rectangular PCMM, batch axis spent on the "
                                   "contraction"
                                 : "batch PCMM, batch axis spent on k/2 inputs")
              << ")" << std::endl;
    std::cout << "[nobatch] ring           : logN " << log_n << " (N = " << n
              << "), d = " << d << ", k = " << k << ", batch k/2 = " << batch
              << std::endl;
    std::cout << "[nobatch] chain          : " << limbs << " limbs of "
              << prime_bits << " bits, " << special << " special primes"
              << std::endl;
    std::cout << "[nobatch] width          : d_model " << shape.channels
              << ", hidden " << shape.hidden << ", " << shape.heads << "/"
              << shape.kv_heads << " heads of " << shape.head_dim << std::endl;
    std::cout << "[nobatch] inputs/pass    : " << (algorithm == 5 ? 1 : batch)
              << std::endl;
    std::cout << "[nobatch] rotation keys  : " << rot_count << " indices, "
              << (rot_count * key_bytes / kGiB) << " GiB" << std::endl;
    std::cout << "[nobatch] channels/ct    : " << (algorithm == 5 ? half : 1)
              << ", so d_model is "
              << (algorithm == 5 ? groups(shape.channels) * d : shape.channels)
              << " ciphertexts of " << (CiphertextBytes(n, limbs) / 1048576.0)
              << " MiB" << std::endl;
    if (algorithm == 5)
        std::cout << "[nobatch] Alg-5 calls    : " << rect_calls
                  << " per block, each " << half << " key switches = "
                  << (rect_calls * half) << " per block" << std::endl;
    else
        std::cout << "[nobatch] Alg-1 columns  : " << alg1_columns
                  << " per block, 0 key switches" << std::endl;

    if (dry_run)
    {
        std::cout << "[nobatch] dry run, nothing allocated" << std::endl;
        return 0;
    }

    // ---------------------------------------------------------------------
    // Context
    // ---------------------------------------------------------------------
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
    heongpu::HEArithmeticOperator<S> ops(context, encoder);

    heongpu::BatchMatrixLayout layout(n, d);
    heongpu::HEBatchMatrixOperator<S> matrix(context, layout);
    heongpu::BatchMatrixEncoder bm(k);

    std::vector<int> rot =
        (algorithm == 5) ? heongpu::get_rectangular_rotation_indices(layout)
                         : heongpu::get_batch_cmt_rotation_indices(layout);
    heongpu::Galoiskey<S> galois(context, rot);
    keygen.generate_galois_key(galois, secret);
    ReportMemory("keys");

    const double scale = std::pow(2.0, prime_bits);
    std::mt19937_64 rng(20260805u);

    // ---------------------------------------------------------------------
    // One activation group, and one weight per projection
    // ---------------------------------------------------------------------
    auto encrypt_group = [&](const std::vector<int64_t>& coeffs, int cols)
    {
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(coeffs, layout, d, cols,
                                                      columns);
        std::vector<heongpu::Ciphertext<S>> cts;
        cts.reserve(cols);
        for (int j = 0; j < cols; ++j)
        {
            heongpu::Plaintext<S> p(context);
            matrix.load_coefficients(p, columns[j], scale);
            heongpu::Ciphertext<S> c(context);
            encryptor.encrypt(c, p);
            cts.push_back(std::move(c));
        }
        return cts;
    };

    using Axis = heongpu::HEBatchMatrixOperator<S>::BlockAxis;

    // The operand pool, built once and outside the measured region.
    //
    // Both algorithms contract over the input channels, so both need an
    // operand as wide as the widest projection's in_channels -- and that is
    // exactly where they differ. Algorithm 5 spends d ciphertexts per group of
    // N/2 channels; Algorithm 1 spends one per channel, k/2 times as many. The
    // pool is sized from the widest projection so that no projection is
    // measured against an operand narrower than its own contraction, which
    // would understate its GEMM by the ratio.
    int widest_in = 0;
    for (const Projection& p : projs)
        widest_in = std::max(widest_in, p.in_channels);

    const int pool_groups = groups(widest_in);
    std::vector<std::vector<heongpu::Ciphertext<S>>> stream;
    if (algorithm == 5)
    {
        for (int g = 0; g < pool_groups; ++g)
            stream.push_back(encrypt_group(
                stream_coefficients(d, half, k, scale, rng), d));
    }
    else
    {
        // Algorithm 1's operand is one matrix encryption per d channels; the
        // pool is the whole width, in d-sized pieces, and a projection takes
        // the leading in_channels/d of them.
        for (int base = 0; base < widest_in; base += d)
            stream.push_back(
                encrypt_group(stream_coefficients(d, half, k, scale, rng), d));
    }
    std::cout << "[nobatch] operand pool   : " << widest_in
              << " channels in "
              << (algorithm == 5 ? pool_groups * d : widest_in)
              << " ciphertexts" << std::endl;
    ReportMemory("stream");

    // Warm up: the first call of each kind builds twiddle tables and pays a
    // one-off allocation, and neither belongs in the measured region.
    {
        std::vector<heongpu::Ciphertext<S>*> in;
        for (auto& c : stream.front())
            in.push_back(&c);
        std::vector<heongpu::Ciphertext<S>> out;
        if (algorithm == 5)
        {
            matrix.encode_plaintext_matrix(
                weight_coefficients(d, half, k, scale, rng), d, half, 0, scale);
            matrix.rectangular_pcmm(out, in, galois, ops, Axis::coefficient,
                                    /*rescale=*/false);
        }
        else
        {
            std::vector<std::vector<std::complex<double>>> w(
                batch, std::vector<std::complex<double>>(
                           static_cast<std::size_t>(d) * d));
            std::uniform_real_distribution<double> pick(-1.0, 1.0);
            for (auto& z : w[0])
                z = std::complex<double>(pick(rng), 0.0);
            for (int s = 1; s < batch; ++s)
                w[s] = w[0];
            std::vector<int64_t> c;
            bm.encode(w, d, d, scale, c);
            matrix.encode_plaintext_matrix(c, d, d, 0, scale);
            matrix.pcmm(out, in, /*rescale=*/false);
        }
    }
    cudaDeviceSynchronize();
    ReportMemory("warm-up");

    // ---------------------------------------------------------------------
    // The measured region
    // ---------------------------------------------------------------------
    std::cout << "[nobatch] --- measured region ---" << std::endl;
    cudaProfilerStart();
    RegionTimer whole;

    std::vector<double> per_projection;
    per_projection.reserve(projs.size());

    for (const Projection& p : projs)
    {
        Range _r(p.name);
        RegionTimer t;

        if (algorithm == 5)
        {
            const int gin = groups(p.in_channels);
            const int gout = groups(p.out_channels);
            for (int gi = 0; gi < gin; ++gi)
            {
                std::vector<heongpu::Ciphertext<S>*> in;
                for (auto& c : stream[gi])
                    in.push_back(&c);
                for (int go = 0; go < gout; ++go)
                {
                    matrix.encode_plaintext_matrix(
                        weight_coefficients(d, half, k, scale, rng), d, half, 0,
                        scale);
                    std::vector<heongpu::Ciphertext<S>> out;
                    matrix.rectangular_pcmm(out, in, galois, ops,
                                            Axis::coefficient,
                                            /*rescale=*/false);
                }
            }
        }
        else
        {
            // Algorithm 1 streams the output axis in column blocks of d, which
            // is what project() does and what keeps the twiddled plaintext to
            // a sane size. The contraction is over in_channels, so the operand
            // is the leading in_channels ciphertexts of the pool and the loop
            // is over the output.
            std::vector<heongpu::Ciphertext<S>*> in;
            for (int base = 0; base < p.in_channels; base += d)
                for (auto& c : stream[base / d])
                    in.push_back(&c);
            const int inner = static_cast<int>(in.size());
            std::uniform_real_distribution<double> pick(-1.0, 1.0);
            for (int base = 0; base < p.out_channels; base += d)
            {
                const int cols = std::min(d, p.out_channels - base);
                std::vector<std::vector<std::complex<double>>> w(
                    batch, std::vector<std::complex<double>>(
                               static_cast<std::size_t>(inner) * cols));
                for (auto& z : w[0])
                    z = std::complex<double>(pick(rng), 0.0);
                for (int s = 1; s < batch; ++s)
                    w[s] = w[0];
                std::vector<int64_t> c;
                bm.encode(w, inner, cols, scale, c);
                matrix.encode_plaintext_matrix(c, inner, cols, 0, scale);
                std::vector<heongpu::Ciphertext<S>> out;
                matrix.pcmm(out, in, /*rescale=*/false);
            }
        }

        cudaDeviceSynchronize();
        per_projection.push_back(t.ms());
    }

    const double total = whole.ms();
    cudaProfilerStop();

    // ---------------------------------------------------------------------
    // Report
    // ---------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(1);
    for (std::size_t i = 0; i < projs.size(); ++i)
        std::cout << "[nobatch] " << projs[i].name << " ("
                  << projs[i].in_channels << " -> " << projs[i].out_channels
                  << "): " << per_projection[i] << " ms" << std::endl;

    const int inputs = (algorithm == 5) ? 1 : batch;
    std::cout << "[nobatch] block projections total: " << total << " ms for "
              << inputs << " input(s)" << std::endl;
    std::cout << "[nobatch] per input, per block   : " << (total / inputs)
              << " ms" << std::endl;
    std::cout << "[nobatch] per input, " << shape.layers
              << " blocks     : " << (total * shape.layers / inputs / 1000.0)
              << " s" << std::endl;
    std::cout << "[nobatch] throughput             : "
              << (1000.0 * inputs / (total * shape.layers))
              << " inputs/s at " << shape.layers << " blocks" << std::endl;
    ReportMemory("run");
    return 0;
}

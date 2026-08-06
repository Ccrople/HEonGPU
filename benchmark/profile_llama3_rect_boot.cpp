// Nsight Systems capture target for one REFRESHED Llama-3 block on Kang's
// Algorithm 5 (every plaintext-ciphertext product) and Algorithm 4 (every
// ciphertext-ciphertext product).
//
//   llama3_rect_boot_profile
//
// WHY THIS EXISTS ALONGSIDE llama3_rect_profile
// ---------------------------------------------
// The unrefreshed target measured a whole block and reported, honestly, that it
// does not fit: 86 levels at full strength against 71 available, made to run
// only by cutting every non-linearity down to a degree nobody would deploy. That
// is a timing measurement and not a circuit.
//
// This target adds the refresh. CKKS bootstrapping is an identity on the
// plaintext POLYNOMIAL -- ModRaise, CoeffToSlot, EvalMod, SlotToCoeff put back
// what they took -- so it refreshes a RECT column and a BATCH matrix encryption
// where they stand, with no encoding crossing at all. That is worth more here
// than anywhere else: a crossing is 86% of an unrefreshed block on this path, so
// a refresh that needed one would cost more than the arithmetic it protects.
//
// WHAT IT MEASURES
// ----------------
//   probe     one refresh in each of the three encodings, decrypted and
//             checked. This is the claim the rest of the target rests on, so
//             it is asserted rather than argued.
//   refresh   one refresh alone, for its cost
//   block     one pre-norm transformer block with the refreshes in place,
//             printing the LEVEL SCHEDULE as it goes
//
// THE SCHEDULE, WHICH IS THE POINT
// --------------------------------
// Six seams, chosen so no stretch spends more than one bootstrap hands back:
//
//   entry .................. the residual stream, between blocks
//   after_attention_norm ... RMSNorm is the deepest stretch and leaves nothing
//                            for the projections behind it
//   post_qk ................ the scores, before the SoftMax
//   post_softmax ........... P and V together, before the value product
//   mid .................... the residual stream, between the two halves
//   after_feed_forward_norm  as above, before the gate and up projections
//
// HEONGPU_RECT_REFRESH_* switches each one off, which is how a schedule is
// shown to be needed rather than assumed.
//
// THE CHAIN
// ---------
// Bootstrapping wants q0 / scale near 2^10, so the chain is a 60-bit bottom
// under 50-bit working primes at a scale of 2^50, as in HEonGPU's own example.
// The special primes are the whole chain: Algorithm 5's CMT needs N/2 - 1 = 2047
// Galois keys at the minimum ring, and dnum = 1 is the only decomposition whose
// key set fits on one card. It also makes the mod-down O(L^2), which is why the
// chain here is as short as the schedule allows.
//
// THE REAL LLAMA-3 8B SHAPE
// -------------------------
// The defaults are a half-width block that runs in ten minutes. The model's own
// numbers are one command, and they need no rounding: 8B's head dim is exactly
// 128, which is the one dimension this encoding fixes.
//
//   HEONGPU_BOOT_D=128 HEONGPU_BOOT_CHANNEL_GROUPS=2
//   HEONGPU_BOOT_HIDDEN_GROUPS=7 HEONGPU_BOOT_KV_HEADS=8
//   HEONGPU_BOOT_HIDDEN_BLOCK_GROUPS=1 HEONGPU_BOOT_LIMBS=38
//
// -> d_model 4096, hidden 14336, 32 heads of 128 over 8 kv heads, 128 tokens.
// Measured on one A6000: worst stretch 12 -- the same as at half the width,
// since levels belong to the circuit and not to the model -- 14 refreshes and
// 3326.8 s. HIDDEN_BLOCK_GROUPS = 1 is what holds the SwiGLU inside the card:
// it forms, activates, refreshes and projects down one group of 2048 hidden
// channels at a time, so the 14336 never exists at once.
//
// As in the sibling targets the plaintext values are random and the fitted
// intervals are nominal: nothing about the cost of a CKKS circuit depends on the
// numbers in it. The probe stage is the exception and says so.

#include <heongpu/heongpu.hpp>

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
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

bool EnvFlag(const char* name, bool fallback)
{
    return EnvInt(name, fallback ? 1 : 0) != 0;
}

struct Range
{
    explicit Range(const std::string& name) { nvtxRangePushA(name.c_str()); }
    ~Range() { nvtxRangePop(); }
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
};

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
    std::cout << "[boot] memory after " << stage << ": "
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

double max_abs_error(const std::vector<double>& a, const std::vector<double>& b)
{
    double worst = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; i++)
        worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

/// The level schedule: every seam of the block, what it left behind, and what
/// the stretch before it spent. This is the deliverable of a refreshed circuit
/// and it appears in no Nsight report.
///
/// A refresh is recognised by its name and not by the sign of the depth change.
/// The two disagree whenever the chain is longer than the schedule needs: a
/// bootstrap always hands back the SAME depth, so on a chain with plenty left
/// it moves the ciphertext DOWN. That is not an error, it is the statement that
/// the chain is longer than one bootstrap's worth -- and the number this class
/// exists to produce, the worst stretch, is exactly what a chain should be cut
/// back to.
class Schedule
{
  public:
    Schedule(int limbs) : limbs_(limbs) {}

    void operator()(const char* name, int depth)
    {
        const bool refresh = std::strstr(name, "refresh") != nullptr;
        std::cout << "[boot] " << std::setw(34) << std::left << name
                  << std::right << " depth " << std::setw(3) << depth
                  << "   limbs left " << std::setw(3) << (limbs_ - depth);
        if (refresh)
        {
            std::cout << "   REFRESH";
            refreshes_++;
        }
        else if (last_ >= 0)
        {
            const int spent = depth - last_;
            std::cout << "   spent " << std::setw(3) << spent;
            if (spent > worst_)
            {
                worst_ = spent;
                worst_name_ = name;
            }
        }
        std::cout << std::endl;
        last_ = depth;
    }

    int worst_stretch() const { return worst_; }
    const std::string& worst_name() const { return worst_name_; }
    int refreshes() const { return refreshes_; }

  private:
    int limbs_;
    int last_ = -1;
    int worst_ = 0;
    std::string worst_name_;
    int refreshes_ = 0;
};

} // namespace

int main(int argc, char* argv[])
{
    (void) argc;
    (void) argv;

    const std::string stage = EnvStr("HEONGPU_BOOT_STAGE", "probe");
    const int log_n = EnvInt("HEONGPU_BOOT_LOGN", 12);
    const int d = EnvInt("HEONGPU_BOOT_D", 64);
    // 60-bit bottom under 50-bit working primes at a scale of 2^50: the ratio
    // q0 / scale near 2^10 that HEonGPU's EvalMod is fitted around. Off it, the
    // bootstrap returns noise and says nothing.
    const int limbs = EnvInt("HEONGPU_BOOT_LIMBS", 36);
    const int prime_bits = EnvInt("HEONGPU_BOOT_PRIME_BITS", 50);
    const int log_scale = EnvInt("HEONGPU_BOOT_LOG_SCALE", prime_bits);
    // dnum = 1. Algorithm 5's 2047 keys fit at no other decomposition.
    const int special = EnvInt("HEONGPU_BOOT_SPECIAL_PRIMES", limbs);
    const int hamming = EnvInt("HEONGPU_BOOT_HAMMING", 16);
    const int ctos = EnvInt("HEONGPU_BOOT_CTOS", 3);
    const int stoc = EnvInt("HEONGPU_BOOT_STOC", 3);
    const int taylor = EnvInt("HEONGPU_BOOT_TAYLOR", 11);
    const bool less_key = EnvFlag("HEONGPU_BOOT_LESS_KEY", true);
    const bool dry_run = EnvFlag("HEONGPU_BOOT_DRY_RUN", false);

    const int n = 1 << log_n;
    const int k = n / d;
    const int step = k / 2;
    const int half = n / 2;

    const int channel_groups = EnvInt("HEONGPU_BOOT_CHANNEL_GROUPS", 1);
    const int hidden_groups = EnvInt("HEONGPU_BOOT_HIDDEN_GROUPS", 1);
    const int channels = channel_groups * half;
    const int hidden = hidden_groups * half;
    const int heads = channel_groups * step;
    const int kv_heads = std::max(1, EnvInt("HEONGPU_BOOT_KV_HEADS", heads));

    const double key_bytes = SwitchingKeyBytes(n, limbs, special);
    const int rot_count = half - 1;

    std::cout << "[boot] stage           : " << stage << std::endl;
    std::cout << "[boot] ring            : logN " << log_n << " (N = " << n
              << "), d = " << d << ", k = " << k << ", blocks k/2 = " << step
              << std::endl;
    std::cout << "[boot] chain           : " << limbs << " limbs (60 + "
              << (limbs - 1) << " x " << prime_bits << "), " << special
              << " special primes of 60, scale 2^" << log_scale << std::endl;
    std::cout << "[boot] bootstrapping   : CtoS " << ctos << ", StoC " << stoc
              << ", taylor " << taylor << ", less key mode "
              << (less_key ? "on" : "off") << ", h = " << hamming << std::endl;
    std::cout << "[boot] width           : d_model " << channels << ", hidden "
              << hidden << ", " << heads << " heads of " << d << " over "
              << kv_heads << " kv heads" << std::endl;
    std::cout << "[boot] rotation keys   : " << rot_count << " indices, "
              << (rot_count * key_bytes / kGiB) << " GiB" << std::endl;

    if (dry_run)
    {
        std::cout << "[boot] dry run, nothing allocated" << std::endl;
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
    // A sparse secret, as bootstrapping wants: EvalMod's range is set by the
    // Hamming weight, and the dense default puts the coefficients outside it.
    heongpu::Secretkey<S> secret(context, hamming);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder(context);

    heongpu::BatchMatrixLayout layout(n, d);
    const double scale = std::pow(2.0, log_scale);
    heongpu::llama::Llama3BatchOperator batch(context, encoder, layout, scale);
    Rect op(context, encoder, layout, scale);

    // The bootstrapping context is per operator, so it is generated on the very
    // arithmetic half that will run the refresh and not on a second one.
    heongpu::BootstrappingConfig boot_config(ctos, stoc, taylor, less_key);
    op.arith().generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);
    ReportMemory("bootstrapping parameters");

    // One key for both jobs. A shift-vector Galois key asked for an index it
    // does not hold is undefined behaviour rather than an error, so the union
    // is taken literally, by index and not by the element it maps to.
    std::vector<int> boot_shifts = op.arith().bootstrapping_key_indexs();
    std::vector<int> shifts = op.rotation_indices();
    {
        std::set<int> all(shifts.begin(), shifts.end());
        const std::size_t before = all.size();
        all.insert(boot_shifts.begin(), boot_shifts.end());
        shifts.assign(all.begin(), all.end());
        std::cout << "[boot] galois key      : " << before << " for Algorithm 5 + "
                  << boot_shifts.size() << " for bootstrapping = "
                  << shifts.size() << " indices, "
                  << (shifts.size() * key_bytes / kGiB) << " GiB" << std::endl;
    }

    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);
    ReportMemory("keys");

    std::mt19937_64 rng(20260806u);
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));

    // -----------------------------------------------------------------------
    // probe: the claim the rest of this target rests on
    // -----------------------------------------------------------------------
    if (stage == "probe")
    {
        // Small values, as an activation carries: the coefficient bound is the
        // one condition a refresh imposes, and a rect coefficient IS one entry
        // times the scale.
        const std::vector<double> values =
            random_matrix(static_cast<std::size_t>(d) * channels, 1.0, rng);

        std::cout << "[boot] --- measured region ---" << std::endl;
        cudaProfilerStart();

        {
            Range _r("probe.rect");
            heongpu::llama::RectActivation x =
                op.encrypt(values, channels, encryptor, scale);
            const int before = x.column.front().depth();
            RegionTimer t;
            op.bootstrap(x, "probe.rect.refresh", galois, relin);
            cudaDeviceSynchronize();
            const double ms = t.ms();
            const std::vector<double> back =
                op.decrypt(x, decryptor, x.column.front().scale());
            std::cout << "[boot] rect  : depth " << before << " -> "
                      << x.column.front().depth() << ", " << std::fixed
                      << std::setprecision(1) << ms << " ms for "
                      << x.column.size() << " ciphertexts, max abs error "
                      << std::scientific << std::setprecision(3)
                      << max_abs_error(values, back) << std::endl;
        }

        {
            // The same claim for the encoding Algorithm 4 consumes. A matrix
            // encryption is a different reading of the same polynomial, so if
            // the refresh is a polynomial identity this has to hold too.
            Range _r("probe.batch");
            heongpu::llama::RectActivation x =
                op.encrypt(values, channels, encryptor, scale);
            heongpu::llama::BatchActivation b = op.to_batch(x, 0, galois);
            std::vector<heongpu::llama::BatchActivation> one;
            one.push_back(std::move(b));
            op.bootstrap(one.front(), "probe.batch.refresh", galois, relin);
            heongpu::llama::RectActivation back_rect =
                op.from_batch(one, half, galois);
            const std::vector<double> back = op.decrypt(
                back_rect, decryptor, back_rect.column.front().scale());
            std::vector<double> expect(static_cast<std::size_t>(d) * half);
            for (int i = 0; i < d; i++)
                for (int c = 0; c < half; c++)
                    expect[static_cast<std::size_t>(i) * half + c] =
                        values[static_cast<std::size_t>(i) * channels + c];
            std::cout << "[boot] batch : refreshed in matrix form, "
                      << "max abs error " << std::scientific
                      << std::setprecision(3) << max_abs_error(expect, back)
                      << "  (a crossing alone costs about 1e-3)" << std::endl;
        }

        cudaProfilerStop();
        std::cout << std::defaultfloat;
        ReportMemory("probe");
        return 0;
    }

    // -----------------------------------------------------------------------
    // The activation every other stage runs on
    // -----------------------------------------------------------------------
    heongpu::llama::RectActivation x = op.encrypt(
        random_matrix(static_cast<std::size_t>(d) * channels, 1.0, rng),
        channels, encryptor, scale);
    ReportMemory("activation");

    Schedule schedule(limbs);
    op.depth_trace = [&schedule](const char* name, int depth)
    { schedule(name, depth); };

    std::cout << "[boot] --- measured region ---" << std::endl;
    cudaProfilerStart();
    RegionTimer whole;

    if (stage == "refresh")
    {
        // One refresh of one group, for its own cost. Everything below is
        // this, times the number of seams and the ciphertexts at each.
        Range _r("stage.refresh");
        op.bootstrap(x, "refresh", galois, relin);
        cudaDeviceSynchronize();
        std::cout << "[boot] refreshed " << x.column.size()
                  << " ciphertexts to depth " << x.column.front().depth()
                  << " (" << (limbs - x.column.front().depth())
                  << " limbs left)" << std::endl;
    }
    else if (stage == "block")
    {
        const std::vector<double> norm_w =
            random_matrix(static_cast<std::size_t>(channels), 0.5, rng);
        const int kv_channels = kv_heads * d;

        Rect::RectTransformerBlockWeights w;
        w.attention_norm = norm_w;
        w.feed_forward_norm = norm_w;
        w.attention.query = random_matrix(
            static_cast<std::size_t>(channels) * channels, amp, rng);
        w.attention.key = random_matrix(
            static_cast<std::size_t>(channels) * kv_channels, amp, rng);
        w.attention.value = w.attention.key;
        w.attention.output = w.attention.query;
        w.feed_forward.gate = random_matrix(
            static_cast<std::size_t>(channels) * hidden, amp, rng);
        w.feed_forward.up = w.feed_forward.gate;
        w.feed_forward.down = random_matrix(
            static_cast<std::size_t>(hidden) * channels, amp, rng);

        Rect::RectTransformerBlockConfig c;
        c.attention_norm.eps = 1e-5;
        c.attention_norm.sum_lo = channels * 0.20;
        c.attention_norm.sum_hi = channels * 0.50;
        // Seven, not fifteen. Over a range that spans a factor of 2.5 the
        // degree-7 fit is already accurate to 3.2e-06 and the degree-15 one to
        // 1.5e-11, and nothing downstream can tell the difference: Section
        // 3.1.2 puts the requirement at 12 bits. The two differ by a level.
        c.attention_norm.degree = EnvInt("HEONGPU_BOOT_NORM_DEGREE", 7);
        c.attention_norm.newton_iterations =
            EnvInt("HEONGPU_BOOT_NORM_NEWTON", 0);
        c.feed_forward_norm = c.attention_norm;

        c.attention.in_channels = channels;
        c.attention.heads = heads;
        c.attention.kv_heads = kv_heads;
        c.attention.causal = EnvFlag("HEONGPU_BOOT_CAUSAL", true);
        c.attention.softmax.bound = EnvDouble("HEONGPU_BOOT_SOFTMAX_BOUND", 8.0);
        c.attention.softmax.iterations =
            EnvInt("HEONGPU_BOOT_SOFTMAX_ITERS", 1);
        c.attention.softmax.exp_degree = EnvInt("HEONGPU_BOOT_EXP_DEGREE", 15);
        c.attention.softmax.inverse_degree =
            EnvInt("HEONGPU_BOOT_INVERSE_DEGREE", 15);
        c.attention.softmax.inverse_newton =
            EnvInt("HEONGPU_BOOT_INVERSE_NEWTON", 0);

        // Section 4.3, the calibrated range of the SoftMax denominator.
        //
        // The worst case is that every score sits at the bottom of [-bound, 0]
        // or every one at the top, which puts the sum of exponentials across a
        // factor of exp(2 bound / 2^iterations) -- 2981 here. No reciprocal is
        // fitted across that: it takes degree 255 and eight levels to reach 12
        // bits, and the degree 7 this target used to carry was wrong by 98.6%
        // over its own stated range. The paper's answer is not a higher degree
        // but a narrower interval, measured rather than bounded.
        //
        // SPREAD is that measurement, in logs: the denominator is taken to lie
        // in [d exp(-spread), d]. Two costs degree 15 and four levels. What
        // earns a number well below the worst case is Section 3.1.1's sink
        // prefix -- public tokens, present in every row, attended by every
        // query -- which is also what stops a causal row from attending to one
        // position and driving the ratio to d on its own.
        const double spread = EnvDouble("HEONGPU_BOOT_SOFTMAX_SPREAD", 2.0);
        const double axis = static_cast<double>(d);
        c.attention.softmax.sum_lo = axis * std::exp(-spread);
        c.attention.softmax.sum_hi = axis;
        // Read only when iterations is above one: after a round the
        // coordinates sum to one, so the denominator is between 1/d and d/d,
        // and this is how far up calibration says it reaches.
        c.attention.softmax.concentration =
            EnvDouble("HEONGPU_BOOT_SOFTMAX_CONCENTRATION", 8.0);

        c.feed_forward.in_channels = channels;
        c.feed_forward.hidden_channels = hidden;
        c.feed_forward.silu_degree = EnvInt("HEONGPU_BOOT_SILU_DEGREE", 31);
        c.feed_forward.hidden_block_groups =
            EnvInt("HEONGPU_BOOT_HIDDEN_BLOCK_GROUPS", 1);

        c.fold_norm_scale = EnvFlag("HEONGPU_BOOT_FOLD_NORM", true);
        c.refresh.entry = EnvFlag("HEONGPU_BOOT_REFRESH_ENTRY", false);
        c.refresh.after_attention_norm =
            EnvFlag("HEONGPU_BOOT_REFRESH_ATTN_NORM", true);
        c.refresh.post_qk = EnvFlag("HEONGPU_BOOT_REFRESH_POST_QK", true);
        c.refresh.post_softmax =
            EnvFlag("HEONGPU_BOOT_REFRESH_POST_SOFTMAX", true);
        c.refresh.mid = EnvFlag("HEONGPU_BOOT_REFRESH_MID", true);
        c.refresh.after_feed_forward_norm =
            EnvFlag("HEONGPU_BOOT_REFRESH_FFN_NORM", true);
        c.refresh.feed_forward_hidden =
            EnvFlag("HEONGPU_BOOT_REFRESH_FFN_HIDDEN", true);

        // What every fit in the block actually achieves over the range it is
        // configured for, against the 12 bits of Section 3.1.2. A degree is
        // not a free parameter -- it is what the range and the precision
        // together leave -- and this is the line that says so. Sampled on the
        // host, so it costs nothing and cannot be argued with.
        {
            const double target = std::pow(2.0, -12);
            const double norm_lo = c.attention_norm.sum_lo;
            const double norm_hi = c.attention_norm.sum_hi;
            const double eps = c.attention_norm.eps;
            const double cw = static_cast<double>(channels);
            const auto report = [&](const char* what,
                                    const std::function<double(double)>& f,
                                    double lo, double hi, int degree,
                                    bool relative)
            {
                const double e = heongpu::llama::chebyshev_max_error(
                    f, lo, hi, degree, relative);
                const int want = heongpu::llama::chebyshev_degree_for(
                    f, lo, hi, target, relative);
                std::cout << "[boot] fit " << what << ": degree " << degree
                          << " over [" << lo << ", " << hi << "] -> "
                          << (relative ? "rel" : "abs") << " err " << e
                          << ", 2^-12 wants ";
                if (want == 0)
                {
                    std::cout << "more than 511 -- the interval is too wide";
                }
                else
                {
                    std::cout << want << " ("
                              << heongpu::llama::chebyshev_levels(want)
                              << " levels)";
                }
                std::cout << (e <= target ? "" : "   <-- MISSES") << std::endl;
            };

            report("rms_norm 1/sqrt",
                   [cw, eps](double s) {
                       return 1.0 / std::sqrt(s / cw + eps);
                   },
                   norm_lo, norm_hi, c.attention_norm.degree, true);
            report("softmax exp",
                   [&c](double x) {
                       return std::exp(
                           x / std::pow(2.0, c.attention.softmax.iterations));
                   },
                   -c.attention.softmax.bound, 0.0,
                   c.attention.softmax.exp_degree, true);
            report("softmax 1/x", [](double x) { return 1.0 / x; },
                   c.attention.softmax.sum_lo, c.attention.softmax.sum_hi,
                   c.attention.softmax.inverse_degree, true);
            report("swiglu silu",
                   [](double x) { return x / (1.0 + std::exp(-x)); },
                   -c.feed_forward.silu_bound, c.feed_forward.silu_bound,
                   c.feed_forward.silu_degree, false);
        }

        const bool refreshed = EnvFlag("HEONGPU_BOOT_REFRESHED", true);
        std::cout << "[boot] refreshes       : " << c.refresh.count()
                  << " seams, norm scale "
                  << (c.fold_norm_scale ? "folded into the projections"
                                        : "applied homomorphically")
                  << std::endl;

        heongpu::llama::RectActivation out = op.transformer_block(
            x, w, c, galois, relin, refreshed ? &galois : nullptr);
        cudaDeviceSynchronize();

        std::cout << "[boot] worst stretch   : " << schedule.worst_stretch()
                  << " levels, at " << schedule.worst_name()
                  << " -- a chain of " << (schedule.worst_stretch() + 26)
                  << " limbs is what this schedule wants, since a refresh "
                     "hands back the chain less 25"
                  << std::endl;
        std::cout << "[boot] refreshes taken : " << schedule.refreshes()
                  << std::endl;
        std::cout << "[boot] block left      : depth "
                  << out.column.front().depth() << ", "
                  << (limbs - out.column.front().depth()) << " limbs"
                  << std::endl;
    }
    else
    {
        std::cerr << "[boot] unknown stage '" << stage << "'" << std::endl;
        return 2;
    }

    const double total = whole.ms();
    cudaProfilerStop();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[boot] " << stage << ": " << total << " ms" << std::endl;
    ReportMemory("run");
    return 0;
}

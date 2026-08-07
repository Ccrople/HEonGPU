// The level ledger of the rectangular path: what every primitive of one
// Llama-3 block actually spends, measured one primitive at a time.
//
//   llama3_levels_profile
//
// WHY THIS EXISTS
// ---------------
// llama3_rect_boot_profile prints the schedule of a whole block -- the depth at
// each named seam -- which is the number that sizes the chain. It does not say
// where inside a stretch the levels went, and a design that wants to move an
// operator, fuse a conversion or shorten a chain needs the per-operator number
// and not the per-stretch one. Every depth in a design document should be a
// reading off this target rather than an estimate, because two of them are not
// what the arithmetic suggests:
//
//   - the batch CMT (Algorithm 3, the K transpose) is FREE. It is monomial
//     multiplications, a Cooley-Tukey tweak, an exact modular scalar and
//     automorphisms; not one of those rescales.
//   - an auxiliary-track fit costs the WIDE track a number that depends on
//     where the wide track was standing, because the refreshed narrow
//     ciphertext comes back at a fixed depth and the product takes the deeper
//     of the two. It is not a constant, and this target measures it at both
//     ends rather than quoting one.
//
// WHAT IT MEASURES
// ----------------
// Each row is one primitive run on its own inputs, printed as
// depth-in -> depth-out. Nothing here is a whole layer, so nothing here has a
// schedule; the point is the delta.
//
//   conversions   RECT <-> SLOT, RECT <-> BATCH, BATCH <-> SLOT, block_map
//   products      Algorithm 5 (rect PCMM), Algorithm 4 (batch CCMM), the CMT
//   non-linear    RMSNorm, SoftMax, SiLU, the gate product, RoPE, the residual
//
// The non-linear rows are run in SLOT form, which is where they live: the
// crossings that carry them there are separate rows above, so a design can add
// the two up in whatever arrangement it is considering rather than inheriting
// this one.
//
// Levels do not depend on the width of the model -- a projection is one
// Algorithm 5 call per (input group, output group) pair and every call spends
// the same one level -- so the defaults are the smallest shape that exercises
// every path. HEONGPU_LEVELS_D and HEONGPU_LEVELS_CHANNEL_GROUPS say otherwise
// if that is ever in doubt.
//
// THE AUXILIARY TRACK
// -------------------
// HEONGPU_LEVELS_AUX=1 adds the narrow-track rows, which need bootstrapping
// parameters and so a chain of the shape the refresh wants (a 60-bit bottom
// under 50-bit primes at a scale of 2^50) rather than the short one the rest of
// the target is happy with. That is a slower run and a much larger key set, so
// it is off by default.

#include <heongpu/heongpu.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace
{

constexpr auto S = heongpu::Scheme::CKKS;
using Rect = heongpu::llama::Llama3RectOperator;

int EnvInt(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return fallback;
    return std::atoi(v);
}

bool EnvFlag(const char* name, bool fallback)
{
    return EnvInt(name, fallback ? 1 : 0) != 0;
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

/// One measured row. The name is what the design document calls the operator,
/// so the two can be read against each other without a translation table.
void Row(const char* group, const char* what, int before, int after,
         const char* note = "")
{
    std::cout << "[lvl] " << std::setw(9) << std::left << group << std::setw(34)
              << what << std::right << " depth " << std::setw(3) << before
              << " -> " << std::setw(3) << after << "   spends "
              << std::setw(2) << (after - before);
    if (note[0] != '\0')
        std::cout << "   " << note;
    std::cout << std::endl;
}

} // namespace

int main()
{
    const int log_n = EnvInt("HEONGPU_LEVELS_LOGN", 12);
    const int d = EnvInt("HEONGPU_LEVELS_D", 128);
    const int channel_groups = EnvInt("HEONGPU_LEVELS_CHANNEL_GROUPS", 1);
    const bool aux = EnvFlag("HEONGPU_LEVELS_AUX", false);

    // The fitted degrees the block runs at, so the level a fit spends here is
    // the level it spends there. Overridable because a degree is the one knob
    // in this whole table that a caller chooses.
    const int norm_degree = EnvInt("HEONGPU_LEVELS_NORM_DEGREE", 7);
    const int exp_degree = EnvInt("HEONGPU_LEVELS_EXP_DEGREE", 15);
    const int inverse_degree = EnvInt("HEONGPU_LEVELS_INVERSE_DEGREE", 15);
    const int silu_degree = EnvInt("HEONGPU_LEVELS_SILU_DEGREE", 31);

    const int n = 1 << log_n;
    const int k = n / d;
    const int step = k / 2;
    const int half = n / 2;
    const int channels = channel_groups * half;
    const int heads = channel_groups * step;

    // Without the refresh a short chain is enough and the keys are small; with
    // it the chain has to be the one EvalMod is fitted around.
    const int limbs = EnvInt("HEONGPU_LEVELS_LIMBS", aux ? 36 : 20);
    const int prime_bits = aux ? 50 : 40;
    const int log_scale = prime_bits;
    const int special = EnvInt("HEONGPU_LEVELS_SPECIAL_PRIMES", limbs);

    std::cout << "[lvl] ring    : logN " << log_n << " (N = " << n
              << "), d = " << d << ", k = " << k << ", blocks k/2 = " << step
              << std::endl;
    std::cout << "[lvl] width   : d_model " << channels << ", " << heads
              << " heads of " << d << std::endl;
    std::cout << "[lvl] chain   : " << limbs << " limbs, scale 2^" << log_scale
              << ", auxiliary track " << (aux ? "on" : "off") << std::endl;
    std::cout << "[lvl] degrees : norm " << norm_degree << ", exp "
              << exp_degree << ", 1/x " << inverse_degree << ", silu "
              << silu_degree << std::endl;
    std::cout << "[lvl] cheby   : levels(" << norm_degree << ") = "
              << heongpu::llama::chebyshev_levels(norm_degree) << ", ("
              << exp_degree << ") = "
              << heongpu::llama::chebyshev_levels(exp_degree) << ", ("
              << inverse_degree << ") = "
              << heongpu::llama::chebyshev_levels(inverse_degree) << ", ("
              << silu_degree << ") = "
              << heongpu::llama::chebyshev_levels(silu_degree) << std::endl;

    std::vector<int> q_bits(limbs, prime_bits);
    if (aux)
        q_bits.front() = 60;
    std::vector<int> p_bits(special, 60);

    heongpu::HEContext<S> context =
        heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(static_cast<std::size_t>(n));
    context->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    context->generate();

    heongpu::HEKeyGenerator<S> keygen(context);
    heongpu::Secretkey<S> secret(context, 16);
    keygen.generate_secret_key(secret);
    heongpu::Publickey<S> pub(context);
    keygen.generate_public_key(pub, secret);
    heongpu::HEEncryptor<S> encryptor(context, pub);
    heongpu::HEEncoder<S> encoder(context);

    heongpu::BatchMatrixLayout layout(n, d);
    const double scale = std::pow(2.0, log_scale);
    Rect op(context, encoder, layout, scale);
    // The halves of a crossing -- the row bridge on its own, the CMT, the
    // causal mask -- are reachable through the batch operator the rect one is
    // composed on. A second instance shares the layout and so the tables.
    heongpu::llama::Llama3BatchOperator batch(context, encoder, layout, scale);

    std::vector<int> shifts = op.rotation_indices();
    if (aux)
    {
        heongpu::BootstrappingConfig boot_config(3, 3, 11, true);
        op.arith().generate_bootstrapping_params(
            scale, boot_config,
            heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);
        const std::vector<int> boot_shifts = op.arith().bootstrapping_key_indexs();
        std::set<int> all(shifts.begin(), shifts.end());
        all.insert(boot_shifts.begin(), boot_shifts.end());
        shifts.assign(all.begin(), all.end());
    }
    // RoPE pairs a channel with the one half a head away. On this encoding
    // that separation is a CIPHERTEXT index and not a slot offset, so the
    // rotation the slot-form rope() takes is not one this path needs; it is
    // measured anyway, and the key for it added, so the row is honest about
    // what the implemented primitive costs.
    {
        std::set<int> all(shifts.begin(), shifts.end());
        all.insert(step * (d / 2));
        shifts.assign(all.begin(), all.end());
    }
    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);

    std::mt19937_64 rng(20260807u);
    const double amp = 1.0 / std::sqrt(static_cast<double>(channels));
    const std::vector<double> values =
        random_matrix(static_cast<std::size_t>(d) * channels, 0.5, rng);

    const auto fresh = [&]() {
        return op.encrypt(values, channels, encryptor, scale);
    };

    std::cout << "[lvl] ---------------------------------------------------"
              << std::endl;

    // -----------------------------------------------------------------------
    // The encoding conversions
    // -----------------------------------------------------------------------
    {
        heongpu::llama::RectActivation x = fresh();
        const int before = x.column.front().depth();
        std::vector<heongpu::Ciphertext<S>> slots = op.to_slots(x, galois);
        const int mid = slots.front().depth();
        Row("convert", "RECT -> SLOT  (to_slots)", before, mid,
            "row bridge + block map");

        heongpu::llama::RectActivation back =
            op.from_slots(slots, channels, galois);
        Row("convert", "SLOT -> RECT  (from_slots)", mid,
            back.column.front().depth(), "block map + row bridge");
    }
    {
        heongpu::llama::RectActivation x = fresh();
        const int before = x.column.front().depth();
        heongpu::llama::BatchActivation b = op.to_batch(x, 0, galois);
        const int mid = b.column.front().depth();
        Row("convert", "RECT -> BATCH (to_batch)", before, mid,
            "bridge + block map + bridge");

        std::vector<heongpu::llama::BatchActivation> one;
        one.push_back(std::move(b));
        heongpu::llama::RectActivation back = op.from_batch(one, half, galois);
        Row("convert", "BATCH -> RECT (from_batch)", mid,
            back.column.front().depth(), "bridge + block map + bridge");
    }
    {
        // The halves of the crossings above, which is what a fused design
        // would have to charge separately.
        heongpu::llama::RectActivation x = fresh();
        heongpu::llama::BatchActivation b = op.to_batch(x, 0, galois);
        const int before = b.column.front().depth();
        std::vector<heongpu::Ciphertext<S>> slots = batch.to_slots(b, galois);
        const int mid = slots.front().depth();
        Row("convert", "BATCH -> SLOT (row bridge)", before, mid,
            "d rotations, d plaintext products");

        heongpu::llama::BatchActivation back =
            batch.from_slots(slots, d, galois);
        Row("convert", "SLOT -> BATCH (row bridge)", mid,
            back.column.front().depth(), "d rotations, d plaintext products");
    }
    {
        heongpu::llama::RectActivation x = fresh();
        heongpu::llama::BatchActivation b = op.to_batch(x, 0, galois);
        std::vector<heongpu::Ciphertext<S>> slots = batch.to_slots(b, galois);
        const int before = slots.front().depth();
        op.block_map(slots, /*inverse=*/true, "levels.block_map", galois);
        Row("convert", "block_map (subring DFT)", before,
            slots.front().depth(), "k-1 rotations, one level, NOT a permutation");
    }

    // -----------------------------------------------------------------------
    // Kang's products
    // -----------------------------------------------------------------------
    {
        heongpu::llama::RectActivation x = fresh();
        const int before = x.column.front().depth();
        const std::vector<double> w = random_matrix(
            static_cast<std::size_t>(channels) * channels, amp, rng);
        heongpu::llama::RectActivation y =
            op.project(x, w, channels, channels, "levels.project", galois);
        Row("product", "Algorithm 5  rect PCMM", before,
            y.column.front().depth(), "RECT in, RECT out (coefficient axis)");

        // The whole point of the coefficient block axis: the output is the
        // input's encoding, so a second projection needs no conversion.
        const int chain_in = y.column.front().depth();
        heongpu::llama::RectActivation z =
            op.project(y, w, channels, channels, "levels.project2", galois);
        Row("product", "Algorithm 5 again, no crossing", chain_in,
            z.column.front().depth(), "chains free -- this is the claim");
    }
    {
        heongpu::llama::RectActivation x = fresh();
        heongpu::llama::BatchActivation a = op.to_batch(x, 0, galois);
        heongpu::llama::RectActivation x2 = fresh();
        heongpu::llama::BatchActivation b = op.to_batch(x2, 0, galois);

        const int before = a.column.front().depth();
        heongpu::llama::BatchActivation t =
            batch.transpose(b, "levels.cmt", galois);
        Row("product", "Algorithm 3  CMT transpose", before,
            t.column.front().depth(), "monomials + tweak + automorphisms");

        heongpu::llama::BatchActivation c =
            op.matmul(a, t, "levels.matmul", galois, relin);
        Row("product", "Algorithm 4  batch CCMM", before,
            c.column.front().depth(), "BATCH in, BATCH out");
    }

    // -----------------------------------------------------------------------
    // The non-linear layers, in slot form
    // -----------------------------------------------------------------------
    const auto slot_batch = [&]() {
        heongpu::llama::RectActivation x = fresh();
        return op.to_slots(x, galois);
    };

    {
        // RMSNorm without the auxiliary track: the constant this design can
        // rely on. Measured twice -- the sublayer as the rect path calls it,
        // crossings included, and the slot-form core on its own, which is what
        // a design that arranges the crossings differently needs.
        const std::vector<double> empty_weights;
        heongpu::llama::RectActivation x = fresh();
        Rect::RectRMSNormConfig cfg;
        cfg.sum_lo = channels * 0.05;
        cfg.sum_hi = channels * 0.30;
        cfg.degree = norm_degree;
        cfg.newton_iterations = 0;
        cfg.fold_mean_into_fit = true;
        cfg.fold_affine_into_mask = true;
        const int before = x.column.front().depth();
        heongpu::llama::RectActivation y =
            op.rms_norm(x, empty_weights, cfg, galois, relin);
        Row("nonlin", "RMSNorm  sublayer, crossings in", before,
            y.column.front().depth(), "to_slots + core + from_slots");

        std::vector<heongpu::Ciphertext<S>> slots = slot_batch();
        heongpu::llama::Llama3Operator::RMSNormConfig slot_cfg;
        slot_cfg.stride = n / 2;
        slot_cfg.count = step;
        slot_cfg.blocked_span = step;
        slot_cfg.channels = channels;
        slot_cfg.token_blocks = 1;
        slot_cfg.eps = cfg.eps;
        slot_cfg.sum_lo = cfg.sum_lo;
        slot_cfg.sum_hi = cfg.sum_hi;
        slot_cfg.degree = norm_degree;
        slot_cfg.newton_iterations = 0;
        slot_cfg.fold_mean_into_fit = true;
        slot_cfg.fold_affine_into_mask = true;
        std::vector<heongpu::Plaintext<S>> no_weights;
        const int slot_before = slots.front().depth();
        std::vector<heongpu::Ciphertext<S>> normed = op.arith().rms_norm(
            slots, no_weights, slot_cfg, galois, relin, nullptr);
        Row("nonlin", "RMSNorm  slot core alone", slot_before,
            normed.front().depth(), "square 1 + mask 1 + fit + product 1");
    }
    {
        // SoftMax, on the encoding the rect attention hands it: the key axis
        // entirely across ciphertexts, one coordinate per part.
        std::vector<heongpu::Ciphertext<S>> slots = slot_batch();
        heongpu::llama::Llama3Operator::SoftmaxConfig cfg;
        cfg.strided = true;
        cfg.stride = n / 2;
        cfg.count = 1;
        cfg.bound = 8.0;
        cfg.iterations = 1;
        cfg.exp_degree = exp_degree;
        cfg.inverse_degree = inverse_degree;
        cfg.inverse_newton = 0;
        cfg.sum_lo = static_cast<double>(d) * std::exp(-2.0);
        cfg.sum_hi = static_cast<double>(d);
        cfg.fold_affine_into_mask = true;
        cfg.pre_scaled_input = true;

        std::vector<std::vector<double>> masks;
        masks.reserve(slots.size());
        for (std::size_t j = 0; j < slots.size(); j++)
            masks.push_back(batch.causal_column_mask(static_cast<int>(j)));

        const int before = slots.front().depth();
        std::vector<heongpu::Ciphertext<S>> p =
            op.arith().softmax(slots, cfg, masks, galois, relin, nullptr);
        Row("nonlin", "SoftMax  slot form, 1 round", before,
            p.front().depth(), "exp + mask + square + 1/x + product");
    }
    {
        std::vector<heongpu::Ciphertext<S>> slots = slot_batch();
        const int before = slots.front().depth();
        heongpu::Ciphertext<S> a = op.arith().silu(
            slots.front(), 10.8, silu_degree, relin, /*pre_scaled=*/true, 1.0);
        Row("nonlin", "SiLU  domain map folded", before, a.depth(),
            "the fit alone");

        heongpu::Ciphertext<S> b = op.arith().silu(
            slots.front(), 10.8, silu_degree, relin, /*pre_scaled=*/false, 1.0);
        Row("nonlin", "SiLU  domain map paid", before, b.depth(),
            "the fit plus its own affine map");

        // The up branch meets the activation at the activation's level, which
        // is where the SwiGLU actually forms this product.
        heongpu::Ciphertext<S> up = slots.front();
        op.arith().drop_to_depth(up, a.depth());
        heongpu::Ciphertext<S> prod =
            op.arith().multiply_and_rescale(a, up, relin);
        Row("nonlin", "SwiGLU gate product", a.depth(), prod.depth(),
            "one ciphertext-ciphertext product");
    }
    {
        // RoPE as implemented, on a slot ciphertext: one rotation to bring the
        // paired channel alongside, then cos and sin as plaintext products.
        std::vector<heongpu::Ciphertext<S>> slots = slot_batch();
        const int before = slots.front().depth();
        std::vector<double> cos_v(n / 2), sin_v(n / 2);
        for (int i = 0; i < n / 2; i++)
        {
            cos_v[i] = std::cos(0.001 * i);
            sin_v[i] = std::sin(0.001 * i);
        }
        // The plaintexts meet the ciphertext at its own level, and at the
        // prime the rescale after them will divide by, so the product comes
        // back at one scale.
        const double plain_scale = static_cast<double>(
            context->get_key_modulus()[slots.front().level()].value);
        const auto encode_here = [&](const std::vector<double>& v) {
            heongpu::Plaintext<S> plain(context);
            encoder.encode(plain, v, plain_scale);
            for (int i = 0; i < before; i++)
                op.arith().mod_drop_inplace(plain);
            return plain;
        };
        heongpu::Plaintext<S> cos_p = encode_here(cos_v);
        heongpu::Plaintext<S> sin_p = encode_here(sin_v);
        heongpu::Ciphertext<S> r = op.arith().rope(slots.front(), cos_p, sin_p,
                                                   step * (d / 2), galois);
        Row("nonlin", "RoPE  as implemented", before, r.depth(),
            "swap rotation + cos/sin plaintext products");
    }
    {
        heongpu::llama::RectActivation a = fresh();
        heongpu::llama::RectActivation b = fresh();
        const std::vector<double> w = random_matrix(
            static_cast<std::size_t>(channels) * channels, amp, rng);
        heongpu::llama::RectActivation deep =
            op.project(b, w, channels, channels, "levels.residual", galois);
        const int before = deep.column.front().depth();
        std::vector<heongpu::Ciphertext<S>> sum =
            op.arith().residual_add(a.column, deep.column);
        Row("nonlin", "residual add", before, sum.front().depth(),
            "level and scale reconciled by a drop and a constant");
    }

    // -----------------------------------------------------------------------
    // The auxiliary track, which is not a constant
    // -----------------------------------------------------------------------
    if (aux)
    {
        std::cout << "[lvl] --- auxiliary track (Figure 2's narrow triangle) ---"
                  << std::endl;
        // What a refresh hands back, which is the number the rows below are
        // relative to.
        {
            heongpu::llama::RectActivation x = fresh();
            const int before = x.column.front().depth();
            op.bootstrap(x, "levels.refresh", galois, relin);
            Row("refresh", "regular bootstrap, RECT", before,
                x.column.front().depth(),
                "hands back a FIXED depth, whatever the chain");
        }
        for (const int entry : {0, 4})
        {
            // The wide track standing at two different depths under the same
            // narrow fit: the product takes the deeper of the two, so what the
            // aux track saves depends on where the wide track was.
            heongpu::llama::RectActivation x = fresh();
            for (int i = 0; i < entry; i++)
            {
                const std::vector<double> w = random_matrix(
                    static_cast<std::size_t>(channels) * channels, amp, rng);
                heongpu::llama::RectActivation y = op.project(
                    x, w, channels, channels, "levels.sink", galois);
                x.column = std::move(y.column);
            }

            Rect::RectRMSNormConfig cfg;
            cfg.sum_lo = channels * 0.05;
            cfg.sum_hi = channels * 0.30;
            cfg.degree = norm_degree;
            cfg.newton_iterations = 0;
            cfg.fold_mean_into_fit = true;
            cfg.fold_affine_into_mask = true;
            cfg.refresh_sum = true;
            const std::vector<double> empty_weights;
            const int before = x.column.front().depth();
            heongpu::llama::RectActivation y =
                op.rms_norm(x, empty_weights, cfg, galois, relin, &galois);
            char note[96];
            std::snprintf(note, sizeof(note),
                          "wide track entered %d level(s) down", entry);
            Row("nonlin", "RMSNorm  auxiliary track", before,
                y.column.front().depth(), note);
        }
    }

    std::cout << "[lvl] ---------------------------------------------------"
              << std::endl;
    std::cout << "[lvl] done" << std::endl;
    return 0;
}

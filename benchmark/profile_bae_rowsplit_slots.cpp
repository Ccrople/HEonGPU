// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// CAN A BLOCK LIVE IN THE ROW-SPLIT LAYOUT? THE TWO FACTS THAT DECIDE IT.
// =======================================================================
//
// §26 built the low-ring Bae product and left one thing open: the low ring is
// a COEFFICIENT-domain construction, and §25.9's slot-resident block is not.
// switch_down splits on coefficient residue classes, so a slot-resident stream
// cannot descend. Moving the block onto the low ring means moving it into the
// row-split coefficient layout, and the question that decides whether that is
// affordable is not the product — it is the two things the non-linear layers
// need:
//
//   1. WHICH SLOT IS WHICH after the crossing to slot form, and
//   2. WHAT THE RMSNORM CHANNEL REDUCTION COSTS in that slot layout.
//
// Both are predictions here, and both are checked rather than argued.
//
// THE PREDICTION
// --------------
// The row-split layout (BaeLayout with cols = c, k = N/c) puts row k*i + u of
// M, column t, at coefficient t*k + u of ciphertext i.
//
// The crossing is NOT the identity on indices, and assuming it was cost this
// file one run. profile_boot_to_slots measures the reading and
// Llama3RectOperator::slot_reading_permutation names it: **coefficient c lands
// in slot bitrev(c)**, bit reversal on log2(N/2) bits, which is what a
// decimation-in-time factorisation leaves behind. Composing:
//
//     slot bitrev(t*k + u) of ciphertext i  =  channel (k*i + u), token t
//
// The channel index u occupies the LOW log2(k) bits of the coefficient index,
// so after reversal it occupies the HIGH log2(k) bits of the slot index. The k
// slots of one channel group are therefore separated by (N/2)/k, not by 1 —
// and the reduction is still
//
//     square, add across the d/k ciphertexts   (free, no rotation)
//     then log2(k) rotate-and-adds by half/k, 2*half/k, ... half/2
//
// with the result landing ALREADY REPLICATED across the group, which is
// exactly the broadcast the normaliser needs. **log2(k) rotations, no mask,
// no extra level, whichever order the DFT leaves** — only the shift amounts
// move. Against §25.9's slot-resident layout, where the reduction is free,
// that is the whole additional cost of the move: three rotations at k = 8.
//
// WHY THE UPPER HALF DOES NOT BITE
// ---------------------------------
// solo_coeff_to_slot carries only coefficients 0..N/2-1: it reads the real
// half and discards the rest. In general that would lose half a row-split
// ciphertext. It does not here, and the reason is the same emptiness that
// makes the product expensive: at B = 1 the tokens occupy t < 128, so the
// highest live coefficient is 127*k + (k-1) = 1023 at k = 8, far inside the
// lower half. **The B = 1 shape that hurts the product is what makes this
// crossing free of a packing constraint.**
//
// WHAT IS NOT MEASURED HERE
// --------------------------
// This is one crossing and one reduction, not a block. It does not run a
// projection, a SiLU or a residual, and it makes no claim about the whole
// sublayer's time.
//
//   HEONGPU_BRS2_LOGN     ring degree exponent          12
//   HEONGPU_BRS2_K        rows per ciphertext            8
//   HEONGPU_BRS2_CHANNELS channels (multiple of K)      32
//   HEONGPU_BRS2_TOKENS   live tokens                   64
//   HEONGPU_BRS2_LIMBS    chain length                  36
//   HEONGPU_BRS2_PRIME    scale prime size              50

#include <heongpu/heongpu.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <vector>

constexpr auto S = heongpu::Scheme::CKKS;

namespace
{
    using Clock = std::chrono::steady_clock;

    double ms_since(Clock::time_point t0)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0)
            .count();
    }

    int EnvInt(const char* name, int fallback)
    {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        return std::atoi(raw);
    }

    /// Best-fit scalar between prediction and measurement, and the residual
    /// left after it. A constant normalisation then shows up as a factor
    /// rather than as failure, which is the distinction profile_boot_to_slots
    /// found worth making.
    void fit_report(const char* label, const std::vector<double>& predicted,
                    const std::vector<double>& got, int count)
    {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < count; ++i)
        {
            num += predicted[static_cast<std::size_t>(i)] *
                   got[static_cast<std::size_t>(i)];
            den += predicted[static_cast<std::size_t>(i)] *
                   predicted[static_cast<std::size_t>(i)];
        }
        const double a = den > 0.0 ? num / den : 0.0;

        double worst = 0.0, worst_raw = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double p = predicted[static_cast<std::size_t>(i)];
            const double g = got[static_cast<std::size_t>(i)];
            worst = std::max(worst, std::abs(g - a * p));
            worst_raw = std::max(worst_raw, std::abs(g - p));
        }
        std::cout << "[brs2] " << label << ": best-fit scalar " << std::fixed
                  << std::setprecision(6) << a << ", residual after it "
                  << std::scientific << std::setprecision(3) << worst
                  << ", raw residual " << worst_raw << std::fixed << std::endl;
    }
    /// perm[c] = the slot that ends up holding coefficient c. Bit reversal on
    /// log2(half) bits; the same map Llama3RectOperator::slot_reading_
    /// permutation() returns, rebuilt here so this target does not need a rect
    /// layout it has no other use for.
    std::vector<int> slot_of_coefficient(int half)
    {
        int bits = 0;
        while ((1 << bits) < half)
            ++bits;
        std::vector<int> p(static_cast<std::size_t>(half));
        for (int c = 0; c < half; ++c)
        {
            unsigned r = 0;
            for (int b = 0; b < bits; ++b)
                if ((static_cast<unsigned>(c) >> b) & 1u)
                    r |= 1u << (bits - 1 - b);
            p[static_cast<std::size_t>(c)] = static_cast<int>(r);
        }
        return p;
    }
} // namespace

int main()
{
    const int log_n = EnvInt("HEONGPU_BRS2_LOGN", 12);
    const int k = EnvInt("HEONGPU_BRS2_K", 8);
    const int channels = EnvInt("HEONGPU_BRS2_CHANNELS", 32);
    const int tokens = EnvInt("HEONGPU_BRS2_TOKENS", 64);
    const int limbs = EnvInt("HEONGPU_BRS2_LIMBS", 36);
    const int prime_bits = EnvInt("HEONGPU_BRS2_PRIME", 50);

    const int n = 1 << log_n;
    const int cols = n / k;
    const int half = n / 2;
    const double scale = std::pow(2.0, prime_bits);

    if (channels % k != 0)
    {
        std::cout << "[brs2] channels must be a multiple of k" << std::endl;
        return 0;
    }
    if (static_cast<long long>(cols) * cols < n)
    {
        std::cout << "[brs2] k is too large: cols = N/k must satisfy Bae's "
                     "cols >= sqrt(N), i.e. k <= sqrt(N)"
                  << std::endl;
        return 0;
    }
    if (tokens * k > half)
    {
        std::cout << "[brs2] the live tokens do not fit in the real half that "
                     "solo_coeff_to_slot carries"
                  << std::endl;
        return 0;
    }

    const int cts = channels / k;
    std::cout << "[brs2] ring N = " << n << ", k = " << k << " rows per "
              << "ciphertext, cols = " << cols << std::endl;
    std::cout << "[brs2] " << channels << " channels in " << cts
              << " ciphertext(s), " << tokens << " live tokens, highest live "
              << "coefficient " << ((tokens - 1) * k + k - 1) << " of " << half
              << std::endl;

    std::vector<int> q_bits(limbs, prime_bits);
    q_bits.front() = 60;
    std::vector<int> p_bits(limbs, 60);

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
    heongpu::HEDecryptor<S> decryptor(context, secret);
    heongpu::HEEncoder<S> encoder(context);
    // Llama3Operator rather than a bare HEArithmeticOperator: solo_coeff_to_slot
    // is protected, and the public entry is bootstrap_to_slots -- which is the
    // honest unit anyway. The row-split stream arrives at the bottom of the
    // chain (the low-ring product left it there, §26.7) and needs a bootstrap
    // before the non-linear layer regardless, so the crossing measured here is
    // the bootstrap it was going to pay, not an extra one. That is MaMBo's
    // claim, priced.
    heongpu::llama::Llama3Operator op(context, encoder, scale);

    const std::vector<int> perm = slot_of_coefficient(half);

    heongpu::BootstrappingConfig boot_config(3, 3, 11, true);
    op.generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

    // The crossing's own keys, plus the log2(k) shifts the reduction needs.
    std::set<int> shift_set;
    for (int r : op.bootstrapping_key_indexs())
        shift_set.insert(r);
    for (int s = half / k; s < half; s <<= 1)
        shift_set.insert(s);
    std::vector<int> shifts(shift_set.begin(), shift_set.end());
    std::cout << "[brs2] keys: " << shifts.size()
              << " rotation indices (crossing + log2(k) = "
              << static_cast<int>(std::log2(static_cast<double>(k)))
              << " reduction shifts)" << std::endl;

    heongpu::Galoiskey<S> galois(context, shifts);
    keygen.generate_galois_key(galois, secret);
    heongpu::Relinkey<S> relin(context);
    keygen.generate_relin_key(relin, secret);

    // ---------------------------------------------------------------------
    // Encode the activation row-split, with the tokens in the live columns
    // and the rest zero -- the real B = 1 shape.
    // ---------------------------------------------------------------------
    std::mt19937_64 rng(20260814u);
    std::uniform_real_distribution<double> dist(-0.4, 0.4);
    std::vector<double> M(static_cast<std::size_t>(channels) * cols, 0.0);
    for (int c = 0; c < channels; ++c)
        for (int t = 0; t < tokens; ++t)
            M[static_cast<std::size_t>(c) * cols + t] = dist(rng);

    // Encoded through the STANDARD coefficient encoder rather than through
    // load_coefficients, so that the Bae staging path is not one of the
    // variables under test here. The index map is the row-split one either
    // way: coefficient t*k + u of ciphertext i is row k*i + u, column t.
    auto encrypt_rows = [&]()
    {
        std::vector<heongpu::Ciphertext<S>> out;
        out.reserve(static_cast<std::size_t>(cts));
        for (int i = 0; i < cts; ++i)
        {
            std::vector<double> coeff(static_cast<std::size_t>(n), 0.0);
            for (int u = 0; u < k; ++u)
                for (int t = 0; t < cols; ++t)
                    coeff[static_cast<std::size_t>(t) * k + u] =
                        M[static_cast<std::size_t>(k * i + u) * cols + t];

            heongpu::Plaintext<S> p(context);
            encoder.encode(p, coeff, scale, heongpu::ExecutionOptions(),
                           heongpu::encoding::COEFFICIENT);
            heongpu::Ciphertext<S> ct(context);
            encryptor.encrypt(ct, p);
            out.push_back(std::move(ct));
        }
        return out;
    };

    auto decode_slots = [&](heongpu::Ciphertext<S>& ct)
    {
        heongpu::Plaintext<S> p(context);
        decryptor.decrypt(p, ct);
        std::vector<double> v;
        encoder.decode(v, p);
        return v;
    };

    // ---------------------------------------------------------------------
    // STEP 0, THE CONTROL. Before asking anything about the row-split layout,
    // re-establish profile_boot_to_slots' own claim inside THIS harness, on a
    // plain COEFFICIENT-encoded vector: slot c must carry coefficient c. If
    // this fails, the parameters or the setup are wrong and nothing below is
    // about the layout at all.
    // ---------------------------------------------------------------------
    {
        std::vector<double> plain_coeffs(static_cast<std::size_t>(n), 0.0);
        for (int c = 0; c < half; ++c)
            plain_coeffs[static_cast<std::size_t>(c)] = dist(rng);

        heongpu::Plaintext<S> p(context);
        encoder.encode(p, plain_coeffs, scale, heongpu::ExecutionOptions(),
                       heongpu::encoding::COEFFICIENT);
        heongpu::Ciphertext<S> ct(context);
        encryptor.encrypt(ct, p);

        heongpu::Ciphertext<S> got_slots =
            op.bootstrap_to_slots(ct, galois, relin);
        const std::vector<double> got = decode_slots(got_slots);

        std::vector<double> permuted(static_cast<std::size_t>(half), 0.0);
        for (int c = 0; c < half; ++c)
            permuted[static_cast<std::size_t>(perm[c])] = plain_coeffs[c];

        fit_report("STEP 0 control, identity (expected to FAIL)", plain_coeffs,
                   got, half);
        fit_report("STEP 0 control, bit reversal (expected to fit)", permuted,
                   got, half);
    }

    std::vector<heongpu::Ciphertext<S>> rows = encrypt_rows();
    std::cout << "[brs2] encrypted " << rows.size() << " row-split ciphertext(s)"
              << " at depth " << rows.front().depth() << std::endl;

    // ---------------------------------------------------------------------
    // FACT 1: which slot is which.
    // ---------------------------------------------------------------------
    std::cout << "[brs2] ---------------------------------------------------"
              << std::endl;
    std::cout << "[brs2] FACT 1  prediction: slot (t*k + u) of ciphertext i "
                 "carries channel (k*i + u), token t"
              << std::endl;

    std::vector<heongpu::Ciphertext<S>> in_slots;
    in_slots.reserve(rows.size());
    cudaDeviceSynchronize();
    const auto t_cross0 = Clock::now();
    for (auto& ct : rows)
        in_slots.push_back(op.bootstrap_to_slots(ct, galois, relin));
    cudaDeviceSynchronize();
    const double cross_ms = ms_since(t_cross0);
    std::cout << "[brs2] crossing (bootstrap_to_slots): " << std::fixed << std::setprecision(3)
              << cross_ms << " ms for " << rows.size() << " ciphertext(s), "
              << (cross_ms / static_cast<double>(rows.size()))
              << " ms each, depth " << in_slots.front().depth() << " (was "
              << rows.front().depth() << ")" << std::endl;

    for (int i = 0; i < cts; ++i)
    {
        std::vector<double> predicted(static_cast<std::size_t>(half), 0.0);
        for (int t = 0; t < tokens; ++t)
            for (int u = 0; u < k; ++u)
                predicted[static_cast<std::size_t>(
                    perm[static_cast<std::size_t>(t) * k + u])] =
                    M[static_cast<std::size_t>(k * i + u) * cols + t];

        std::vector<double> got =
            decode_slots(in_slots[static_cast<std::size_t>(i)]);
        // Over the WHOLE slot vector, not just the live prefix: the live
        // values are scattered by the reversal, and scoring only slots
        // 0..tokens*k would score mostly zeros.
        fit_report(("ciphertext " + std::to_string(i)).c_str(), predicted, got,
                   half);
    }

    // ---------------------------------------------------------------------
    // FACT 2: the channel reduction, in log2(k) rotations.
    // ---------------------------------------------------------------------
    std::cout << "[brs2] ---------------------------------------------------"
              << std::endl;
    std::cout << "[brs2] FACT 2  sum of squares over the channel axis: add "
                 "across ciphertexts, then log2(k) rotate-and-adds"
              << std::endl;

    cudaDeviceSynchronize();
    const auto t_red0 = Clock::now();

    auto square = [&](heongpu::Ciphertext<S>& ct)
    {
        heongpu::Ciphertext<S> copy = ct;
        heongpu::Ciphertext<S> out(context);
        op.multiply(ct, copy, out);
        op.relinearize_inplace(out, relin);
        op.rescale_inplace(out);
        return out;
    };

    heongpu::Ciphertext<S> total = square(in_slots[0]);
    for (std::size_t i = 1; i < in_slots.size(); ++i)
    {
        heongpu::Ciphertext<S> term = square(in_slots[i]);
        op.add(total, term, total);
    }
    const double square_ms = ms_since(t_red0);

    const auto t_rot0 = Clock::now();
    int rotations = 0;
    for (int s = half / k; s < half; s <<= 1)
    {
        heongpu::Ciphertext<S> shifted(context);
        op.rotate_rows(total, shifted, galois, s);
        op.add(total, shifted, total);
        ++rotations;
    }
    cudaDeviceSynchronize();
    const double rot_ms = ms_since(t_rot0);

    std::cout << "[brs2] squares + cross-ciphertext sum " << std::fixed
              << std::setprecision(3) << square_ms << " ms, " << rotations
              << " rotate-and-adds " << rot_ms << " ms" << std::endl;

    std::vector<double> want(static_cast<std::size_t>(half), 0.0);
    for (int t = 0; t < tokens; ++t)
    {
        double s = 0.0;
        for (int c = 0; c < channels; ++c)
            s += M[static_cast<std::size_t>(c) * cols + t] *
                 M[static_cast<std::size_t>(c) * cols + t];
        // The reduction leaves the group sum in every slot of the group, which
        // is the broadcast the normaliser wants and is asserted here as such.
        for (int u = 0; u < k; ++u)
            want[static_cast<std::size_t>(
                perm[static_cast<std::size_t>(t) * k + u])] = s;
    }
    fit_report("channel sum of squares, replicated over the group", want,
               decode_slots(total), half);

    std::cout << "[brs2] ---------------------------------------------------"
              << std::endl;
    std::cout << "[brs2] cost of the move, per RMSNorm: " << rotations
              << " rotations against 0 for the slot-resident layout of §25.9,"
              << std::endl;
    std::cout << "[brs2] and " << cts << " ciphertexts against " << channels
              << " -- the crossing and the bootstrap after it are paid "
              << k << "x fewer times." << std::endl;
    return 0;
}

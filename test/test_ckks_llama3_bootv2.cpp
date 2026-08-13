// Copyright 2026
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Llama3Operator::bootstrap routed through the v2 model.
//
// The module's single refresh entry point (llama3.cu, Llama3Operator::bootstrap)
// called regular_bootstrapping -- the v1 Taylor model -- and there was no way to
// reach regular_bootstrapping_v2 through the Llama-3 API at all; only the
// two-ring benchmark got at it, by going around the operator. use_v2_bootstrapping
// supplies the two switch keys v2's ModRaise needs and flips the entry point over.
//
// Measured on one A6000 at logN 16, holding the number of RETURNED levels equal
// (v1 spends 25 levels of chain depth against v2's 15, so an unmatched
// comparison flatters v1):
//
//     returned   v1         v2         v2 is
//        2       328.3 ms   117.5 ms   2.79x
//       14       682.6 ms   258.8 ms   2.64x
//
// What these tests pin, in order:
//   1. the pair is all-or-nothing -- half of it would ModRaise under the sparse
//      secret and never switch back, which decrypts to noise rather than throwing;
//   2. the default is still v1, so the flag is genuinely opt-in;
//   3. v2 through the module API refreshes to the precision the model is known
//      for and hands back the levels the chain says it should;
//   4. the refresh is an identity on the plaintext, which is the property the
//      rectangular and matrix encodings rely on to be refreshed where they stand.
//
// logN is 13 here, not the 16 the timings above were taken at: the contract
// under test is the routing, and a smaller ring makes the suite runnable on a
// shared card. v2's precision is N-dependent (it falls off above logN 14), so
// the bound below is set from what logN 13 actually delivers.

#include <heongpu/heongpu.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace
{
    constexpr auto S = heongpu::Scheme::CKKS;
    namespace llama = heongpu::llama;

    // The v2 chain of profile_boot_precision: one q0, nbase working primes,
    // then StoC_piece + n_sine + CtoS_piece for the bootstrap's own stages.
    struct BootChain
    {
        static constexpr int kLogN = 13;
        static constexpr int kQ0 = 41;
        static constexpr int kP = 33;
        static constexpr int kStcBits = 32;
        static constexpr int kSineBits = 60;
        static constexpr int kCtsBits = 56;
        static constexpr int kNbase = 4;
        static constexpr int kStcPiece = 3;
        static constexpr int kCtsPiece = 4;
        static constexpr int kSineDeg = 30;
        static constexpr int kDangle = 3;
        static constexpr int kK = 16;

        // ceil(log2(sine_deg + 1)) + double_angle
        static constexpr int kNSine = 5 + kDangle;
        static constexpr int kStcStart = kNbase + kStcPiece;
        static constexpr int kEmStart = kStcStart + kNSine;
        static constexpr int kCtsStart = kEmStart + kCtsPiece;
        static constexpr int kL = 1 + kNbase + kStcPiece + kNSine + kCtsPiece;
    };

    // Everything a v2 refresh needs, built once per test.
    struct BootFixture
    {
        heongpu::HEContext<S> context =
            heongpu::GenHEContext<S>(heongpu::sec_level_type::none);
        std::unique_ptr<heongpu::HEKeyGenerator<S>> keygen;
        std::unique_ptr<heongpu::Secretkey<S>> secret;
        std::unique_ptr<heongpu::Secretkey<S>> sparse_secret;
        std::unique_ptr<heongpu::Publickey<S>> public_key;
        std::unique_ptr<heongpu::Relinkey<S>> relin_key;
        std::unique_ptr<heongpu::Galoiskey<S>> boot_key;
        std::unique_ptr<heongpu::Switchkey<S>> swk_d2s;
        std::unique_ptr<heongpu::Switchkey<S>> swk_s2d;
        std::unique_ptr<heongpu::HEEncoder<S>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<S>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<S>> decryptor;
        std::unique_ptr<llama::Llama3Operator> op;

        double scale = std::pow(2.0, BootChain::kP);
        int slots = 1 << (BootChain::kLogN - 1);

        // v2 keeps a DENSE working secret and switches into a sparse one for
        // the ModRaise alone. That is the security difference from v1, whose
        // ModRaise needs the scheme's own secret to be sparse.
        void build(bool with_v2_keys)
        {
            const size_t N = size_t(1) << BootChain::kLogN;
            context->set_poly_modulus_degree(N);

            std::vector<int> q_bits;
            q_bits.push_back(BootChain::kQ0);
            for (int i = 0; i < BootChain::kNbase; i++)
                q_bits.push_back(BootChain::kP);
            for (int i = 0; i < BootChain::kStcPiece; i++)
                q_bits.push_back(BootChain::kStcBits);
            for (int i = 0; i < BootChain::kNSine; i++)
                q_bits.push_back(BootChain::kSineBits);
            for (int i = 0; i < BootChain::kCtsPiece; i++)
                q_bits.push_back(BootChain::kCtsBits);

            std::vector<Modulus64> q_mods =
                heongpu::generate_primes(N, q_bits);
            std::vector<Data64> q_vals;
            for (auto& m : q_mods)
                q_vals.push_back(m.value);
            std::vector<Data64> p_vals =
                heongpu::generate_proper_primes(Data64(2) *
                                                    Data64(N),
                                                61, 5);
            context->set_coeff_modulus_values(q_vals, p_vals);
            context->generate();

            keygen = std::make_unique<heongpu::HEKeyGenerator<S>>(context);
            secret = std::make_unique<heongpu::Secretkey<S>>(context, 192);
            keygen->generate_secret_key_v2(*secret);
            public_key = std::make_unique<heongpu::Publickey<S>>(context);
            keygen->generate_public_key(*public_key, *secret);
            relin_key = std::make_unique<heongpu::Relinkey<S>>(context);
            keygen->generate_relin_key(*relin_key, *secret);

            if (with_v2_keys)
            {
                sparse_secret =
                    std::make_unique<heongpu::Secretkey<S>>(context, 32);
                keygen->generate_secret_key_v2(*sparse_secret);
                swk_d2s = std::make_unique<heongpu::Switchkey<S>>(context);
                keygen->generate_switch_key(*swk_d2s, *sparse_secret, *secret);
                swk_s2d = std::make_unique<heongpu::Switchkey<S>>(context);
                keygen->generate_switch_key(*swk_s2d, *secret, *sparse_secret);
            }

            encoder = std::make_unique<heongpu::HEEncoder<S>>(context);
            encryptor =
                std::make_unique<heongpu::HEEncryptor<S>>(context, *public_key);
            decryptor =
                std::make_unique<heongpu::HEDecryptor<S>>(context, *secret);
            op = std::make_unique<llama::Llama3Operator>(context, *encoder,
                                                         scale);

            heongpu::EvalModConfig eval_mod_config(
                0, BootChain::kEmStart, 256.0, BootChain::kK,
                BootChain::kSineDeg, BootChain::kDangle, 0, 0.0);
            heongpu::BootstrappingConfigV2 boot_config(
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::SLOTS_TO_COEFFS,
                    BootChain::kStcStart, 2.0, BootChain::kStcPiece),
                eval_mod_config,
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::COEFFS_TO_SLOTS,
                    BootChain::kCtsStart, 2.0, BootChain::kCtsPiece));
            op->generate_bootstrapping_params_v2(scale, boot_config);

            std::vector<int> key_index = op->bootstrapping_key_indexs();
            boot_key =
                std::make_unique<heongpu::Galoiskey<S>>(context, key_index);
            keygen->generate_galois_key(*boot_key, *secret);
        }

        std::vector<Complex64> random_message(uint64_t seed) const
        {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::vector<Complex64> v;
            v.reserve(slots);
            for (int i = 0; i < slots; i++)
                v.push_back(Complex64(dist(rng), dist(rng)));
            return v;
        }
    };

    // log2(1 / max relative error), the figure v2's precision map is quoted in.
    double precision_bits(const std::vector<Complex64>& want,
                          const std::vector<Complex64>& got)
    {
        double worst = 0.0;
        const size_t n = std::min(want.size(), got.size());
        for (size_t i = 0; i < n; i++)
        {
            const double dr = want[i].real() - got[i].real();
            const double di = want[i].imag() - got[i].imag();
            worst = std::max(worst, std::sqrt(dr * dr + di * di));
        }
        if (worst == 0.0)
        {
            return std::numeric_limits<double>::infinity();
        }
        return -std::log2(worst);
    }

} // namespace

// A half-configured pair is the dangerous case: v2 would raise the modulus
// under the sparse secret and never switch back, and CKKS reports that as
// noise, not as an error. So it is refused at the point of configuration.
TEST(Llama3BootV2, SwitchKeyPairIsAllOrNothing)
{
    BootFixture f;
    f.build(/*with_v2_keys=*/true);

    EXPECT_THROW(f.op->use_v2_bootstrapping(f.swk_d2s.get(), nullptr),
                 std::invalid_argument);
    EXPECT_THROW(f.op->use_v2_bootstrapping(nullptr, f.swk_s2d.get()),
                 std::invalid_argument);

    // Refused means unchanged, not half-applied.
    EXPECT_FALSE(f.op->v2_bootstrapping_enabled());

    f.op->use_v2_bootstrapping(f.swk_d2s.get(), f.swk_s2d.get());
    EXPECT_TRUE(f.op->v2_bootstrapping_enabled());

    // And it is reversible, which is what makes it safe to default off.
    f.op->use_v2_bootstrapping(nullptr, nullptr);
    EXPECT_FALSE(f.op->v2_bootstrapping_enabled());
}

// The routing has to be opt-in: an existing caller that never mentions the
// switch keys must keep the v1 behaviour it was built against.
TEST(Llama3BootV2, DefaultsToV1)
{
    BootFixture f;
    f.build(/*with_v2_keys=*/false);
    EXPECT_FALSE(f.op->v2_bootstrapping_enabled());
}

// The refresh itself, through the module API rather than around it.
TEST(Llama3BootV2, RefreshesThroughTheModuleApi)
{
    BootFixture f;
    f.build(/*with_v2_keys=*/true);
    f.op->use_v2_bootstrapping(f.swk_d2s.get(), f.swk_s2d.get());

    std::vector<Complex64> message = f.random_message(20260813ULL);
    heongpu::Plaintext<S> P1(f.context);
    f.encoder->encode(P1, message, f.scale);
    heongpu::Ciphertext<S> C1(f.context);
    f.encryptor->encrypt(C1, P1);

    // bootstrap() drops to one prime itself; spend the chain first so the
    // refresh is doing real work rather than refreshing a fresh encryption.
    for (int i = 0; i < BootChain::kL - 1; i++)
        f.op->mod_drop_inplace(C1);

    heongpu::Ciphertext<S> refreshed =
        f.op->bootstrap(C1, *f.boot_key, *f.relin_key);

    // The bootstrap spends StoC_piece + n_sine + CtoS_piece levels of depth
    // and hands back the rest of the chain.
    EXPECT_EQ(refreshed.depth(),
              BootChain::kStcPiece + BootChain::kNSine + BootChain::kCtsPiece);

    heongpu::Plaintext<S> P_res(f.context);
    f.decryptor->decrypt(P_res, refreshed);
    std::vector<Complex64> got;
    f.encoder->decode(got, P_res);

    const double bits = precision_bits(message, got);
    std::cout << "[ v2 refresh ] worst-slot precision " << bits << " bits"
              << std::endl;

    // v2 at logN 13 measures ~21 bits mean; this is a worst-SLOT bound, which
    // runs about five bits below the mean, with headroom for the noise being
    // redrawn every run. Anything near the v1 model's failure would be far
    // below this, and a silent ModRaise fault would be negative.
    EXPECT_GT(bits, 10.0);
}

// The property the rectangular and matrix encodings depend on: a refresh is an
// identity on the plaintext polynomial, so it does not have to be crossed into
// slot form and back. Checked against the SAME message refreshed twice --
// agreement to within the model's own error, from two independent noise draws.
TEST(Llama3BootV2, RefreshIsAnIdentityOnThePlaintext)
{
    BootFixture f;
    f.build(/*with_v2_keys=*/true);
    f.op->use_v2_bootstrapping(f.swk_d2s.get(), f.swk_s2d.get());

    std::vector<Complex64> message = f.random_message(760813ULL);

    auto refresh_once = [&]()
    {
        heongpu::Plaintext<S> P(f.context);
        f.encoder->encode(P, message, f.scale);
        heongpu::Ciphertext<S> C(f.context);
        f.encryptor->encrypt(C, P);
        for (int i = 0; i < BootChain::kL - 1; i++)
            f.op->mod_drop_inplace(C);
        heongpu::Ciphertext<S> out =
            f.op->bootstrap(C, *f.boot_key, *f.relin_key);
        heongpu::Plaintext<S> Pr(f.context);
        f.decryptor->decrypt(Pr, out);
        std::vector<Complex64> got;
        f.encoder->decode(got, Pr);
        return got;
    };

    std::vector<Complex64> a = refresh_once();
    std::vector<Complex64> b = refresh_once();

    // Both must track the input, and each other: a refresh that permuted or
    // dropped a coefficient half would still look self-consistent, so the
    // check against `message` is the load-bearing one and the a-vs-b check
    // catches a refresh that is merely unstable.
    EXPECT_GT(precision_bits(message, a), 10.0);
    EXPECT_GT(precision_bits(message, b), 10.0);
    EXPECT_GT(precision_bits(a, b), 9.0);
}

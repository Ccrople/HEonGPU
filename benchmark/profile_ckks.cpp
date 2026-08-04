// Single-call Nsight Systems capture targets for HEonGPU CKKS.
//
// Each mode performs exactly one measured operation between cudaProfilerStart()
// and cudaProfilerStop(), so a report captured with
//
//   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop
//
// contains that operation alone: context generation, key generation and a
// warm-up call all fall outside the capture range. Every measured call is
// preceded by a warm-up on its own operand, because the first launch of a
// kernel pays JIT/module-load cost and because the in-place operations here
// consume a level.
//
//   ckks_profile boot     one regular_bootstrapping_v2
//   ckks_profile relin    one relinearisation (a key switch with no automorphism)
//   ckks_profile rotate   one rotation (the same key switch plus an automorphism)
//   ckks_profile pcmm     one batch PCMM (Algorithm 1)
//   ckks_profile ccmm     one batch CCMM (Algorithm 4)
//
// The modulus chain is the one from example/bootstrapping/5_ckks_regular_-
// bootstrapping_v2.cpp: the only chain in this tree demonstrated to bootstrap
// at logN = 16. Every mode shares it, so all five reports are directly
// comparable with each other.

#include <heongpu/heongpu.hpp>

#include <cuda_profiler_api.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::CKKS;

namespace
{

// 25 Q primes (level 24) and 5 P primes, i.e. KEYSWITCHING_METHOD_II with
// ceil(25/5) = 5 digits.
const std::vector<Data64> kQ = {
    0x10000000006e0001, // 60  Q0
    0x10000140001,      // 40
    0xffffe80001,       // 40
    0xffffc40001,       // 40
    0x100003e0001,      // 40
    0xffffb20001,       // 40
    0x10000500001,      // 40
    0xffff940001,       // 40
    0xffff8a0001,       // 40
    0xffff820001,       // 40
    0x7fffe60001,       // 39  StC
    0x7fffe40001,       // 39  StC
    0x7fffe00001,       // 39  StC
    0xfffffffff840001,  // 60  Sine (double angle)
    0x1000000000860001, // 60  Sine (double angle)
    0xfffffffff6a0001,  // 60  Sine (double angle)
    0x1000000000980001, // 60  Sine
    0xfffffffff5a0001,  // 60  Sine
    0x1000000000b00001, // 60  Sine
    0x1000000000ce0001, // 60  Sine
    0xfffffffff2a0001,  // 60  Sine
    0x100000000060001,  // 56  CtS
    0xfffffffff00001,   // 56  CtS
    0xffffffffd80001,   // 56  CtS
    0x1000000002a0001   // 56  CtS
};

const std::vector<Data64> kP = {
    0x1fffffffffe00001, // 61
    0x1fffffffffc80001, // 61
    0x1fffffffffb40001, // 61
    0x1fffffffff500001, // 61
    0x1fffffffff420001  // 61
};

constexpr size_t kPolyModulusDegree = 1 << 16;
constexpr int kDenseSecretWeight = 192;
constexpr int kEphemeralSecretWeight = 32;
// Levels to drop so the ciphertext enters bootstrapping at the bottom.
constexpr int kDropsToBottom = 24;

int EnvInt(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return fallback;
    return std::atoi(v);
}

/// Wall time of the measured region.
///
/// The Nsight summary reports each CUDA API separately, and a bootstrap here
/// spends most of its host time in blocking cudaMemcpy rather than in
/// cudaDeviceSynchronize, so no single row of that table is the answer. Events
/// bracket the whole region and give one number.
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

} // namespace

int main(int argc, char* argv[])
{
    const std::string mode = (argc > 1) ? argv[1] : "boot";
    if (mode != "boot" && mode != "relin" && mode != "rotate" &&
        mode != "pcmm" && mode != "ccmm")
    {
        std::cerr << "usage: " << argv[0] << " <boot|relin|rotate|pcmm|ccmm>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    heongpu::HEContext<Scheme> context =
        heongpu::GenHEContext<Scheme>(heongpu::sec_level_type::none);
    context->set_poly_modulus_degree(kPolyModulusDegree);
    context->set_coeff_modulus_values(kQ, kP);
    context->generate();
    context->print_parameters();

    const double scale = pow(2.0, 40);

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context, kDenseSecretWeight);
    keygen.generate_secret_key_v2(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    const int slot_count = kPolyModulusDegree / 2;
    std::vector<Complex64> message(slot_count, Complex64(0.2, 0.4));

    heongpu::Plaintext<Scheme> P1(context);
    encoder.encode(P1, message, scale);

    // A fresh bottom-level ciphertext. Bootstrapping is not idempotent on its
    // operand's level, so the warm-up and the measured call each need one.
    auto fresh_bottom = [&]()
    {
        heongpu::Ciphertext<Scheme> c(context);
        encryptor.encrypt(c, P1);
        for (int i = 0; i < kDropsToBottom; ++i)
            operators.mod_drop_inplace(c);
        return c;
    };

    auto fresh_top = [&]()
    {
        heongpu::Ciphertext<Scheme> c(context);
        encryptor.encrypt(c, P1);
        return c;
    };

    if (mode == "boot")
    {
        heongpu::Switchkey<Scheme>* swk_dense_to_sparse = nullptr;
        heongpu::Switchkey<Scheme>* swk_sparse_to_dense = nullptr;
        if (kEphemeralSecretWeight > 0)
        {
            heongpu::Secretkey<Scheme> sparse_secret_key(
                context, kEphemeralSecretWeight);
            keygen.generate_secret_key_v2(sparse_secret_key);

            swk_dense_to_sparse = new heongpu::Switchkey<Scheme>(context);
            keygen.generate_switch_key(*swk_dense_to_sparse, sparse_secret_key,
                                       secret_key);

            swk_sparse_to_dense = new heongpu::Switchkey<Scheme>(context);
            keygen.generate_switch_key(*swk_sparse_to_dense, secret_key,
                                       sparse_secret_key);
        }

        heongpu::EvalModConfig eval_mod_config(EnvInt("HEONGPU_PROFILE_EVALMOD", 20));
        heongpu::BootstrappingConfigV2 boot_config(
            heongpu::EncodingMatrixConfig(
                heongpu::LinearTransformType::SLOTS_TO_COEFFS,
                EnvInt("HEONGPU_PROFILE_STOC", 12)),
            eval_mod_config,
            heongpu::EncodingMatrixConfig(
                heongpu::LinearTransformType::COEFFS_TO_SLOTS,
                EnvInt("HEONGPU_PROFILE_CTOS", 24)));

        operators.generate_bootstrapping_params_v2(scale, boot_config);

        std::vector<int> key_index = operators.bootstrapping_key_indexs();
        std::cout << "[profile] galois keys for bootstrapping: "
                  << key_index.size() << std::endl;
        heongpu::Galoiskey<Scheme> galois_key(context, key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        {
            heongpu::Ciphertext<Scheme> warm = fresh_bottom();
            heongpu::Ciphertext<Scheme> discard =
                operators.regular_bootstrapping_v2(warm, galois_key, relin_key,
                                                   swk_dense_to_sparse,
                                                   swk_sparse_to_dense);
            cudaDeviceSynchronize();
        }

        heongpu::Ciphertext<Scheme> C1 = fresh_bottom();
        std::cout << "[profile] level before bootstrapping: " << C1.level()
                  << std::endl;

        cudaProfilerStart();
        RegionTimer timer;
        heongpu::Ciphertext<Scheme> cipher_boot =
            operators.regular_bootstrapping_v2(C1, galois_key, relin_key,
                                               swk_dense_to_sparse,
                                               swk_sparse_to_dense);
        cudaDeviceSynchronize();
        const float elapsed = timer.ms();
        cudaProfilerStop();

        std::cout << "[profile] one bootstrap: " << elapsed << " ms"
                  << std::endl;
        std::cout << "[profile] level after bootstrapping: "
                  << cipher_boot.level() << std::endl;

        heongpu::Plaintext<Scheme> P_res(context);
        decryptor.decrypt(P_res, cipher_boot);
        std::vector<Complex64> decrypted;
        encoder.decode(decrypted, P_res);
        heongpu::PrecisionStats prec = heongpu::get_precision_stats(message, decrypted);
        std::cout << "[profile] " << prec.to_string() << std::endl;

        delete swk_dense_to_sparse;
        delete swk_sparse_to_dense;
    }
    else if (mode == "relin")
    {
        // multiply_inplace leaves a degree-2 ciphertext; relinearising it is one
        // key switch (decompose, mod up, dot the key, mod down) with no
        // automorphism, at the top of the modulus chain.
        {
            heongpu::Ciphertext<Scheme> a = fresh_top();
            heongpu::Ciphertext<Scheme> b = fresh_top();
            operators.multiply_inplace(a, b);
            operators.relinearize_inplace(a, relin_key);
            cudaDeviceSynchronize();
        }

        heongpu::Ciphertext<Scheme> C1 = fresh_top();
        heongpu::Ciphertext<Scheme> C2 = fresh_top();
        operators.multiply_inplace(C1, C2);
        cudaDeviceSynchronize();
        std::cout << "[profile] level before relinearisation: " << C1.level()
                  << std::endl;

        cudaProfilerStart();
        RegionTimer timer;
        operators.relinearize_inplace(C1, relin_key);
        cudaDeviceSynchronize();
        const float elapsed = timer.ms();
        cudaProfilerStop();

        std::cout << "[profile] one relinearisation: " << elapsed
                  << " ms, level " << C1.level() << std::endl;
    }
    else if (mode == "pcmm")
    {
        // Shape matched to the FIDESlib PaperProfile.PCMM capture so the two
        // reports differ only in the modulus chain: module rank d = 1024, so
        // k = 64 and 32 complex matrices ride together, an encrypted 1024 x 8
        // matrix encryption and an 8 x 8 plaintext matrix.
        const int d = EnvInt("HEONGPU_PROFILE_D", 1024);
        const int inner = EnvInt("HEONGPU_PROFILE_INNER", 8);
        const int cols = EnvInt("HEONGPU_PROFILE_COLS", 8);

        heongpu::BatchMatrixLayout layout(static_cast<int>(kPolyModulusDegree),
                                          d);
        heongpu::HEBatchMatrixOperator<Scheme> op(context, layout);
        heongpu::BatchMatrixEncoder enc(layout.k);
        const int nslots = enc.slots();

        const double scale_m = pow(2.0, 20);
        const double scale_u = pow(2.0, 20);

        // BatchMatrixEncoder takes std::complex<double>, not the library's
        // Complex64.
        using cd = std::complex<double>;
        std::vector<std::vector<cd>> M(
            nslots,
            std::vector<cd>(static_cast<size_t>(d) * inner, cd(0.3, -0.2)));
        std::vector<std::vector<cd>> U(
            nslots,
            std::vector<cd>(static_cast<size_t>(inner) * cols, cd(0.1, 0.4)));

        // The encrypted operand: column j of the matrix encryption holds
        // sum_i M[i][j] X^i (Definition 2), which is a coefficient-domain
        // object and so goes in through load_coefficients, not the encoder.
        std::vector<int64_t> m_coeffs;
        enc.encode(M, d, inner, scale_m, m_coeffs);
        std::vector<std::vector<int64_t>> columns;
        heongpu::build_matrix_encryption_coefficients(m_coeffs, layout, d,
                                                      inner, columns);
        std::vector<heongpu::Ciphertext<Scheme>> cts;
        cts.reserve(inner);
        for (int j = 0; j < inner; ++j)
        {
            heongpu::Plaintext<Scheme> pt(context);
            op.load_coefficients(pt, columns[j], scale_m);
            heongpu::Ciphertext<Scheme> c(context);
            encryptor.encrypt(c, pt);
            cts.push_back(std::move(c));
        }

        std::vector<int64_t> u_coeffs;
        enc.encode(U, inner, cols, scale_u, u_coeffs);
        op.encode_plaintext_matrix(u_coeffs, inner, cols, 0, scale_u);

        std::vector<heongpu::Ciphertext<Scheme>*> in;
        for (auto& ct : cts)
            in.push_back(&ct);

        {
            std::vector<heongpu::Ciphertext<Scheme>> warm;
            op.pcmm(warm, in, /*rescale=*/false);
            cudaDeviceSynchronize();
        }

        std::cout << "[profile] PCMM N=" << layout.N << " d=" << d
                  << " k=" << layout.k << " inner=" << inner
                  << " cols=" << cols << " limbs=" << (cts[0].level() + 1)
                  << std::endl;

        std::vector<heongpu::Ciphertext<Scheme>> out;
        cudaProfilerStart();
        RegionTimer timer;
        op.pcmm(out, in, /*rescale=*/false);
        cudaDeviceSynchronize();
        const float elapsed = timer.ms();
        cudaProfilerStop();

        std::cout << "[profile] one PCMM: " << elapsed << " ms, "
                  << out.size() << " output ciphertexts" << std::endl;
    }
    else if (mode == "ccmm")
    {
        // Shape matched to the FIDESlib BatchMatrixProfile.OneCCMM capture so
        // the two reports differ only in the modulus chain: module rank d = 64,
        // so k = 1024 and 512 complex 64 x 64 matrices ride together.
        //
        // CCMM is captured at four limbs, not at the top of the chain, for the
        // same reason FIDESlib captures it at level 3: both operands are d full
        // ciphertexts and the routine holds eight num_limbs x d x N working
        // tensors on top of them, so the cost is linear in the limb count and
        // quadratic in nothing the top of a 25-prime chain can hold.
        const int d = EnvInt("HEONGPU_PROFILE_D", 64);
        const int limbs = EnvInt("HEONGPU_PROFILE_LIMBS", 4);

        heongpu::BatchMatrixLayout layout(static_cast<int>(kPolyModulusDegree),
                                          d);
        heongpu::HEBatchMatrixOperator<Scheme> op(context, layout);
        heongpu::BatchMatrixEncoder enc(layout.k);
        const int nslots = enc.slots();

        // Asymmetric on purpose, as in the CCMM correctness test: step 1 CMTs
        // the right operand and that key switching noise is then multiplied by
        // the left one, so the surviving error depends on scale_b alone.
        const double scale_a = pow(2.0, 25);
        const double scale_b = pow(2.0, 35);

        // The three internal CMTs need the automorphisms X -> X^(2kt+1).
        std::vector<int> rot = heongpu::get_batch_cmt_rotation_indices(layout);
        std::cout << "[profile] galois keys for CMT: " << rot.size()
                  << std::endl;
        heongpu::Galoiskey<Scheme> galois_key(context, rot);
        keygen.generate_galois_key(galois_key, secret_key);

        using cd = std::complex<double>;
        auto encrypt_operand =
            [&](cd fill, double scale,
                std::vector<heongpu::Ciphertext<Scheme>>& cts)
        {
            std::vector<std::vector<cd>> M(
                nslots, std::vector<cd>(static_cast<size_t>(d) * d, fill));
            std::vector<int64_t> coeffs;
            enc.encode(M, d, d, scale, coeffs);
            std::vector<std::vector<int64_t>> columns;
            heongpu::build_matrix_encryption_coefficients(coeffs, layout, d, d,
                                                          columns);
            cts.clear();
            cts.reserve(d);
            for (int j = 0; j < d; ++j)
            {
                heongpu::Plaintext<Scheme> pt(context);
                op.load_coefficients(pt, columns[j], scale);
                heongpu::Ciphertext<Scheme> c(context);
                encryptor.encrypt(c, pt);
                // mod_drop keeps the scale and simply narrows the RNS base, so
                // a matrix encryption survives it: its coefficients are far
                // smaller than the product of the limbs that remain.
                while (c.level() > limbs - 1)
                    operators.mod_drop_inplace(c);
                cts.push_back(std::move(c));
            }
        };

        std::vector<heongpu::Ciphertext<Scheme>> ca, cb;
        encrypt_operand(cd(0.3, -0.2), scale_a, ca);
        encrypt_operand(cd(0.1, 0.4), scale_b, cb);

        std::vector<heongpu::Ciphertext<Scheme>*> pa, pb;
        for (auto& c : ca)
            pa.push_back(&c);
        for (auto& c : cb)
            pb.push_back(&c);

        // ccmm copies the right operand before transposing it and only reads
        // the left, so unlike bootstrapping the warm-up can share the operands.
        {
            std::vector<heongpu::Ciphertext<Scheme>> warm;
            op.ccmm(warm, pa, pb, galois_key, relin_key, operators,
                    /*rescale=*/false);
            cudaDeviceSynchronize();
        }

        std::cout << "[profile] CCMM N=" << layout.N << " d=" << d
                  << " k=" << layout.k << " batch=" << layout.batch
                  << " limbs=" << (ca[0].level() + 1) << std::endl;

        std::vector<heongpu::Ciphertext<Scheme>> out;
        cudaProfilerStart();
        RegionTimer timer;
        op.ccmm(out, pa, pb, galois_key, relin_key, operators,
                /*rescale=*/false);
        cudaDeviceSynchronize();
        const float elapsed = timer.ms();
        cudaProfilerStop();

        std::cout << "[profile] one CCMM: " << elapsed << " ms, " << out.size()
                  << " output ciphertexts" << std::endl;
    }
    else
    {
        const int shift = EnvInt("HEONGPU_PROFILE_SHIFT", 1);
        // Galoiskey takes a non-const reference, so the index list needs a name.
        std::vector<int> shift_vec{ shift };
        heongpu::Galoiskey<Scheme> galois_key(context, shift_vec);
        keygen.generate_galois_key(galois_key, secret_key);

        {
            heongpu::Ciphertext<Scheme> warm = fresh_top();
            operators.rotate_rows_inplace(warm, galois_key, shift);
            cudaDeviceSynchronize();
        }

        heongpu::Ciphertext<Scheme> C1 = fresh_top();
        std::cout << "[profile] level before rotation: " << C1.level()
                  << std::endl;

        cudaProfilerStart();
        RegionTimer timer;
        operators.rotate_rows_inplace(C1, galois_key, shift);
        cudaDeviceSynchronize();
        const float elapsed = timer.ms();
        cudaProfilerStop();

        std::cout << "[profile] one rotation: " << elapsed << " ms, level "
                  << C1.level() << std::endl;
    }

    return EXIT_SUCCESS;
}

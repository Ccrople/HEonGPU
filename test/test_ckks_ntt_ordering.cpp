// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Pins down the output ordering of the forward NTT that HEonGPU drives through
// gpuntt. The batch matrix subring transform is derived from this convention:
// it assumes
//     NTT(m)[j] = m(psi^(2 * brv_N(j) + 1))
// i.e. bit-reversed evaluation order. If gpuntt ever switched to natural order
// the subring transform would silently produce wrong ciphertexts that still
// decrypt to plausible values, so the convention is asserted here rather than
// assumed at the call site.

#include <heongpu/heongpu.hpp>
#include <heongpu/util/util.cuh>
#include <gtest/gtest.h>

#include <gpuntt/ntt_merge/ntt.cuh>

#include <vector>

namespace
{
    // Modular exponentiation on the host.
    Data64 powmod(Data64 b, Data64 e,
                           Data64 m)
    {
        Data64 r = 1;
        b %= m;
        while (e)
        {
            if (e & 1)
                r = static_cast<__uint128_t>(r) * b % m;
            b = static_cast<__uint128_t>(b) * b % m;
            e >>= 1;
        }
        return r;
    }
} // namespace

TEST(HEonGPU, CKKS_ForwardNTTIsBitReversedEvaluationOrder)
{
    const int n_power = 8;
    const int n = 1 << n_power;

    // One CKKS-sized NTT-friendly prime is enough to identify the ordering.
    std::vector<Modulus64> primes =
        heongpu::generate_internal_primes(n, 1);
    ASSERT_EQ(primes.size(), 1u);
    const Data64 p = primes[0].value;

    std::vector<Data64> psi =
        heongpu::generate_primitive_root_of_unity(n, primes);
    ASSERT_EQ(psi.size(), 1u);

    // psi must be a primitive 2n-th root of unity.
    ASSERT_EQ(powmod(psi[0], 2 * n, p), 1u);
    ASSERT_NE(powmod(psi[0], n, p), 1u);

    std::vector<Root64> table =
        heongpu::generate_ntt_table(psi, primes, n_power);

    // m(X) = X, so NTT(m)[j] is exactly the evaluation point itself.
    std::vector<Data64> host(n, 0);
    host[1] = 1;

    heongpu::DeviceVector<Data64> data(host);
    heongpu::DeviceVector<Root64> ntt_table(table);
    heongpu::DeviceVector<Modulus64> modulus(primes);

    gpuntt::ntt_rns_configuration<Data64> cfg = {
        .n_power = n_power,
        .ntt_type = gpuntt::FORWARD,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = cudaStreamDefault};

    gpuntt::GPU_NTT_Inplace(data.data(), ntt_table.data(), modulus.data(), cfg,
                            1, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<Data64> out(n);
    cudaMemcpy(out.data(), data.data(), n * sizeof(Data64),
               cudaMemcpyDeviceToHost);

    int bitrev_matches = 0;
    int natural_matches = 0;
    for (int j = 0; j < n; ++j)
    {
        const Data64 bitrev = powmod(
            psi[0], 2ull * gpuntt::bitreverse(j, n_power) + 1ull, p);
        const Data64 natural = powmod(psi[0], 2ull * j + 1ull, p);
        bitrev_matches += (out[j] == bitrev);
        natural_matches += (out[j] == natural);
    }

    EXPECT_EQ(bitrev_matches, n)
        << "forward NTT is NOT in bit-reversed evaluation order; the batch "
           "matrix subring transform must be re-derived before use";
    EXPECT_LT(natural_matches, n)
        << "forward NTT appears to be in natural evaluation order";
}

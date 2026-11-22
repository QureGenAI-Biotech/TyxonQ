// 单卡 FP64 的 Slater–Condon 算符。分别存储 α、β 自旋串的连接，按需组成乘积空间的行。
#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace connected_rows_detail {

using Mask = uint64_t;

struct Single {
    int column;
    int signed_pair;  // 轨道对编号加一，正负号表示本次替换的费米相位。
    double base;      // 含相位的一体项和同自旋库仑、交换项。
};

struct DoubleLink {
    double value;
    int column;
};

struct HostRows {
    std::vector<int> single_offsets{0}, double_offsets{0};
    std::vector<Single> singles;
    std::vector<DoubleLink> doubles;
    std::vector<double> field;

    size_t bytes() const {
        return (single_offsets.size() + double_offsets.size()) * sizeof(int)
             + singles.size() * sizeof(Single) + doubles.size() * sizeof(DoubleLink)
             + field.size() * sizeof(double);
    }
};

struct RowView {
    const int *single_offsets, *double_offsets;
    const Single* singles;
    const DoubleLink* doubles;
    const double* field;
};

template<typename T>
const T* ptr(const thrust::device_vector<T>& values) {
    return thrust::raw_pointer_cast(values.data());
}

struct DeviceRows {
    thrust::device_vector<int> single_offsets, double_offsets;
    thrust::device_vector<Single> singles;
    thrust::device_vector<DoubleLink> doubles;
    Device field;

    void upload(const HostRows& host) {
        single_offsets = host.single_offsets;
        double_offsets = host.double_offsets;
        singles = host.singles;
        doubles = host.doubles;
        field = host.field;
    }
    RowView view() const {
        return {ptr(single_offsets), ptr(double_offsets), ptr(singles), ptr(doubles), ptr(field)};
    }
    size_t bytes() const {
        return (single_offsets.capacity() + double_offsets.capacity()) * sizeof(int)
             + singles.capacity() * sizeof(Single) + doubles.capacity() * sizeof(DoubleLink)
             + field.capacity() * sizeof(double);
    }
};

inline int orbital_pair(int p, int q) {
    const int hi = std::max(p, q);
    return hi * (hi + 1) / 2 + std::min(p, q);
}

inline int change_sign(Mask& occupied, int orbital) {
    const Mask bit = Mask(1) << orbital;
    const int sign = __builtin_parityll(occupied & (bit - 1)) ? -1 : 1;
    occupied ^= bit;
    return sign;
}

inline int pop_orbital(Mask& bits) {
    const int orbital = __builtin_ctzll(bits);
    bits &= bits - 1;
    return orbital;
}

// 同一份 SBD 压缩积分供对角元、RDM 和新算符使用。只展开轨道对矩阵，供 GPU 直接索引。
inline std::vector<double> expand_integrals(const sbd::twoInt<double>& eri, int pairs) {
    std::vector<double> values(size_t(pairs) * pairs);
    for (int i = 0; i < pairs; ++i) {
        for (int j = 0; j <= i; ++j) {
            values[i * pairs + j] = values[j * pairs + i] = eri.store[i * (i + 1) / 2 + j];
        }
    }
    return values;
}

inline HostRows build_rows(const std::vector<Mask>& labels, const sbd::oneInt<double>& h1,
                           const std::vector<double>& eri, int norb, int pairs) {
    HostRows rows;
    rows.field.resize(labels.size() * pairs, 0.0);
    const int count = static_cast<int>(labels.size());
    const int occupied = __builtin_popcountll(labels.front()), empty = norb - occupied;
    const size_t possible = occupied * empty
                          + size_t(occupied * (occupied - 1) / 2) * (empty * (empty - 1) / 2);
    // 比较扫描所有标签和枚举所有单、双替换的代价，选择候选数较少的一种。
    std::unordered_map<Mask, int> lookup;
    if (possible < labels.size()) {
        lookup.reserve(count);
        for (int j = 0; j < count; ++j) lookup.emplace(labels[j], j);
    }

    for (int row = 0; row < count; ++row) {
        const Mask bra = labels[row];
        double* field = rows.field.data() + size_t(row) * pairs;
        for (Mask bits = bra; bits;) {
            const int k = pop_orbital(bits);
            const double* coulomb = eri.data() + orbital_pair(k, k) * pairs;
            for (int i = 0; i < pairs; ++i) field[i] += coulomb[i];
        }

        auto append = [&](int column, Mask holes, Mask added) {
            Mask state = bra;
            const int p = pop_orbital(holes), r = pop_orbital(added);
            int sign = change_sign(state, p);
            const int rp = orbital_pair(r, p);
            if (!holes) {
                sign *= change_sign(state, r);
                double base = h1.Value(2 * p, 2 * r);
                for (Mask bits = bra & labels[column]; bits;) {
                    const int k = pop_orbital(bits);
                    base += eri[rp * pairs + orbital_pair(k, k)]
                          - eri[orbital_pair(r, k) * pairs + orbital_pair(k, p)];
                }
                rows.singles.push_back({column, sign * (rp + 1), sign * base});
            } else {
                const int q = pop_orbital(holes), s = pop_orbital(added);
                sign *= change_sign(state, q);
                sign *= change_sign(state, s);
                sign *= change_sign(state, r);
                const double value = eri[rp * pairs + orbital_pair(s, q)]
                                   - eri[orbital_pair(r, q) * pairs + orbital_pair(s, p)];
                if (value != 0.0) rows.doubles.push_back({sign * value, column});
            }
        };
        if (lookup.empty()) {
            for (int j = 0; j < count; ++j) {
                const Mask diff = bra ^ labels[j];
                const int changes = __builtin_popcountll(diff);
                if (changes == 2 || changes == 4) append(j, bra & diff, labels[j] & diff);
            }
        } else {
            auto find = [&](Mask holes, Mask added) {
                const auto it = lookup.find(bra ^ holes ^ added);
                if (it != lookup.end()) append(it->second, holes, added);
            };
            const Mask unoccupied = ((Mask(1) << norb) - 1) ^ bra;
            for (Mask occ = bra; occ;) {
                const Mask p = Mask(1) << pop_orbital(occ);
                for (Mask virt = unoccupied; virt;) {
                    const Mask r = Mask(1) << pop_orbital(virt);
                    find(p, r);
                    for (Mask occ2 = occ; occ2;) {
                        const Mask q = Mask(1) << pop_orbital(occ2);
                        for (Mask virt2 = virt; virt2;) {
                            const Mask s = Mask(1) << pop_orbital(virt2);
                            find(p | q, r | s);
                        }
                    }
                }
            }
        }
        if (rows.singles.size() > INT_MAX || rows.doubles.size() > INT_MAX)
            throw std::overflow_error("connected row offsets exceed INT_MAX");
        rows.single_offsets.push_back(static_cast<int>(rows.singles.size()));
        rows.double_offsets.push_back(static_cast<int>(rows.doubles.size()));
    }
    return rows;
}

__device__ inline int interleaved_sign(Mask upper_alpha, Mask beta) {
    return (__popcll(upper_alpha & beta) & 1) ? -1 : 1;
}

__global__ void signed_input(int count, int nb, const Mask* upper, const Mask* beta,
                             const double* input, double* output) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(count)) output[i] = interleaved_sign(upper[i / nb], beta[i % nb]) * input[i];
}

__device__ inline double single_weight(const Single& link, const double* opposite_field) {
    const double field = opposite_field[abs(link.signed_pair) - 1];
    return link.base + (link.signed_pair > 0 ? field : -field);
}

__device__ inline double mixed_weight(int a, int b, int pairs, const double* eri) {
    const double value = eri[(abs(a) - 1) * pairs + abs(b) - 1];
    return (a < 0) == (b < 0) ? value : -value;
}

template<int lanes>
__global__ void multiply(int count, int nb, int pairs, RowView a, RowView b, const double* eri,
                         const Mask* upper, const Mask* beta, const double* diagonal,
                         const double* c, double* out) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = tid / lanes;
    if (row >= size_t(count)) return;
    const int lane = threadIdx.x % lanes;
    const int ia = row / nb, ib = row % nb;
    double sum = 0.0;
    for (int64_t k = int64_t(a.double_offsets[ia]) + lane; k < a.double_offsets[ia + 1]; k += lanes) {
        const auto d = a.doubles[k];
        sum += d.value * c[size_t(d.column) * nb + ib];
    }
    for (int64_t k = int64_t(b.double_offsets[ib]) + lane; k < b.double_offsets[ib + 1]; k += lanes) {
        const auto d = b.doubles[k];
        sum += d.value * c[size_t(ia) * nb + d.column];
    }
    const int a0 = a.single_offsets[ia], na_single = a.single_offsets[ia + 1] - a0;
    const int b0 = b.single_offsets[ib], nb_single = b.single_offsets[ib + 1] - b0;
    for (int k = lane; k < na_single; k += lanes) {
        const auto s = a.singles[a0 + k];
        sum += single_weight(s, b.field + size_t(ib) * pairs) * c[size_t(s.column) * nb + ib];
    }
    for (int k = lane; k < nb_single; k += lanes) {
        const auto s = b.singles[b0 + k];
        sum += single_weight(s, a.field + size_t(ia) * pairs) * c[size_t(ia) * nb + s.column];
    }
    if constexpr (lanes == 1) {
        // 稀疏行由单线程遍历，内层循环复用同一个 α 单激发。
        for (int i = 0; i < na_single; ++i) {
            const auto u = a.singles[a0 + i];
            for (int j = 0; j < nb_single; ++j) {
                const auto v = b.singles[b0 + j];
                sum += mixed_weight(u.signed_pair, v.signed_pair, pairs, eri) * c[size_t(u.column) * nb + v.column];
            }
        }
    } else {
        for (int k = lane; k < na_single * nb_single; k += lanes) {
            const auto u = a.singles[a0 + k / nb_single], v = b.singles[b0 + k % nb_single];
            sum += mixed_weight(u.signed_pair, v.signed_pair, pairs, eri) * c[size_t(u.column) * nb + v.column];
        }
        // 每个有效行都有一整个线程组。最后一个 warp 中的其他行可能已返回。
        const unsigned group_start = (threadIdx.x % 32) / lanes * lanes;
        const unsigned mask = (0xffffffffu >> (32 - lanes)) << group_start;
        for (int stride = lanes / 2; stride; stride /= 2)
            sum += __shfl_down_sync(mask, sum, stride, lanes);
    }
    if (lane == 0) out[row] = interleaved_sign(upper[ia], beta[ib]) * (sum + diagonal[row] * c[row]);
}
} // namespace connected_rows_detail

class ConnectedRows {
    using Mask = connected_rows_detail::Mask;
    connected_rows_detail::DeviceRows alpha_, beta_;
    thrust::device_vector<Mask> upper_alpha_, beta_labels_;
    Device eri_, signed_c_;
    int nb_, count_, pairs_, lanes_;

    template<int lanes>
    void launch(const Device& diagonal, Device& hc) const {
        using namespace connected_rows_detail;
        constexpr int threads = 128;
        const unsigned blocks = (size_t(count_) * lanes + threads - 1) / threads;
        multiply<lanes><<<blocks, threads>>>(count_, nb_, pairs_, alpha_.view(), beta_.view(), ptr(eri_),
            ptr(upper_alpha_), ptr(beta_labels_), ptr(diagonal), ptr(signed_c_), thrust::raw_pointer_cast(hc.data()));
    }
public:
    ConnectedRows(const sbd::oneInt<double>& h1, const sbd::twoInt<double>& eri,
                  const std::vector<Mask>& sa, const std::vector<Mask>& sb,
                  int norb, int lanes, size_t byte_budget)
        : pairs_(norb * (norb + 1) / 2), lanes_(lanes) {
        using namespace connected_rows_detail;
        if (sa.size() > INT_MAX / sb.size())
            throw std::overflow_error("connected row dimension exceeds INT_MAX");
        nb_ = static_cast<int>(sb.size());
        count_ = static_cast<int>(sa.size() * sb.size());
        const size_t labels = sa.size() + sb.size();
        const size_t fixed = (size_t(pairs_) * pairs_ + count_) * sizeof(double) + labels * sizeof(Mask);
        if (fixed + labels * pairs_ * sizeof(double) > byte_budget)
            throw std::runtime_error("SBD CUDA memory budget exceeded before connected row tables");

        const auto integrals = expand_integrals(eri, pairs_);
        const auto a = build_rows(sa, h1, integrals, norb, pairs_);
        const auto b = build_rows(sb, h1, integrals, norb, pairs_);
        if (fixed + a.bytes() + b.bytes() > byte_budget)
            throw std::runtime_error("SBD CUDA memory budget exceeded before connected row upload");

        // 位 k 记录其上方 α 占据数的奇偶性；与 β 标签相交即可转换两种自旋排序。
        auto upper = sa;
        for (auto& x : upper) {
            for (int shift = 1; shift < 64; shift *= 2) x ^= x >> shift;
            x >>= 1;
        }
        alpha_.upload(a);
        beta_.upload(b);
        upper_alpha_ = upper;
        beta_labels_ = sb;
        eri_ = integrals;
        signed_c_.resize(count_);
    }

    size_t memory_bytes() const {
        return alpha_.bytes() + beta_.bytes() + (upper_alpha_.capacity() + beta_labels_.capacity()) * sizeof(Mask)
             + (eri_.capacity() + signed_c_.capacity()) * sizeof(double);
    }

    void apply(const Device& diagonal, const Device& c, Device& hc) {
        using namespace connected_rows_detail;
        signed_input<<<(size_t(count_) + 255) / 256, 256>>>(count_, nb_, ptr(upper_alpha_),
            ptr(beta_labels_), ptr(c), thrust::raw_pointer_cast(signed_c_.data()));
        checked(cudaGetLastError());
        switch (lanes_) {
            case 1:  launch<1>(diagonal, hc);  break;
            case 4:  launch<4>(diagonal, hc);  break;
            case 8:  launch<8>(diagonal, hc);  break;
            case 32: launch<32>(diagonal, hc); break;
        }
        checked(cudaGetLastError());
    }
};

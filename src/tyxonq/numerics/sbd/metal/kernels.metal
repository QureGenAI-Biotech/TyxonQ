// Native gather primitives used by the single-device MLX SBD solver.
// Connection tables and fermionic signs are built once on the host and cached.

// @kernel gather_slots
uint element = thread_position_in_grid.x;
uint row = element / width[0];
uint column = element % width[0];
uint source = uint(address[row]) * width[0] + column;
out[element] = coefficients[source] * signs[row];

// @kernel pair_contract_links
uint element = thread_position_in_grid.x;
uint link = element / width[0];
uint column = element % width[0];
uint pair_index = uint(pair[link]);
uint outer_index = uint(outer[link]);
float value = 0.0f;
for (uint inner = 0; inner < uint(inner_dimension[0]); ++inner) {
    value += matrix[pair_index * uint(inner_dimension[0]) + inner]
           * temporary[inner * temporary_width[0] + outer_index * width[0] + column];
}
out[element] = value * signs[link];

// @kernel pair_contract_mixed
uint element = thread_position_in_grid.x;
uint link = element / alpha_count[0];
uint alpha_index = element % alpha_count[0];
uint pair_index = uint(pair[link]);
uint outer_index = uint(outer[link]);
float value = 0.0f;
for (uint inner = 0; inner < uint(inner_dimension[0]); ++inner) {
    value += matrix[pair_index * uint(inner_dimension[0]) + inner]
           * temporary[inner * temporary_width[0] + alpha_index * width[0] + outer_index];
}
out[element] = value * signs[link];

// @kernel pair_contract
uint element = thread_position_in_grid.x;
uint row = element / width[0];
uint column = element % width[0];
float value = 0.0f;
for (uint inner = 0; inner < uint(inner_dimension[0]); ++inner) {
    value += matrix[row * uint(inner_dimension[0]) + inner]
           * temporary[inner * width[0] + column];
}
out[element] = value;

// @kernel projected_reduce
uint output = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint row = output / width[0];
uint column = output % width[0];
float value = 0.0f;
for (uint inner = lane; inner < uint(inner_dimension[0]); inner += 256) {
    value += matrix[row * uint(inner_dimension[0]) + inner]
           * temporary[inner * width[0] + column];
}
threadgroup float partial[256];
partial[lane] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) partial[lane] += partial[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lane == 0) out[output] = partial[0];

// @kernel spin_exchange
uint det = thread_position_in_grid.x;
uint nb = dimensions[0], n2 = dimensions[1], a = det / nb, b = det % nb;
uint doubly = popcount(alpha_bits[2*a] & beta_bits[2*b])
            + popcount(alpha_bits[2*a+1] & beta_bits[2*b+1]);
float value = (diagonal_constant[0] - float(doubly)) * c[det];
for (uint k = offsets[a]; k < uint(offsets[a+1]); ++k) {
    int left = alpha_moves[2*k];
    int right = beta_lookup[b*n2 + uint(alpha_moves[2*k+1])];
    if (right != 0) {
        float x = c[(uint(abs(left))-1)*nb + uint(abs(right))-1];
        value += (left < 0) == (right < 0) ? -x : x;
    }
}
out[det] = value;

// @kernel connected_rows
uint lane = thread_index_in_simdgroup;
uint nb = dimensions[0], npair = dimensions[1];
uint a = thread_position_in_grid.y, b = thread_position_in_grid.x / 32;
uint det = a*nb + b;
float acc = 0;
for (uint k = aa_offset[a] + lane; k < uint(aa_offset[a+1]); k += 32)
    acc += aa_val[k] * c[uint(aa_col[k])*nb + b];
for (uint k = bb_offset[b] + lane; k < uint(bb_offset[b+1]); k += 32)
    acc += bb_val[k] * c[a*nb + uint(bb_col[k])];
uint a0 = a_offset[a], b0 = b_offset[b];
uint na_link = uint(a_offset[a+1]) - a0, nb_link = uint(b_offset[b+1]) - b0;
for (uint k = lane; k < na_link; k += 32) {
    uint pos = 2*(a0+k);
    int ps = a_single[pos+1];
    float x = b_field[b*npair+uint(abs(ps)-1)] * c[uint(a_single[pos])*nb+b];
    acc += ps > 0 ? x : -x;
}
for (uint k = lane; k < nb_link; k += 32) {
    uint pos = 2*(b0+k);
    int ps = b_single[pos+1];
    float x = a_field[a*npair+uint(abs(ps)-1)] * c[a*nb+uint(b_single[pos])];
    acc += ps > 0 ? x : -x;
}
for (uint k = lane; k < na_link*nb_link; k += 32) {
    uint ia = 2*(a0+k/nb_link), ib = 2*(b0+k%nb_link);
    int pa = a_single[ia+1], pb = b_single[ib+1];
    float x = mixed[uint(abs(pa)-1)*npair+uint(abs(pb)-1)] * c[uint(a_single[ia])*nb+uint(b_single[ib])];
    acc += (pa < 0) == (pb < 0) ? x : -x;
}
float value = simd_sum(acc);
if (lane == 0) out[det] = value + diag[det]*c[det];

// @kernel dot_tiles
uint group = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint dot = group / shape[2], tile = group % shape[2];
uint row = dot / shape[1], col = dot % shape[1];
uint begin = tile * 1024, end = min(begin + 1024, shape[0]);
float value = 0;
for (uint k = begin + lane; k < end; k += 256)
    value += left[row*shape[0] + k] * right_t[col*shape[0] + k];
float sum = simd_sum(value);
threadgroup float partial[8];
if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simdgroup_index_in_threadgroup == 0) {
    float v = thread_index_in_simdgroup < 8 ? partial[thread_index_in_simdgroup] : 0;
    float total = simd_sum(v);
    if (thread_index_in_simdgroup == 0) out[group] = total;
}

// @kernel mixed_rdm_q60
uint g = threadgroup_position_in_grid.x, lane = thread_position_in_threadgroup.x;
uint oa = g/shape[1], ob = g%shape[1], nb = shape[0];
uint b0 = bo[ob], bcount = bo[ob+1] - b0;
uint a0 = ao[oa], terms = (ao[oa+1]-a0)*bcount;
long value = 0;
for (uint t = lane; t < terms; t += 256) {
    uint a = a0+t/bcount, b = b0+t%bcount;
    int ia = al[2*a + 1], ib = bl[2*b + 1];
    uint i = uint(al[2*a])*nb+uint(bl[2*b]);
    uint j = (uint(abs(ia))-1)*nb+uint(abs(ib))-1;
    volatile float rounded = hi[i]*hi[j];
    float error = metal::fma(hi[i], hi[j], -rounded);
    error += metal::fma(hi[i], lo[j], lo[i]*hi[j]);
    long units = long(rint(rounded*0x1p60f)) + long(rint(error*0x1p60f));
    value += (ia < 0) == (ib < 0) ? units : -units;
}
threadgroup long sums[256];
sums[lane] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride=128; stride>0; stride/=2) {
    if (lane<stride) sums[lane] += sums[lane+stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lane == 0) out[g] = sums[0];

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

// @kernel gather_spin
uint element = thread_position_in_grid.x;
uint row = element / column_count[0];
uint column = element % column_count[0];
uint source = uint(rows[row]) * determinant_columns[0] + uint(columns[column]);
out[element] = coefficients[source] * signs[element];

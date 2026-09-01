from __future__ import annotations

import numpy as np
import pytest

from tyxonq.applications.chem.algorithms.lucj import LUCJ
from tyxonq.applications.chem.algorithms.sqd import (
    bitstring_matrix_to_integers,
    recover_configurations,
    reverse_bitstring_halves,
    samples_to_arrays,
)


def test_lucj_raw_bitstring_is_converted_to_sqd_determinants():
    """CAS(4e,4o) 的 HF 样本应映射到 alpha/beta determinant 3。"""
    raw_hf = "11001100"
    sqd_hf = reverse_bitstring_halves(raw_hf)
    bitstrings, probabilities = samples_to_arrays({sqd_hf: 1})

    assert sqd_hf == "00110011"
    assert reverse_bitstring_halves(sqd_hf) == raw_hf
    np.testing.assert_allclose(probabilities, [1.0])
    np.testing.assert_array_equal(
        bitstring_matrix_to_integers(bitstrings[:, :4]),
        [3],
    )
    np.testing.assert_array_equal(
        bitstring_matrix_to_integers(bitstrings[:, 4:]),
        [3],
    )

    with pytest.raises(ValueError, match="even"):
        reverse_bitstring_halves("101")
    with pytest.raises(ValueError, match="only '0' and '1'"):
        reverse_bitstring_halves("10x0")


@pytest.mark.parametrize(
    ("bitstrings", "occupancies"),
    [
        (np.ones((1, 4), dtype=bool), (np.ones(2), np.ones(2))),
        (np.zeros((1, 4), dtype=bool), (np.zeros(2), np.zeros(2))),
    ],
)
def test_recovery_uses_uniform_fallback_for_zero_flip_weights(bitstrings, occupancies):
    """严格 0/1 占据下也必须把两个自旋半区修到目标粒子数。"""
    recovered, probabilities = recover_configurations(
        bitstrings,
        np.array([1.0]),
        avg_occupancies=occupancies,
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=7,
    )

    np.testing.assert_array_equal(np.sum(recovered[:, :2], axis=1), [1])
    np.testing.assert_array_equal(np.sum(recovered[:, 2:], axis=1), [1])
    np.testing.assert_allclose(probabilities, [1.0])


def test_recovery_uses_only_eligible_flip_weights():
    """非候选 bit 有权重时，候选权重全零仍应使用均匀回退。"""
    recovered, probabilities = recover_configurations(
        np.array([[True, True, False, True, True, False]], dtype=bool),
        np.array([1.0]),
        avg_occupancies=(np.ones(3), np.ones(3)),
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=7,
    )

    np.testing.assert_array_equal(np.sum(recovered[:, :3], axis=1), [1])
    np.testing.assert_array_equal(np.sum(recovered[:, 3:], axis=1), [1])
    np.testing.assert_allclose(probabilities, [1.0])


def test_lucj_metadata_includes_the_appended_final_rotation_shape():
    """线路实际包含 final rotation 时，metadata 必须报告对应参数 shape。"""
    params = {
        "orbital_rotations": np.eye(2, dtype=complex)[None, :, :],
        "diag_coulomb_mats": np.zeros((1, 2, 2, 2), dtype=float),
        "final_orbital_rotation": np.array(
            [[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]],
            dtype=complex,
        ),
    }

    circuit = LUCJ(2, 2, 1, "square").get_circuit(params)
    metadata = circuit.metadata["lucj"]

    assert metadata["has_final_orbital_rotation"] is True
    assert metadata["parameter_shapes"]["final_orbital_rotation"] == (2, 2)
    assert any(
        operation["block"] == "final_orbital_rotation"
        for operation in metadata["logical_ops"]
    )

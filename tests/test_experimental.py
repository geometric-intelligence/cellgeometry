"""Test functions in the experimental module."""


import dyn.datasets.experimental as experimental
import dyn.features.basic as basic
import numpy as np


def test_load_treated_osteosarcoma_cells():
    """Test that function load_treated_osteosarcoma_cells runs.

    And outputs correct shape.
    """
    n_sampling_points = 10
    n_cells = 5

    (
        cells,
        cell_shapes,
        lines,
        treatments,
    ) = experimental.load_treated_osteosarcoma_cells(
        n_cells=n_cells,
        n_sampling_points=n_sampling_points,
        quotient=["scaling", "rotation"],
    )
    assert cells.shape == (n_cells, n_sampling_points, 2), cells.shape
    assert len(lines) == n_cells, len(lines)
    assert len(treatments) == n_cells, len(treatments)
    for cell in cell_shapes:
        assert np.allclose(basic.perimeter(cell), 1.0), basic.perimeter(cell)

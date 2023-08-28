"""Utils to load experimental datasets of cells."""

from utils import basic as basic
import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import numpy as np
import skimage.io as skio
from geomstats.geometry.pre_shape import PreShapeSpace
from skimage import measure
from skimage.filters import threshold_otsu
import streamlit as st

M_AMBIENT = 2


if "n_sampling_points" not in st.session_state:
    st.session_state["n_sampling_points"] = 50
    n_sampling_points = st.session_state["n_sampling_points"]
else:
    n_sampling_points = st.session_state["n_sampling_points"]


PRESHAPE_SPACE = PreShapeSpace(m_ambient=M_AMBIENT, k_landmarks=n_sampling_points)
PRESHAPE_SPACE.equip_with_group_action("rotations")
PRESHAPE_SPACE.equip_with_quotient_structure()


def img_to_contour(img):
    """Extract the longest cluster/cell contour from an image.

    Parameters
    ----------
    img : array-like
        Image showing cells or cell clusters.

    Returns
    -------
    contour : array-like, shape=[n_sampling_points, 2]
        Contour of the longest cluster/cell in the image,
        as an array of 2D coordinates on points sampling
        the contour.
    """
    thresh = threshold_otsu(img)
    binary = img > thresh
    contours = measure.find_contours(binary, 0.8)
    lengths = [len(c) for c in contours]
    max_length = max(lengths)
    index_max_length = lengths.index(max_length)
    contour = contours[index_max_length]
    return contour


def _tif_video_to_lists(tif_path):
    """Convert a cell video into two trajectories of contours and images.

    Parameters
    ----------
    tif_path : absolute path of video in .tif format.

    Returns
    -------
    contours_list : list of arrays
        List of 2D coordinates of points defining the contours of each cell
        within the video.
    imgs_list : list of array
        List of images in the input video.
    """
    img_stack = skio.imread(tif_path, plugin="tifffile")
    contours_list = []
    imgs_list = []
    for img in img_stack:
        imgs_list.append(img)
        contour = img_to_contour(img)
        contours_list.append(contour)

    return contours_list, imgs_list


def _interpolate(curve, n_sampling_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Parameters
    ----------
    curve : array-like, shape=[n_points, 2]
    n_sampling_points : int

    Returns
    -------
    interpolation : array-like, shape=[n_sampling_points, 2]
       Discrete curve with n_sampling_points
    """
    old_length = curve.shape[0]
    interpolation = np.zeros((n_sampling_points, 2))
    incr = old_length / n_sampling_points
    pos = np.array(0.0, dtype=np.float32)
    for i in range(n_sampling_points):
        index = int(np.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return gs.array(interpolation, dtype=gs.float32)


def _remove_consecutive_duplicates(curve, tol=1e-2):
    """Preprocess curve to ensure that there are no consecutive duplicate points.

    Returns
    -------
    curve : discrete curve
    """
    dist = curve[1:] - curve[:-1]
    dist_norm = gs.sqrt(gs.sum(dist**2, axis=1))

    if gs.any(dist_norm < tol):
        for i in range(len(curve) - 2):
            if gs.sqrt(gs.sum((curve[i + 1] - curve[i]) ** 2, axis=0)) < tol:
                curve[i + 1] = (curve[i] + curve[i + 2]) / 2

    return curve


# def _exhaustive_align(curve, base_curve):
#     """Project a curve in shape space.

#     This happens in 2 steps:
#     - remove translation (and scaling?) by projecting in pre-shape space.
#     - remove rotation by exhaustive alignment minimizing the L² distance.

#     Returns
#     -------
#     aligned_curve : discrete curve
#     """
#     n_sampling_points = curve.shape[-2]
#     preshape = PreShapeSpace(m_ambient=M_AMBIENT, k_landmarks=n_sampling_points)

#     nb_sampling = len(curve)
#     distances = gs.zeros(nb_sampling)
#     for shift in range(nb_sampling):
#         reparametrized = gs.array(
#             [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
#         )
#         aligned = preshape.align(point=reparametrized, base_point=base_curve)
#         distances[shift] = preshape.total_space_metric.norm(
#             gs.array(aligned) - gs.array(base_curve)
#         )
#     shift_min = gs.argmin(distances)
#     reparametrized_min = gs.array(
#         [curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)]
#     )
#     aligned_curve = preshape.align(point=reparametrized_min, base_point=base_curve)
#     return aligned_curve


def _exhaustive_align(curve, base_curve):
    """Align curve to base_curve to minimize the L² distance.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)
    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = PRESHAPE_SPACE.fiber_bundle.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = PRESHAPE_SPACE.embedding_space.metric.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = PRESHAPE_SPACE.fiber_bundle.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve


def preprocess(
    cells,
    labels_a,
    labels_b,
    n_cells,
    n_sampling_points,
    quotient=["scaling", "rotation"],
):
    """Preprocess a dataset of cells.

    Parameters
    ----------
    cells : list of all cells
        Each cell is an array of points in 2D.
    labels_a : list of str
        List of labels associated with each cell.
    labels_b : list of str
        List of labels associated with each cell.
    n_cells : int
        Number of cells to (randomly) select from this dataset.
    n_sampling_points : int
        Number of sampling points along the boundary of each cell.
    """
    if n_cells > 0:
        print(f"... Selecting only a random subset of {n_cells} / {len(cells)} cells.")
        indices = sorted(
            np.random.choice(gs.arange(0, len(cells), 1), size=n_cells, replace=False)
        )
        cells = [cells[idx] for idx in indices]
        labels_a = [labels_a[idx] for idx in indices]
        labels_b = [labels_b[idx] for idx in indices]

    if n_sampling_points > 0:
        print(
            "... Interpolating: "
            f"Cell boundaries have {n_sampling_points} samplings points."
        )
        interpolated_cells = gs.zeros((n_cells, n_sampling_points, 2))
        for i_cell, cell in enumerate(cells):
            interpolated_cells[i_cell] = _interpolate(cell, n_sampling_points)

        cells = interpolated_cells

    print("... Removing potential duplicate sampling points on cell boundaries.")
    for i_cell, cell in enumerate(cells):
        cells[i_cell] = _remove_consecutive_duplicates(cell)

    print("\n- Cells: quotienting translation.")
    cells = cells - gs.mean(cells, axis=-2)[..., None, :]

    cell_shapes = gs.zeros_like(cells)
    if "scaling" in quotient:
        print("- Cell shapes: quotienting scaling (length).")
        for i_cell, cell in enumerate(cells):
            cell_shapes[i_cell] = cell / basic.perimeter(cell)

    if "rotation" in quotient:
        print("- Cell shapes: quotienting rotation.")
        if "scaling" not in quotient:
            for i_cell, cell_shape in enumerate(cells):
                cell_shapes[i_cell] = _exhaustive_align(cell_shape, cells[0])
        else:
            for i_cell, cell_shape in enumerate(cell_shapes):
                cell_shapes[i_cell] = _exhaustive_align(cell_shape, cell_shapes[0])

    return cells, cell_shapes, labels_a, labels_b


def nolabel_preprocess(
    cells,
    # labels_a,
    # labels_b,
    n_cells,
    n_sampling_points,
    quotient=["scaling", "rotation"],
):
    """Preprocess a dataset of cells.

    Parameters
    ----------
    cells : list of all cells
        Each cell is an array of points in 2D.
    labels_a : list of str
        List of labels associated with each cell.
    labels_b : list of str
        List of labels associated with each cell.
    n_cells : int
        Number of cells to (randomly) select from this dataset.
    n_sampling_points : int
        Number of sampling points along the boundary of each cell.
    """
    # if n_cells > 0:
    #     print(f"... Selecting only a random subset of {n_cells} / {len(cells)} cells.")
    #     indices = sorted(
    #         np.random.choice(gs.arange(0, len(cells), 1), size=n_cells, replace=False)
    #     )
    #     cells = [cells[idx] for idx in indices]
    #     labels_a = [labels_a[idx] for idx in indices]
    #     labels_b = [labels_b[idx] for idx in indices]

    if n_sampling_points > 0:
        print(
            "... Interpolating: "
            f"Cell boundaries have {n_sampling_points} samplings points."
        )
        interpolated_cells = gs.zeros((n_cells, n_sampling_points, 2))
        for i_cell, cell in enumerate(cells):
            interpolated_cells[i_cell] = _interpolate(cell, n_sampling_points)

        cells = interpolated_cells

    print("... Removing potential duplicate sampling points on cell boundaries.")
    for i_cell, cell in enumerate(cells):
        cells[i_cell] = _remove_consecutive_duplicates(cell)

    print("\n- Cells: quotienting translation.")
    cells = cells - gs.mean(cells, axis=-2)[..., None, :]

    cell_shapes = gs.zeros_like(cells)
    if "scaling" in quotient:
        print("- Cell shapes: quotienting scaling (length).")
        for i_cell, cell in enumerate(cells):
            cell_shapes[i_cell] = cell / basic.perimeter(cell)

    if "rotation" in quotient:
        print("- Cell shapes: quotienting rotation.")
        if "scaling" not in quotient:
            for i_cell, cell_shape in enumerate(cells):
                cell_shapes[i_cell] = _exhaustive_align(cell_shape, cells[0])
        else:
            for i_cell, cell_shape in enumerate(cell_shapes):
                cell_shapes[i_cell] = _exhaustive_align(cell_shape, cell_shapes[0])

    return cells, cell_shapes  # , labels_a, labels_b


def load_treated_osteosarcoma_cells(
    n_cells=-1, n_sampling_points=10, quotient=["scaling", "rotation"]
):
    """Load dataset of osteosarcoma cells (bone cancer cells).

    This cell dataset contains cell boundaries of mouse osteosarcoma
    (bone cancer) cells. The dlm8 cell line is derived from dunn and is more
    aggressive as a cancer. The cells have been treated with one of three
    treatments : control (no treatment), jasp (jasplakinolide)
    and cytd (cytochalasin D). These are drugs which perturb the cytoskelet
    of the cells.

    Parameters
    ----------
    n_sampling_points : int
        Number of points used to interpolate each cell boundary.
        Optional, Default: 0.
        If equal to 0, then no interpolation is performed.

    Returns
    -------
    cells : array of n_cells planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order.
        Their barycenters are fixed at 0 (translation has been removed).
        Their lengths are not necessarily equal (scaling has not been removed).
    cell_shapes : array of n_cells planar discrete curves shapes
        Each curve represents the boundary of a cell in counterclockwise order.
        Their barycenters are fixed at 0 (translation has been removed).
        Their lengths are fixed at 1 (scaling has been removed).
        They are aligned in rotation to the first cell (rotation has been removed).
    lines : list of n_cells strings
        List of the cell lines of each cell (dlm8 or dunn).
    treatments : list of n_cells strings
        List of the treatments given to each cell (control, cytd or jasp).
    """
    cells, lines, treatments = data_utils.load_cells()
    return preprocess(
        cells, lines, treatments, n_cells, n_sampling_points, quotient=quotient
    )

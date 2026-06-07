"""Tests for obstacle detection and update logic.

These tests do not require a drone, ROS, model weights, or camera hardware.
They exercise the pure Python/NumPy logic in map_obstacle.py.

map_obstacle is loaded directly via importlib to avoid triggering the
tellodrone package __init__.py, which would pull in av, pygame, and rospy.
"""
import sys
import os
import math
import types
import importlib.util
from pathlib import Path
import numpy as np
import pytest

# Ensure the task/ directory is on the path (for vector.py)
TASK_DIR = Path(__file__).parent.parent / "task"
sys.path.insert(0, str(TASK_DIR))

# Set CWD to project root so calibration_data.npz is found at module load time
os.chdir(Path(__file__).parent.parent)

# Stub out heavy packages that map_obstacle.py imports at module level but that
# are not needed for the pure-logic functions under test (segment_depth, etc.)
def _stub(name: str, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m

_sentinel = object()  # dummy attribute value for unused imports

for _pkg in ("av", "torch", "matplotlib", "matplotlib.pyplot"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _stub(_pkg)

if "PIL" not in sys.modules:
    _pil = _stub("PIL")
    _pil_image = _stub("PIL.Image", fromarray=lambda *a, **k: None)
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _pil_image

if "transformers" not in sys.modules:
    sys.modules["transformers"] = _stub(
        "transformers",
        ZoeDepthImageProcessor=type("ZoeDepthImageProcessor", (), {}),
        ZoeDepthForDepthEstimation=type("ZoeDepthForDepthEstimation", (), {}),
    )

# Load map_obstacle directly, bypassing tellodrone/__init__.py → core.py → av/rospy
_spec = importlib.util.spec_from_file_location(
    "map_obstacle",
    TASK_DIR / "tellodrone" / "map_obstacle.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

segment_depth = _mod.segment_depth
filter_rows = _mod.filter_rows
clean_binary = _mod.clean_binary
detect_obstacles = _mod.detect_obstacles
update_obstacles = _mod.update_obstacles

from vector import Vector3D


# ---------------------------------------------------------------------------
# segment_depth
# ---------------------------------------------------------------------------

def test_segment_depth_returns_correct_shapes():
    depth_map = np.random.rand(64, 64).astype(np.float32)
    clustered, centers = segment_depth(depth_map, cluster_count=3)
    assert clustered.shape == (64, 64)
    assert centers.shape == (3, 1)


def test_segment_depth_labels_in_range():
    depth_map = np.random.rand(32, 32).astype(np.float32)
    clustered, centers = segment_depth(depth_map, cluster_count=4)
    assert clustered.min() >= 0
    assert clustered.max() <= 3


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------

def test_filter_rows_clears_dense_rows():
    binary = np.zeros((10, 100), dtype=np.uint8)
    # Row 5 is nearly all white (99%)
    binary[5, :] = 255
    binary[5, 0] = 0
    result = filter_rows(binary.copy(), threshold_ratio=0.85)
    assert np.all(result[5, :] == 0), "Dense row should be zeroed out"


def test_filter_rows_keeps_sparse_rows():
    binary = np.zeros((10, 100), dtype=np.uint8)
    # Only 20 pixels white in row 3 (20%)
    binary[3, :20] = 255
    result = filter_rows(binary.copy(), threshold_ratio=0.85)
    assert np.any(result[3, :] == 255), "Sparse row should be preserved"


# ---------------------------------------------------------------------------
# clean_binary
# ---------------------------------------------------------------------------

def test_clean_binary_removes_small_components():
    binary = np.zeros((200, 200), dtype=np.uint8)
    # 5×5 blob = 25 px, well below min_area=20000
    binary[10:15, 10:15] = 255
    result = clean_binary(binary, min_area=20000)
    assert np.all(result == 0), "Small component should be removed"


def test_clean_binary_keeps_large_components():
    binary = np.zeros((200, 200), dtype=np.uint8)
    # 150×150 block = 22500 px > 20000
    binary[10:160, 10:160] = 255
    result = clean_binary(binary, min_area=20000)
    assert np.any(result > 0), "Large component should be preserved"


# ---------------------------------------------------------------------------
# detect_obstacles
# ---------------------------------------------------------------------------

def test_detect_obstacles_on_circle():
    import cv2
    binary = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(binary, (100, 100), 30, 255, -1)
    obstacles = detect_obstacles(binary)
    assert len(obstacles) == 1
    (cx, cy), radius = obstacles[0]
    assert abs(cx - 100) < 5
    assert abs(cy - 100) < 5
    assert radius > 0


def test_detect_obstacles_empty_map():
    binary = np.zeros((100, 100), dtype=np.uint8)
    assert detect_obstacles(binary) == []


def test_detect_obstacles_multiple():
    import cv2
    binary = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(binary, (50, 50), 20, 255, -1)
    cv2.circle(binary, (250, 250), 20, 255, -1)
    assert len(detect_obstacles(binary)) == 2


# ---------------------------------------------------------------------------
# update_obstacles
# ---------------------------------------------------------------------------

X_BOUNDS = (0.0, 10.0)
Y_BOUNDS = (0.0, 10.0)
Z_BOUNDS = (0.0, 10.0)


def _obs(x, y, z, r=0.2):
    return (Vector3D(x, y, z), r)


def test_update_adds_new_obstacle():
    result = update_obstacles([], [_obs(5, 5, 5)], 0.5, X_BOUNDS, Y_BOUNDS, Z_BOUNDS)
    assert len(result) == 1


def test_update_replaces_nearby_obstacle():
    # New obstacle within 0.5 m threshold should replace the existing one
    result = update_obstacles(
        [_obs(5.0, 5.0, 5.0)], [_obs(5.1, 5.0, 5.0)],
        0.5, X_BOUNDS, Y_BOUNDS, Z_BOUNDS,
    )
    assert len(result) == 1
    assert math.isclose(result[0][0].x, 5.1, abs_tol=1e-9)


def test_update_adds_distant_obstacle():
    # New obstacle > 0.5 m away should be added separately
    result = update_obstacles(
        [_obs(1.0, 1.0, 1.0)], [_obs(8.0, 8.0, 8.0)],
        0.5, X_BOUNDS, Y_BOUNDS, Z_BOUNDS,
    )
    assert len(result) == 2


def test_update_rejects_out_of_bounds():
    result = update_obstacles([], [_obs(-5, 5, 5)], 0.5, X_BOUNDS, Y_BOUNDS, Z_BOUNDS)
    assert len(result) == 0


def test_update_rejects_large_radius():
    result = update_obstacles(
        [], [(Vector3D(5, 5, 5), 1.5)],  # radius > 1.0 m limit
        0.5, X_BOUNDS, Y_BOUNDS, Z_BOUNDS,
    )
    assert len(result) == 0

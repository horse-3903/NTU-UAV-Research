"""Tests for Vector3D arithmetic and utilities."""
import sys
import math
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "task"))

from vector import Vector3D


def test_add():
    v = Vector3D(1, 2, 3) + Vector3D(4, 5, 6)
    assert (v.x, v.y, v.z) == (5, 7, 9)


def test_sub():
    v = Vector3D(4, 5, 6) - Vector3D(1, 2, 3)
    assert (v.x, v.y, v.z) == (3, 3, 3)


def test_mul_scalar():
    v = Vector3D(1, 2, 3) * 2
    assert (v.x, v.y, v.z) == (2, 4, 6)


def test_truediv():
    v = Vector3D(2, 4, 6) / 2
    assert (v.x, v.y, v.z) == (1, 2, 3)


def test_truediv_zero():
    with pytest.raises(ValueError):
        Vector3D(1, 2, 3) / 0


def test_neg():
    v = -Vector3D(1, -2, 3)
    assert (v.x, v.y, v.z) == (-1, 2, -3)


def test_magnitude():
    v = Vector3D(3, 4, 0)
    assert math.isclose(v.magnitude(), 5.0)


def test_magnitude_zero():
    assert Vector3D(0, 0, 0).magnitude() == 0.0


def test_normalize():
    v = Vector3D(3, 0, 0).normalize()
    assert math.isclose(v.x, 1.0) and v.y == 0 and v.z == 0


def test_normalize_zero_vector():
    v = Vector3D(0, 0, 0).normalize()
    assert (v.x, v.y, v.z) == (0, 0, 0)


def test_dot():
    assert Vector3D(1, 2, 3).dot(Vector3D(4, 5, 6)) == pytest.approx(32.0)


def test_cross():
    v = Vector3D(1, 0, 0).cross(Vector3D(0, 1, 0))
    assert (v.x, v.y, v.z) == (0, 0, 1)


def test_from_arr_roundtrip():
    original = Vector3D(1.5, -2.0, 3.7)
    v = Vector3D.from_arr(original.to_arr())
    assert math.isclose(v.x, 1.5)
    assert math.isclose(v.y, -2.0)
    assert math.isclose(v.z, 3.7)


def test_to_ndarr_shape():
    arr = Vector3D(1, 2, 3).to_ndarr()
    assert arr.shape == (3,)
    assert list(arr) == [1, 2, 3]

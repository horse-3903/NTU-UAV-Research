"""Tests for the Artificial Potential Field controller.

All tests run without drone hardware, ROS, or model weights.
"""
import sys
import math
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "task"))

from vector import Vector3D
from apf import apf, apf_with_bounds

X_BOUNDS = (0.0, 10.0)
Y_BOUNDS = (0.0, 10.0)
Z_BOUNDS = (0.0, 10.0)


# ---------------------------------------------------------------------------
# Basic APF (no bounds)
# ---------------------------------------------------------------------------

def test_attractive_force_direction():
    """Total force should point from cur_pos toward target when no obstacles."""
    cur = Vector3D(0, 0, 0)
    target = Vector3D(5, 0, 0)
    total, attract, repel = apf(cur, target, [], attract_coeff=10, repel_coeff=1, influence_dist=1.0)
    assert total.x > 0, "Should be attracted in positive X"
    assert math.isclose(total.y, 0.0, abs_tol=1e-9)
    assert math.isclose(total.z, 0.0, abs_tol=1e-9)


def test_no_repulsion_outside_influence():
    """Obstacle beyond influence_dist produces zero repulsive force."""
    cur = Vector3D(0, 0, 0)
    target = Vector3D(5, 0, 0)
    obstacle = (Vector3D(10, 0, 0), 0.1)
    total, attract, repel = apf(cur, target, [obstacle], attract_coeff=10, repel_coeff=1, influence_dist=1.0)
    assert math.isclose(repel.magnitude(), 0.0, abs_tol=1e-9)


def test_repulsion_within_influence():
    """Obstacle inside influence_dist produces non-zero repulsive force."""
    cur = Vector3D(0, 0, 0)
    target = Vector3D(5, 0, 0)
    obstacle = (Vector3D(0.3, 0, 0), 0.0)
    total, attract, repel = apf(cur, target, [obstacle], attract_coeff=1, repel_coeff=10, influence_dist=1.0)
    assert repel.magnitude() > 0


def test_repulsion_pushes_away_from_obstacle():
    """Repulsive force direction should point away from the obstacle."""
    cur = Vector3D(0, 0, 0)
    target = Vector3D(10, 0, 0)
    obstacle = (Vector3D(0.5, 0, 0), 0.0)
    total, attract, repel = apf(cur, target, [obstacle], attract_coeff=0, repel_coeff=10, influence_dist=2.0)
    # Obstacle is at +X, so repulsion should push in -X direction
    assert repel.x < 0


# ---------------------------------------------------------------------------
# APF with bounds
# ---------------------------------------------------------------------------

def test_boundary_repulsion_near_min_x():
    """Near minimum X bound: repulsion should push in +X direction."""
    cur = Vector3D(0.5, 5.0, 5.0)
    target = Vector3D(5, 5, 5)
    total, _, repel = apf_with_bounds(
        cur, target, [],
        attract_coeff=0, repel_coeff=1, influence_dist=0,
        x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, z_bounds=Z_BOUNDS,
        bounds_influence_dist=1.0,
    )
    assert total.x > 0, "Should be pushed away from min X wall"


def test_boundary_repulsion_near_max_y():
    """Near maximum Y bound: repulsion should push in -Y direction."""
    cur = Vector3D(5.0, 9.5, 5.0)
    target = Vector3D(5, 5, 5)
    total, _, repel = apf_with_bounds(
        cur, target, [],
        attract_coeff=0, repel_coeff=1, influence_dist=0,
        x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, z_bounds=Z_BOUNDS,
        bounds_influence_dist=1.0,
    )
    assert total.y < 0, "Should be pushed away from max Y wall"


def test_no_boundary_force_at_centre():
    """At the centre of the space, no boundary repulsion should be active."""
    cur = Vector3D(5.0, 5.0, 5.0)
    target = Vector3D(5, 5, 5)
    total, _, _ = apf_with_bounds(
        cur, target, [],
        attract_coeff=0, repel_coeff=1, influence_dist=0,
        x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, z_bounds=Z_BOUNDS,
        bounds_influence_dist=1.0,
    )
    assert math.isclose(total.magnitude(), 0.0, abs_tol=1e-9)


def test_multiple_boundary_forces_sum():
    """Near two walls simultaneously, both repulsions contribute."""
    cur = Vector3D(0.5, 9.5, 5.0)
    target = Vector3D(5, 5, 5)
    total, _, _ = apf_with_bounds(
        cur, target, [],
        attract_coeff=0, repel_coeff=1, influence_dist=0,
        x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, z_bounds=Z_BOUNDS,
        bounds_influence_dist=1.0,
    )
    assert total.x > 0, "Should be pushed from min X wall"
    assert total.y < 0, "Should be pushed from max Y wall"


def test_apf_reaches_target_with_no_obstacles():
    """Simulated drone should converge toward target under pure attraction."""
    pos = Vector3D(0.0, 0.0, 0.0)
    target = Vector3D(5.0, 0.0, 0.0)
    for _ in range(200):
        force, _, _ = apf_with_bounds(
            pos, target, [],
            attract_coeff=5, repel_coeff=0, influence_dist=1.0,
            x_bounds=(-1.0, 11.0), y_bounds=(-1.0, 11.0), z_bounds=(-1.0, 11.0),
            bounds_influence_dist=0.5,
        )
        pos = pos + force * 0.05
        if (pos - target).magnitude() < 0.1:
            break
    assert (pos - target).magnitude() < 0.5, "Drone should converge near target"

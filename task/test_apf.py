from apf import apf_with_bounds
from vector import Vector3D

def run_bounds_repulsion_tests():
    x_bounds = (1.0, 10.0)
    y_bounds = (1.0, 10.0)
    z_bounds = (1.0, 10.0)
    bounds_influence_dist = 1.0
    repel_coeff = 1.0
    attract_coeff = 0.0  # No target attraction for these tests
    target_pos = Vector3D(0, 0, 0)
    obstacles = []  # No obstacles for these tests

    test_cases = [
        # Test 1
        {
            "cur_pos": Vector3D(1.5, 5.0, 5.0),
            "description": "Close to the minimum x-bound",
            "expected_force_dir": Vector3D(1, 0, 0),
        },
        # Test 2
        {
            "cur_pos": Vector3D(5.0, 9.5, 5.0),
            "description": "Close to the maximum y-bound",
            "expected_force_dir": Vector3D(0, -1, 0),
        },
        # Test 3
        {
            "cur_pos": Vector3D(5.0, 5.0, 1.2),
            "description": "Close to the minimum z-bound",
            "expected_force_dir": Vector3D(0, 0, 1),
        },
        # Test 4
        {
            "cur_pos": Vector3D(5.0, 5.0, 5.0),
            "description": "Outside the influence range of all bounds",
            "expected_force_dir": Vector3D(0, 0, 0),
        },
        # Test 5
        {
            "cur_pos": Vector3D(1.2, 9.8, 1.1),
            "description": "Close to multiple bounds",
            "expected_force_dir": Vector3D(1, -1, 1),
        },
        # Test 6
        {
            "cur_pos": Vector3D(1.0, 5.0, 10.0),
            "description": "Exactly at a boundary",
            "expected_force_dir": Vector3D(1, 0, -1),
        },
        # Test 7
        {
            "cur_pos": Vector3D(3.0, 1.2, 7.0),
            "description": "Within influence of a bound but not others",
            "expected_force_dir": Vector3D(0, 1, 0),
        },
    ]

    for test in test_cases:
        cur_pos = test["cur_pos"]
        description = test["description"]
        expected_force_dir = test["expected_force_dir"]

        # Run the APF function
        total_force, _, repulsive_force = apf_with_bounds(
            cur_pos,
            target_pos,
            obstacles,
            attract_coeff,
            repel_coeff,
            influence_dist=0,  # No obstacle influence for these tests
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            bounds_influence_dist=bounds_influence_dist,
        )

        # Normalize forces for comparison
        normalized_force = (
            total_force.normalize() if total_force.magnitude() > 0 else Vector3D(0, 0, 0)
        )

        # Compare the result with the expected direction
        is_correct = normalized_force == expected_force_dir
        print(
            f"Test: {description}\n"
            f"Current Position: {cur_pos}\n"
            f"Total Force: {total_force}\n"
            f"Repulsive Force: {repulsive_force}\n"
            f"Expected Direction: {expected_force_dir}\n"
            f"Result: {'PASS' if is_correct else 'FAIL'}\n"
        )


if __name__ == "__main__":
    run_bounds_repulsion_tests()
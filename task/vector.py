from __future__ import annotations

import math
import numpy as np

class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar) -> Vector3D:
        if scalar != 0:
            return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
        else:
            raise ValueError("Cannot divide by zero.")
    
    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    def dot(self, other: Vector3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> Vector3D:
        magnitude = self.magnitude()
        if magnitude != 0:
            return self / magnitude
        return Vector3D(0, 0, 0)
        
    def is_origin(self) -> bool:
        return self == Vector3D(0, 0, 0)

    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}, z={self.z}"
        
    def __repr__(self) -> str:
        return f"Vector3D({self.x}, {self.y}, {self.z})"
    
    def to_ndarr(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def to_arr(self) -> list:
        return [self.x, self.y, self.z]
    
    @staticmethod
    def from_arr(arr) -> Vector3D:
        return Vector3D(arr[0], arr[1], arr[2])
import math
from __future__ import annotations

class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Vector3D):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        if scalar != 0:
            return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
        else:
            raise ValueError("Cannot divide by zero.")

    def dot(self, other: Vector3D):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D):
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        magnitude = self.magnitude()
        if magnitude != 0:
            return self / magnitude
        else:
            raise ValueError("Cannot normalize a zero vector.")
        
    def is_origin(self):
        return self == Vector3D(0, 0, 0)

    def __repr__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"
    
    @classmethod
    def from_arr(arr):
        return Vector3D(arr[0], arr[1], arr[2])

if __name__ == "__main__":
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    print(v1 + v2)
    print(v1.dot(v2))
    print(v1.cross(v2))

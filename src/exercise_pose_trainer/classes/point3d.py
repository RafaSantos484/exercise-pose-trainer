import math


class Point3d:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_landmark(landmark) -> 'Point3d':
        return Point3d(landmark.x, landmark.y, landmark.z)

    @staticmethod
    def get_angle_between(p1: 'Point3d', p2: 'Point3d', p3: 'Point3d', degrees=False, normalize=True) -> float:
        v1 = p2 - p1
        v2 = p3 - p2
        angle = v1.get_angle(v2, degrees=degrees)
        if normalize:
            if degrees:
                angle /= 180.
            else:
                angle /= math.pi
        return angle

    def __add__(self, other: 'Point3d') -> 'Point3d':
        return Point3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3d') -> 'Point3d':
        return Point3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3d':
        return Point3d(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Point3d':
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return f'({self.x}, {self.y}, {self.z})'

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def get_mid_point(self, other: 'Point3d') -> 'Point3d':
        return Point3d((self.x + other.x) / 2, (self.y + other.y) / 2, (self.z + other.z) / 2)

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Point3d':
        n = self.norm()
        return Point3d(self.x / n, self.y / n, self.z / n)

    def dot(self, other: 'Point3d') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Point3d') -> 'Point3d':
        return Point3d(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def get_angle(self, other: 'Point3d', degrees=False) -> float:
        dot_product = self.dot(other)
        norm_self = self.norm()
        norm_other = other.norm()
        angle_rad = math.acos(dot_product / (norm_self * norm_other))
        if not degrees:
            return angle_rad
        return math.degrees(angle_rad)

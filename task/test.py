from typing import List, Tuple
from vector import Vector3D

c

lst = [[Vector3D.from_arr([3.8544211673736575, 1.7178230486481314, -2.485152958005534]), 0.5135856922952151] ]
lst2 = [[Vector3D.from_arr([2.6472609329223635, 3.098878841449581, -2.5846128964334207]), 0.39637647387338526], [Vector3D.from_arr([4.444624228477478, 1.7896531523964028, -2.3382440353140015]), 0.5546311738190008]]
x = (-0.85, 6.85)
y = (0, 4.5)
z = (-3.75, 0.0)
print(update_obstacles(lst, lst2, threshold=1, x_bounds=x, y_bounds=y, z_bounds=z))
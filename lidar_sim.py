# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:00:37 2023

@author: HJ
"""

import numpy as np
from shapely import affinity, Polygon, Point, LineString, LinearRing
#MakeCircle = lambda position, radius: Point(position).buffer(radius)

from typing import Union
ObstacleLike = Union[Polygon, Point, LineString, LinearRing]

__all__ = ['LidarModel', 'Polygon', 'Point', 'LineString', 'LinearRing']



# 只因光雷达模型
class LidarModel:
    def __init__(self, max_range=500.0, scan_angle=128.0, num_angle=128):
        """激光雷达模型
        Args:
            max_range (float): 最大扫描距离(m).
            scan_angle (float): 最大扫描角度(deg).
            num_angle (int): 扫描角度个数.
        """
        # 雷达参数
        self.max_range = float(max_range)
        self.scan_angle = float(scan_angle) # deg
        self.num_angle = int(num_angle)
        # 射线属性
        self.__angles = np.deg2rad(np.linspace(-self.scan_angle/2, self.scan_angle/2, self.num_angle)) # rad
        self.__d = self.max_range
        # 障碍物
        self.__obstacles: list[ObstacleLike] = []

    @property
    def obstacles(self):
        """所有障碍物"""
        return self.__obstacles

    def add_obstacles(self, obstacles: Union[ObstacleLike, list[ObstacleLike]]):
        """添加/初始化障碍物
        Args:
            obstacle (ObstacleLike, list[ObstacleLike]): 障碍物.
        """
        if isinstance(obstacles, list):
            self.__obstacles.extend(obstacles)
        else:
            self.__obstacles.append(obstacles)
    
    def move_obstacle(self, index: int, dx: float, dy: float, drot: float = 0.0):
        """移动和旋转障碍物
        Args:
            index (int): 障碍物的索引.
            dx (float): x方向移动距离.
            dy (float): y方向移动距离.\n
            drot (float): 逆时针旋转角度(rad).
        """
        obstacle = self.__obstacles[index]
        if drot != 0.0:
            obstacle = affinity.rotate(obstacle, drot, use_radians=True)
        obstacle = affinity.translate(obstacle, dx, dy)
        self.__obstacles[index] = obstacle
        return obstacle
    
    def scan(self, x: float, y: float, yaw: float, *, abs_return=False):
        """扫描
        Args:
            x (float): x坐标(m).
            y (float): y坐标(m).
            yaw (float): 偏航角(rad).\n
            abs_return (bool): 是否返回障碍绝对坐标, 默认False.
        Returns:
            距离点云数据 (ndarray), -1表示没有障碍, shape = (num_angle, ).\n
            障碍绝对坐标 (list), len0 = 0 ~ num_angle, len1 = 2.
        """
        scan_data = -np.ones_like(self.__angles) if not abs_return else []
        for i, angle in enumerate(self.__angles):
            line = LineString([
                (x, y), 
                (x + self.__d * np.cos(yaw + angle), y + self.__d * np.sin(yaw + angle))
            ])
            P, distance = self.__compute_intersection(line)
            if P is not None:
                if not abs_return:
                    scan_data[i] = distance
                else:
                    scan_data.append(P)
        #end one scan
        return scan_data
    
    def get_angle(self, idx: int):
        """获取第idx个距离对应的姿态角度(rad)"""
        return self.__angles[idx]
    
    def __compute_intersection(self, line: LineString):
        """获取激光与障碍物的交点、距离"""
        P_nearest = None
        distance = self.__d
        for obstacle in self.__obstacles:
            #if obstacle.intersects(line): # 判断是否有交集
            intersections = obstacle.intersection(line) # 线段和图形的交集: 点或线 / Multi, 不可能是MultiPolygon
            if intersections.is_empty:
                continue
            if intersections.geom_type in {'MultiPoint', 'MultiLineString', 'GeometryCollection'}:
                multi_geom = list(intersections.geoms)
            else:
                multi_geom = [intersections]
            for single_geom in multi_geom:
                for P in single_geom.coords: # list(coords)返回组成线段的点的列表 [(x, y), ...]
                    d = np.linalg.norm(np.array(P) - line.coords[0])
                    if d < distance:
                        distance = d
                        P_nearest = list(P) # [x, y]
                #end for
            #end for
        #end for
        return P_nearest, distance
    



# # 绘制shapely的图形
# def plot_obstacles(obstacles: Union[ObstacleLike, list[ObstacleLike]], ax=None, *, color='gray'):
#     """绘制Polygon、Point、LineString、LinearRing图形"""
#     if ax is None:
#         import matplotlib.pyplot as plt
#         ax = plt.gca()

#     if isinstance(obstacles, list):
#         for obstacle in obstacles:
#             plot_obstacles(obstacle, ax, color=color)
#     else:
#         if isinstance(obstacles, Polygon): # 面 (Polygon 或 buffer)
#             ax.add_patch(plt.Polygon(obstacles.exterior.coords, closed=True, fc=color))
#         elif isinstance(obstacles, Point): # 点 (没有buffer)
#             ax.add_line(plt.Line2D(obstacles.x, obstacles.y, color=color, marker='x'))
#         else:                             # 线段 / 闭合线段 (没有buffer)
#             ax.add_line(plt.Line2D(*obstacles.xy, color=color))





r"""
1. Polygon
buffer也是Polygon
有exterior属性(LinearRing), 返回面的封闭边 LinearRing(coords)
有interiors属性(iter[LinearRing]), 返回面内所有空洞的封闭边的可迭代对象, 通过list(interiors)转换成list[LinearRing]
NOTE Polygon没有 coords 和 xy 属性 (Polygon是面, 不是点和线!!)

2. Point
有coords属性, 返回 [[x, y]]
有xy属性, 返回 (array('d', [x]), array('d', [y])), 可通过list转换成 ([x], [y])
有x/y属性, 返回 x 和 y

3. LineString
有coords属性, 返回线段的组成 [(x0, y0), ...]
有xy属性, 返回 (array('d', [x0, ...]), array('d', [y0, ...])), 可通过list转换成 ([x0, ...], [y0, ...])

4. LinearRing
有coords属性, 返回闭合线段的组成 [(x0, y0), ..., (x0, y0)]
有xy属性, 返回 (array('d', [x0, ..., x0]), array('d', [y0, ..., y0])), 可通过list转换成 ([x0, ..., x0], [y0, ..., y0])

5. 多图形
MultiPoint, MultiLineString, MultiPolygon, GeometryCollection(混合类型)
有geoms属性(iter), 返回包含所有图形的迭代器, NOTE GeometryCollection返回list???

6. 公共属性
均有geom_type属性(str), 返回图形的类别, 可以代替isinstance判断

均有centroid属性(Point), 返回形心point, 可用于避障算法
均有area属性(float), 返回图形面积, NOTE 闭合线段的面经为0
均有length属性(float), 返回图形的周长



"""










# debug
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from collections import deque
    
    # 创建LIDAR模型
    car_lidar = LidarModel(max_range=100.0, scan_angle=128.0, num_angle=128)
    car_lidar.add_obstacles([
            Polygon([(40, 60), (60, 60), (60, 80), (40, 80)]), 
            Polygon([(-75, -50), (-75, -75), (-50, -75)]).buffer(10), 
            Polygon([(-75, 75), (-50, 100), (-60, 75), (-50, 50), (-75, 25), (-100, 75)], holes=([(-75, 50), (-60, 50), (-70, 60)], )),
            LineString([(50, -75), (95, -50), (95, 50)]),
            Point(10, 10).buffer(25),
            Point(0, -75).buffer(2),
            LinearRing([(75, 25), (75, 75), (100, 75)]),
            Point(-75, 0),
        ])

    # 创建可视化窗口
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    # 创建障碍物可视化对象
    show_obstacles = []
    for obstacle in car_lidar.obstacles:
        if isinstance(obstacle, Polygon): # 面 (Polygon 或 buffer)
            show_obstacles.append(ax.add_patch(plt.Polygon(obstacle.exterior.coords, fc='gray')))
 
        elif isinstance(obstacle, Point): # 点 (没有buffer)
            show_obstacles.append(ax.add_line(plt.Line2D(*obstacle.xy, color='gray', marker='x')))
        else:                             # 线段 / 闭合线段 (没有buffer)
            show_obstacles.append(ax.add_line(plt.Line2D(*obstacle.xy, color='gray')))
    
    # 创建扫描区域可视化对象
    show_scan_points, = ax.plot([], [], 'ro', markersize=2.5)
    show_scan_left, = ax.plot([], [], 'g--', linewidth=0.5)
    show_scan_right, = ax.plot([], [], 'g--', linewidth=0.5)

    # 创建车辆轨迹可视化对象
    show_car_path, = ax.plot([], [], 'k-.', linewidth=1.5)
    show_car_point, = ax.plot([], [], 'bo')

    # 动画更新
    car_path = deque(maxlen=80)
    def update(frame):
        # 移动
        x, y, yaw = 50*np.cos(frame), 50*np.sin(frame), frame+np.pi/2
        car_path.append([x, y])
        
        show_car_path.set_data(np.array(car_path).T)
        show_car_point.set_data(x, y)

        # 移动0号障碍物
        obstacle = car_lidar.move_obstacle(0, -2.5*np.sin(frame), 0, -frame/70)
        if isinstance(obstacle, Polygon):
            show_obstacles[0].set_xy(np.array(obstacle.exterior.coords))
        else:
            show_obstacles[0].set_data(*obstacle.xy)
        
        # 旋转1号障碍物
        obstacle = car_lidar.move_obstacle(1, 0, 0, 0.03)
        if isinstance(obstacle, Polygon):
            show_obstacles[1].set_xy(np.array(obstacle.exterior.coords))
        else:
            show_obstacles[1].set_data(*obstacle.xy)

        # 扫描
        scan_data = car_lidar.scan(x, y, yaw, abs_return=True)
        
        if scan_data:
            points = np.array(scan_data)
            show_scan_points.set_data(points[:, 0], points[:, 1])
        else:
            show_scan_points.set_data([], [])
        
        x1 = x + car_lidar.max_range * np.cos(yaw + np.deg2rad(car_lidar.scan_angle/2))
        x2 = x + car_lidar.max_range * np.cos(yaw - np.deg2rad(car_lidar.scan_angle/2))
        y1 = y + car_lidar.max_range * np.sin(yaw + np.deg2rad(car_lidar.scan_angle/2))
        y2 = y + car_lidar.max_range * np.sin(yaw - np.deg2rad(car_lidar.scan_angle/2))
        show_scan_left.set_data([x, x1], [y, y1])
        show_scan_right.set_data([x, x2], [y, y2])
        
    # 创建动画对象
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 100), interval=5)
    plt.show()
    


    
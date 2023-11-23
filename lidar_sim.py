# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:00:37 2023

@author: HJ
"""

import numpy as np
from shapely import affinity
from shapely import Polygon, Point, LineString, LinearRing
#MakeCircle = lambda position, radius: Point(position).buffer(radius)

from typing import Union
ObstacleLike = Union[Polygon, Point, LineString, LinearRing]

__all__ = ['LidarModel', 'plot_shapely']



# 绘制shapely的图形
def plot_shapely(geom, ax=None, color=None):
    from shapely import geometry as geo
    from shapely.plotting import plot_line, plot_points, plot_polygon
    if isinstance(geom, (geo.MultiPolygon, geo.Polygon)):
        plot_polygon(geom, ax, False, color)
    elif isinstance(geom, (geo.MultiPoint, geo.Point)):
        plot_points(geom, ax, color, marker='x') # 点没有形状, 为了和小圆区分, 用x表示
    elif isinstance(geom, (geo.MultiLineString, geo.LineString, geo.LinearRing)):
        plot_line(geom, ax, False, color)



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
        self.__obstacles: list[ObstacleLike] = [] # 障碍物形状
        self.__obstacle_intensities: list[int] = [] # 障碍物反射强度

    @property
    def obstacles(self):
        """所有障碍物"""
        return self.__obstacles

    def add_obstacles(self, obstacles: Union[ObstacleLike, list[ObstacleLike]], intensities: Union[int, list[int]]=255):
        """添加/初始化障碍物
        Args:
            obstacles (ObstacleLike, list[ObstacleLike]): 障碍物.
            intensities (int, list[int]): 障碍物反射强度, 0~255.
        """
        if isinstance(obstacles, list):
            if isinstance(intensities, list):
                assert len(obstacles) == len(intensities), "障碍物个数必须和反射强度个数相同"
            else:
                intensities = [intensities] * len(obstacles)
            self.__obstacles.extend(obstacles)
            self.__obstacle_intensities.extend(intensities)
        else:
            self.__obstacles.append(obstacles)
            self.__obstacle_intensities.append(intensities)
    
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
    
    def scan(self, x: float, y: float, yaw: float, *, mode=0):
        """扫描

        Args:
            x (float): x坐标(m).\n
            y (float): y坐标(m).\n
            yaw (float): 偏航角(rad).\n
            mode (int): 返回模式(默认0)\n

        Returns:
            scan_data (ndarray): 激光扫描数据, shape = (3, num_angle), 第0维为扫描角度, 第1维为测距(-1表示没障碍, -2表示在Polygon障碍里面), 第2维表示点云强度.\n
            scan_points (list[list], mode!=0): 测量到的障碍点的位置, 空list无障碍, len0 = 0~num_angle, len1 = 2.\n
        """
        scan_data = np.vstack((self.__angles, -np.ones_like(self.__angles), np.zeros_like(self.__angles))) # (3, num_angle)
        scan_points = []
        # 碰撞检测
        for o in self.__obstacles:
            if o.geom_type == "Polygon" and o.contains(Point(x, y)):
                scan_data[1, :] = -2
                return scan_data if mode == 0 else (scan_data, scan_points)
        # 雷达测距
        for i, angle in enumerate(self.__angles):
            line = LineString([
                (x, y), 
                (x + self.__d * np.cos(yaw + angle), y + self.__d * np.sin(yaw + angle))
            ])
            P, distance, intensity = self.__compute_intersection(line)
            if P is not None:
                scan_data[1][i] = distance
                scan_data[2][i] = intensity
                if mode != 0:
                    scan_points.append(P)
        #end one scan
        return scan_data if mode == 0 else (scan_data, scan_points)
    
    def __compute_intersection(self, line: LineString):
        """获取激光与障碍物的交点、测量距离、反射强度"""
        P_nearest = None
        distance = self.__d # 距离 0-max_range
        intensity = None    # 点云强度 0-255
        for obstacle, obstacle_intensity in zip(self.__obstacles, self.__obstacle_intensities):
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
                        intensity = obstacle_intensity
                #end for
            #end for
        #end for
        return P_nearest, distance, intensity
    














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
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:00:37 2023

@author: HJ
"""

import numpy as np
from shapely.geometry import Polygon, Point, LineString
#MakeCircle = lambda position, radius: Point(position).buffer(radius)

from typing import Union
ObstacleLike = Union[Polygon, Point, LineString]

__all__ = ['Polygon', 'Point', 'LineString', 'LidarModel']


class LidarModel:
    def __init__(self, max_range=500.0, scan_angle=128.0, num_angle=128):
        """激光雷达模型
        Args:
            max_range (float): 最大扫描距离(m).
            scan_angle (float): 最大扫描角度(deg).
            num_angle (int): 扫描角度个数.
        """
        # 雷达参数
        self.max_range = max_range
        self.scan_angle = scan_angle # deg
        self.num_angle = int(num_angle)
        # 射线属性
        self.__angles = np.deg2rad(np.linspace(-self.scan_angle/2, self.scan_angle/2, self.num_angle)) # rad
        self.__d = max_range
        # 障碍物
        self.__obstacles: list[ObstacleLike] = []

    def add_obstacle(self, obstacle: Union[ObstacleLike, list[ObstacleLike]]):
        """添加环境障碍物
        Args:
            obstacle (ObstacleLike, list[ObstacleLike]): 障碍物.
            ObstacleLike = Union[Polygon, Point, LineString]
        """
        if isinstance(obstacle, list):
            self.__obstacles += obstacle
        else:
            self.__obstacles.append(obstacle)
    
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
            #if obstacle.intersects(line): # 判断是否相交
            inter_line = obstacle.intersection(line) # 交线段
            if inter_line.is_empty:
                continue
            for P in inter_line.coords: # coords返回组成图像的点的列表 [(x, y), ...]
                d = np.linalg.norm(np.array(P) - line.coords[0])
                if d < distance:
                    distance = d
                    P_nearest = list(P) 
        #end for
        return P_nearest, distance
    



















# debug
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from collections import deque
    
    # 障碍物
    obstacle_list = [
            Point(10, 10).buffer(25),
            Polygon([(40, 60), (60, 60), (60, 80), (40, 80)]), 
            Polygon([(-75, -50), (-75, -75), (-50, -75)]).buffer(10), 
            Polygon([(-75, 75), (-50, 100), (-60, 75), (-50, 50), (-75, 25),(-100, 75)]),
            LineString([(50, -75), (95, -50), (95, 50)]),
            Point(0, -75).buffer(1),
        ]
    # 创建LIDAR模型
    car_lidar = LidarModel(max_range=100.0, scan_angle=128.0, num_angle=128)
    car_lidar.add_obstacle(obstacle_list)

    # 创建可视化窗口
    fig, ax = plt.subplots()
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    # 创建障碍物可视化对象
    for obstacle in obstacle_list:
        if isinstance(obstacle, Polygon): # 多边形、有buffer的线段/点
            ax.add_patch(plt.Polygon(obstacle.exterior.coords, closed=True, fc='gray'))
        elif isinstance(obstacle, Point): # 点 (没有buffer)
            #ax.add_patch(plt.Circle(obstacle.centroid, obstacle.radius, fc='gray')) # 有buffer时才有radius属性!
            ax.add_line(plt.Line2D(*obstacle.xy, color='gray', marker='x'))
        elif isinstance(obstacle, LineString): # 线段 (没有buffer)
            ax.add_line(plt.Line2D(*obstacle.xy, color='gray'))
            
    # 创建扫描区域可视化对象
    scan_points, = ax.plot([], [], 'ro', markersize=2.5)
    scan_left, = ax.plot([], [], 'g--', linewidth=0.5)
    scan_right, = ax.plot([], [], 'g--', linewidth=0.5)

    # 创建车辆轨迹可视化对象
    path_line, = ax.plot([], [], 'k-.', linewidth=1.5)
    path_point, = ax.plot([], [], 'bo')

    # 动画更新
    car_path = deque(maxlen=80)
    def update(frame):
        # 移动
        x, y, yaw = 50*np.cos(frame), 50*np.sin(frame), frame+np.pi/2
        car_path.append([x, y])
        
        path_line.set_data(np.array(car_path).T)
        path_point.set_data(x, y)

        # 扫描
        scan_data = car_lidar.scan(x, y, yaw, abs_return=True)
        
        if scan_data:
            points = np.array(scan_data)
            scan_points.set_data(points[:, 0], points[:, 1])
        else:
            scan_points.set_data([], [])
        
        x1 = x + car_lidar.max_range * np.cos(yaw + np.deg2rad(car_lidar.scan_angle/2))
        x2 = x + car_lidar.max_range * np.cos(yaw - np.deg2rad(car_lidar.scan_angle/2))
        y1 = y + car_lidar.max_range * np.sin(yaw + np.deg2rad(car_lidar.scan_angle/2))
        y2 = y + car_lidar.max_range * np.sin(yaw - np.deg2rad(car_lidar.scan_angle/2))
        scan_left.set_data([x, x1], [y, y1])
        scan_right.set_data([x, x2], [y, y2])
        
    # 创建动画对象
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 100), interval=5)
    plt.axis('equal')
    plt.show()
    


    
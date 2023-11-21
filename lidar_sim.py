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
            scan_data (ndarray): 激光扫描数据, shape = (3, num_angle), 第0维为扫描角度, 第1维为测距(-1表示没障碍), 第2维表示点云强度(0表示没障碍).\n
            scan_points (list[list], mode!=0): 测量到的障碍点的位置, 空list无障碍, len0 = 0~num_angle, len1 = 2.\n
        """
        scan_data = np.vstack((self.__angles, -np.ones_like(self.__angles), np.zeros_like(self.__angles))) # (3, num_angle)
        scan_points = []
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
            Polygon([(-75, 75), (-50, 100), (-60, 75), (-50, 50), (-75, 25), (-100, 75)], holes=[[(-75, 35), (-60, 50), (-75, 65)],]),
            Point(10, 10).buffer(25),
            Point(0, -75).buffer(2),
            LineString([(50, -75), (95, -50), (95, 50)]),
            LinearRing([(75, 25), (75, 75), (100, 75)]),
            Point(-75, 0),
        ], 
        [50, 100, 150, 255, 10, 200, 120, 0] # 材质
        )
    #fig, ([ax0, ax1, ax2]) = plt.subplots(1, 3, figsize=(15, 4)) # 1*3的网格


    # A.创建绘图窗口
    fig = plt.figure(num=1, figsize=(15, 8)) # 创建窗口
    gs = fig.add_gridspec(2, 2)              # 创建2*2网格
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1], projection='polar') # 创建polar极坐标
    ax2 = fig.add_subplot(gs[1, 1])
    
    # B.动画更新
    car_path = deque(maxlen=80)
    def update(frame):
        # 移动
        x, y, yaw = 50*np.cos(frame), 50*np.sin(frame), frame+np.pi/2
        car_path.append([x, y])

        car_lidar.move_obstacle(0, -2.5*np.sin(frame), 0, -frame/70) # 移动0号障碍物
        car_lidar.move_obstacle(1, 0, 0, 0.03) # 旋转1号障碍物
        
        # 扫描
        scan_data, scan_points = car_lidar.scan(x, y, yaw, mode=1)
        scan_angles = scan_data[0]
        scan_distances = scan_data[1]
        scan_intensities = scan_data[2]
        scan_distances[scan_distances == -1] = np.inf
        
        # 0.轨迹可视化
        ax0.clear()
        ax0.set_title('Trajectory')
        ax0.set_aspect("equal")
        ax0.set_xlim(-100, 100)
        ax0.set_ylim(-100, 100)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.grid(True)

        for obstacle in car_lidar.obstacles:
            plot_shapely(obstacle, ax0) # 障碍
        
        ax0.plot(*np.array(car_path).T, 'k-.', linewidth=1.5) # 轨迹
        ax0.plot(x, y, 'bo') # 质点

        if scan_points:
            ax0.plot(*np.array(scan_points).T, 'ro', markersize=2.5) # 点云

        x1 = x + car_lidar.max_range * np.cos(yaw + np.deg2rad(car_lidar.scan_angle/2))
        y1 = y + car_lidar.max_range * np.sin(yaw + np.deg2rad(car_lidar.scan_angle/2))
        ax0.plot([x, x1], [y, y1], 'm--', linewidth=0.5) # 左扫描区域

        x2 = x + car_lidar.max_range * np.cos(yaw - np.deg2rad(car_lidar.scan_angle/2))
        y2 = y + car_lidar.max_range * np.sin(yaw - np.deg2rad(car_lidar.scan_angle/2))
        ax0.plot([x, x2], [y, y2], 'm--', linewidth=0.5) # 右扫描区域

        # 1.激光雷达可视化
        ax1.clear()
        ax1.set_title('Lidar')
        ax1.set_theta_direction(1)       # 极坐标旋转方向, -1顺时针, 默认1逆时针
        ax1.set_theta_zero_location('N') # 极坐标 0°轴指向
        ax1.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
        ax1.set_rlim([0, car_lidar.max_range])
        ax1.set_rlabel_position(0)
        # ax1.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))                   # 设置刻度位置 # note: θ轴是deg制
        # ax1.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°']) # 设置刻度名称
        ax1.scatter(scan_angles, scan_distances, s=2.5, c=scan_intensities, cmap='jet_r', vmin=0, vmax=255) # note: x轴是rad制
        ax1.plot([scan_angles[0]]*2, [0, car_lidar.max_range], 'm--', linewidth=0.5)
        ax1.plot([scan_angles[-1]]*2, [0, car_lidar.max_range], 'm--', linewidth=0.5)
    
        # 2.距离深度图像可视化
        ax2.clear()
        ax2.set_title('Depth Image')
        ax2.set_aspect('auto')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow([scan_distances[::-1]]*10, cmap='gray', vmin=0, vmax=car_lidar.max_range) # 10 * num_angle的图片
    #end update

    # C.创建动画对象
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 100), interval=5)
    ani.save('LidarSimulation.gif', writer='pillow', fps=60)
    plt.show()
    


    
from math import floor, ceil, sqrt
from operator import itemgetter
import heapq
from collections import deque
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
import  math


class Generator:
    """
    生成元のクラス。座標、色(id)、距離の値を保持する。
    
    Attributes
    ----------
    x : int
        生成元のx座標
    y : int
        生成元のy座標
    color : int
        生成元の色(id)
    d : int
        生成元の内側の半径
    """
    
    def __init__(self, x, y, color):
        """
        Parameters
        ----------
        x : int
            生成元のx座標
        y : int
            生成元のy座標
        color : int
            生成元の色(id)
        """
        
        self.x = x
        self.y = y
        self.color = color
        self.d = 0


class Voronoi:
    """
    ボロノイ図のクラス。
    
    Attributes
    ----------
    H : int
        離散平面の縦幅
    W : int
        離散平面の横幅
    plane : ndarray
        uint8型の3次元配列
        生成元を書きこむ変数
    check_overlap : ndarray
        uint32型の2次元配列
        生成元の重なり判定に使用する
    counter : int
        生成元の個数
    diagram : ndarray
        uint32型の2次元配列
        ボロノイ図
    E : set of tuple of int
        ボロノイ境界の集合
        {(x_1, y_1), ..., (x_n, y_n)}
    Em : set of tuple of float
        平均のボロノイ境界の集合
        {(x_1, y_1), ..., (x_n, y_n)}
    V : set of tuple of int
        ボロノイ頂点の集合
        {(x_1, y_1), ..., (x_n, y_n)}
    Vm : set of tuple of float
        平均のボロノイ頂点の集合
        {(x_1, y_1), ..., (x_n, y_n)}
    
    """
    
    def __init__(self, H, W):
        """
        Parameters
        ----------
        H : int
            離散平面の縦幅
        W : int
            離散平面の横幅
        """
        self.H = H
        self.W = W
        self.plane = np.zeros((H, W, 3), np.uint8)
        self.check_overlap = np.zeros((H, W), np.uint32)
        self.counter = 1  # 図形のid, 図形を描くごとにカウントアップしていく
        self.diagram = np.zeros((self.H, self.W), np.uint32)
        self.E = set()
        self.Em = set()
        self.V = set()
        self.Vm = set()
        self.gen_data = list()
                
    def counter2color(self, x):
        """
        counterの値を256進法3桁の値に変換する。
        
        Parameters
        ----------
        x : int
            counterの値
        
        Returns
        -------
        color : tuple of int
            各桁の値のタプル
        """
        
        R = x // 256**2
        x -= R * 256**2
        G = x // 256
        x -= G * 256
        B = x
        
        return (R, G, B)
    
    def add_point(self, x, y):
        """
        生成元として点を入力する。
        
        Parameters
        ----------
        x : int
            入力する点のx座標
        y : int
            入力する点のy座標
        """
        
        self.gen_data.append(["point", x, y])
        
        id = self.counter2color(self.counter)
        if 0 <= x < self.W and 0 <= y < self.H:
            self.plane[y, x] = id
        
            tmp_plane = np.zeros((self.H, self.W), np.uint8)
            tmp_plane[y, x] = 1
            self.check_overlap += tmp_plane
        
            self.counter += 1
    
    def add_points(self, pts):
        """
        生成元として複数の点を入力する。
        
        Parameters
        ----------
        pts : list of tuple
            入力する点のリスト
            [(x_1, y_1), ..., (x_n, y_n)]
        """
        
        self.gen_data.append(["points", pts])
        
        id = self.counter2color(self.counter)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        for x, y in pts:
            if 0 <= x < self.W and 0 <= y < self.H:
                self.plane[y, x] = id
                tmp_plane[y, x] = 1
        self.check_overlap += tmp_plane
        
        self.counter += 1

    def add_line(self, p1, p2):
        """
        生成元として線分を入力する。
        
        Parameters
        ----------
        p1 : tuple of int
            入力する線分の始点(x_1, y_1)
        p2 : tuple of int
            入力する線分の終点(x_2, y_2)
        """
        
        self.gen_data.append(["line", p1, p2])
        
        id = self.counter2color(self.counter)
        cv2.line(self.plane, p1, p2, id)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.line(tmp_plane, p1, p2, 1)
        self.check_overlap += tmp_plane
        
        self.counter += 1
        
    def add_lines(self, pts):
        """
        生成元として折れ線を入力する。
        
        Parameters
        ----------
        pts : list of tuple
            折れ線を構成する点のリスト
            [(x_1, y_1), ..., (x_n, y_n)]
        """
        
        pts = np.array(pts)
        self.gen_data.append(["lines", pts])
        
        id = self.counter2color(self.counter)
        cv2.polylines(self.plane, [pts], False, id)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.polylines(tmp_plane, [pts], False, 1)
        self.check_overlap += tmp_plane
        
        self.counter += 1

    def add_circle(self, p, r):
        """
        生成元として内部が塗り潰された円を入力する。
        
        Parameters
        ----------
        p : tuple of int
            円の中心座標(x, y)
        r : int
            円の半径
        """
        
        self.gen_data.append(["circle", p, r])
        
        id = self.counter2color(self.counter)
        cv2.circle(self.plane, p, r, id, thickness=-1)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.circle(tmp_plane, p, r, 1, thickness=-1)
        self.check_overlap += tmp_plane
        
        self.counter += 1

    def add_ellipse(self, p, axis, angle):
        """
        生成元として内部が塗り潰された楕円を入力する。
        
        Parameters
        ----------
        p : tuple of int
            楕円の中心座標(x, y)
        axis : tuple of int
            楕円の長半径と短半径(a, b)
        angle : float
            楕円の角度
        """
        
        box = (p, axis, angle)
        self.gen_data.append(["ellipse", box])
        
        id = self.counter2color(self.counter)
        cv2.ellipse(self.plane, box, id, thickness=-1)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.ellipse(tmp_plane, box, 1, thickness=-1)
        self.check_overlap += tmp_plane
        
        self.counter += 1

    def add_polygon(self, pts):
        """
        生成元として内部が塗り潰された多角形を入力する。
        
        Parameters
        ----------
        pts : list of tuple
            多角形を構成する頂点のリスト
            [(x_1, y_1), ..., (x_n, y_n)]
        """
        
        pts = np.array(pts)
        self.gen_data.append(["polygon", pts])
        
        id = self.counter2color(self.counter)
        cv2.fillPoly(self.plane, [pts], id)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(tmp_plane, [pts], 1)
        self.check_overlap += tmp_plane
        
        self.counter += 1
        
    def file_input(self, file_name, origin=(0, 0)):
        """
        指定されたファイルを読み込み、生成元を入力する。
        
        Parameters
        ----------
        file_name : str
            ファイルの名前、パス
        origin : tupel of int, default (0, 0)
            原点の座標
            (x, y)の形式で入力する
        """
        
        xo, yo = origin
        with open(file_name) as f:
            reader = csv.reader(f)
            for data in reader:
                if data[0].lower() == "point":
                    x, y = int(float(data[1]) - xo), int(float(data[2]) - yo)
                    self.add_point(x, y)
                    
                if data[0].lower() == "points":
                    pts = []
                    for i in range(len(data) // 2):
                        pts.append((floor(float(data[2*i + 1]) - xo), floor(float(data[2*(i + 1)]) - yo)))
                    self.add_points(pts)
                
                if data[0].lower() == "line":
                    p = []
                    for i in range(2):
                        p.append((int(float(data[2*i + 1]) - xo), int(float(data[2*(i + 1)]) - yo)))
                    self.add_line(p[0], p[1])
                    
                if data[0].lower() == "lines":
                    pts = []
                    for i in range(len(data) // 2):
                        pts.append((floor(float(data[2*i + 1]) - xo), floor(float(data[2*(i + 1)]) - yo)))
                    self.add_lines(pts)
                    
                if data[0].lower() == "circle":
                    p = (int(float(data[1]) - xo), int(float(data[2]) - yo))
                    r = max(1, int(float(data[3])))
                    self.add_circle(p, r)  
                    
                if data[0].lower() == "ellipse":
                    p = ((floor(float(data[1]) - xo), floor(float(data[2]) - yo)))
                    axis = ((max(1, floor(float(data[3]))), max(1, floor(float(data[4])))))
                    angle = floor(float(data[5]))
                    self.add_ellipse(p, axis, angle)
                    
                if data[0].lower() == "polygon":
                    pts = []
                    for i in range(len(data) // 2):
                        pts.append((floor(float(data[2*i + 1]) - xo), floor(float(data[2*(i + 1)]) - yo)))
                    self.add_polygon(pts)

    def colloect_seeds(self):
        """
        生成元を構成する点を集める。
        
        Returns
        -------
        generators : list of tuple
            生成元のリスト
            [(x_1, y_1, color_1), ..., (x_n, y_n, color_n)]
        """
        
        self.plane[self.check_overlap>1, :] = 0
        
        newp = np.zeros((self.H, self.W), np.uint32)
        self.plane = self.plane.astype(np.uint32)
        newp = self.plane[:, :, 0]*256**2 + self.plane[:, :, 1]*256 + self.plane[:, :, 2]
        
        generators = list()
        for y in range(self.H):
            for x in range(self.W):
                if newp[y, x] != 0:
                    generators.append((x, y, newp[y, x]))

        return sorted(generators, key=itemgetter(2))

    def make_dist_table(self, H, W):
        """
        順序表を作成する。
        
        Parameters
        ----------
        H : int
            離散平面の縦幅
        W : int
            離散平面の横幅
        
        Returns
        -------
        dist_table : list of tuple
            順序表
        """
        
        dist_table = list()
        for dx in range(max(H, W)):
            for dy in range(dx + 1):
                dist_table.append((dx, dy, dx**2 + dy**2))

        return sorted(dist_table, key=itemgetter(2, 1))

    def wave_front(self, generators):
        """
        波面法を行う。
        
        Parameters
        ----------
        generators : list of tuple
            生成元のリスト
            [(x_1, y_1, color_1), ..., (x_n, y_n, color_n)]
        
        Returns
        -------
        diagram : ndarray
            uint32型の2次元配列
            ボロノイ図
        """
        
        # 生成元のリストを作成
        generator_list = list()
        for x, y, color in generators:
            generator_list.append(Generator(x, y, color))

        # 距離表を作成
        dist_table = self.make_dist_table(self.H, self.W)

        # 波面法
        for i in range(len(dist_table)):
            for generator in generator_list[:]:
                j = i
                colored = False
                while True:
                    dx, dy, r = dist_table[j]
                    j += 1
                    dxs = [+dx, +dx, -dx, -dx, +dy, +dy, -dy, -dy]
                    dys = [+dy, -dy, -dy, +dy, +dx, -dx, -dx, +dx]
                    x = generator.x
                    y = generator.y
                    color = generator.color
                    if dy == 0:
                        if dx == 0:
                            '''
                            (dx, dy)
                            '''
                            self.diagram[y + dy, x + dx] = color
                            colored = True
                        else:
                            '''
                            (+dy, +dx), (-dx, +dy), (-dy, -dx), (+dx, -dy)
                            '''
                            for k in range(4):
                                if 0 <= x + dxs[(k * 5) % 8] < self.W and 0 <= y + dys[(k * 5) % 8] < self.H\
                                        and self.diagram[y + dys[(k * 5) % 8], x + dxs[(k * 5) % 8]] == 0:
                                    self.diagram[y + dys[(k * 5) % 8], x + dxs[(k * 5) % 8]] = color
                                    colored = True
                    else:
                        if dx == dy:
                            '''
                            (+dy, +dx), (-dy, +dx), (-dy, -dx), (+dy, -dx)
                            '''
                            for k in range(4):
                                if 0 <= x + dxs[k] < self.W and 0 <= y + dys[k] < self.H\
                                        and self.diagram[y + dys[k], x + dxs[k]] == 0:
                                    self.diagram[y + dys[k], x + dxs[k]] = color
                                    colored = True
                        else:
                            '''
                            (+dy, +dx), (-dy, +dx), (-dy, -dx), (+dy, -dx),
                            (+dx, +dy), (-dx, +dy), (-dx, -dy), (+dx, -dy)
                            '''
                            for k in range(8):
                                if 0 <= x + dxs[k] < self.W and 0 <= y + dys[k] < self.H\
                                        and self.diagram[y + dys[k], x + dxs[k]] == 0:
                                    self.diagram[y + dys[k], x + dxs[k]] = color
                                    colored = True

                    if j == len(dist_table) or dist_table[j][2] != r:
                        break

                if colored:
                    generator.d = r
                else:
                    e = r
                    if e > generator.d + 2*floor(sqrt(generator.d)) + 1:
                        generator_list.remove(generator)

        return self.diagram

    def voronoi_tessellation(self):
        """
        ボロノイ分割を行う。
    
        Returns
        -------
        diagram : ndarray
            uint32型の2次元配列
            ボロノイ図
        """
        
        return self.wave_front(self.colloect_seeds())

    def extractEandV(self, diagram=None):
        """
        ボロノイ図からボロノイ境界とボロノイ頂点を抽出する。
        
        Parameters
        ----------
        diagram : ndarray
            uint32型の2次元配列
            ボロノイ図
        
        Returns
        -------
        E : set of tuple of int
            ボロノイ境界の集合
            {(x_1, y_1), ..., (x_n, y_n)}
        Em : set of tuple of float
            平均のボロノイ境界の集合
            {(x_, y_1), ..., (x_n, y_n)}
        V : set of tuple of int
            ボロノイ頂点の集合
            {(x_1, y_1), ..., (x_n, y_n)}
        Vm : set of tuple of float
            平均のボロノイ頂点の集合
            {(x_1, y_1), ..., (x_n, y_n)}
        """
        
        if diagram is None:
            diagram = self.diagram
        
        for y in range(self.H - 1):
            for x in range(self.W - 1):
                d = [0, 0, 1, 1, 0]
                d_e = [0.5, -0.5, 0.5, 1.5, 0.5]
                pre_x, pre_y = x, y + 1
                cnt = 0
                v = set()
                for i in range(4):
                    dx , dy = x + d[i + 1], y + d[i]
                    if diagram[dy, dx] != diagram[pre_y, pre_x]:
                        self.E.add((pre_x, pre_y))
                        self.E.add((dx, dy))
                        self.Em.add(((x + 0.5, y + 0.5), (x + d_e[i + 1], y + d_e[i])))
                        cnt += 1
                    v.add((dx, dy))
                    pre_x, pre_y = dx, dy
                
                if 3 <= cnt:
                    self.V = self.V.union(v)
                    self.Vm.add((x + 0.5, y + 0.5))
        
        return self.E, self.Em, self.V, self.Vm
    
    def calc_area(self, diagram=None):
        """
        各ボロノイ領域の面積を計算する。
        
        Returns
        -------
        areas : list of int
            各ボロノイ領域の面積のリスト
        """
        
        if diagram is None:
            diagram = self.diagram
        
        self.areas = [np.count_nonzero(diagram == i) for i in range(self.counter)]
                
        return self.areas
    
    def search_path(self,maze=None, s=None, g=None,  neighborhood=4):
        """
        startからgoalまでのボロノイ境界を辿る最短経路をbfs（幅優先探索）で探索する。
        s=(sx, sy)とg=(gx, gy)が与えられる。
        ある境界上の点までの最短経路をsとgからbfsで探索し、見つかった境界上の点をsb=(sbx, sby), gb=(gbx, gby)とする。
        sbからsgまでのボロノイ境界上を移動する経路をbfsで探索する。
        s->sb->gb->gと繋げて最短経路が求まる。
        
        Parameters
        ----------
        maze : ndarray
            uint8型の2次元配列
            探索するボロノイ図。境界の値は1、他は0
        s : tuple of int
            スタート地点の座標
        g : tuple of int
            ゴール地点の座標
        neighborhood : int , default 8
            探索の条件、周囲のどの近傍を探索するか
            8の場合は周囲8近傍、4の場合は周囲4近傍を探索する
        
        Returns
        -------
        path : deque of tuple of int
            経路のリスト（deque）
            スタート地点からゴール地点までの経路の座標を順に格納してある
        """
        
        i = 0
        sw = 0
        for column in self.check_overlap.T:
            j = 0
            for value in column:
                if value != 0:
                    maze[j,i] = 3
                j += 1    
            i += 1
        
        if maze is None:
            maze = np.zeros((self.H, self.W), np.uint8)
            for x, y in self.E:
                maze[y, x] = 1
        
        if s is None:
            s=(0, 0)
        
        if g is None:
            g=(self.W - 1, self.H - 1)
        def calculate_distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5  
        def calculate_slope(x1, y1, x2, y2):
            if x1 == x1:  # 傾きが無限大になるケース
                return 0
            return abs((y1 - y1) / (y2 - x1))

        def find_points(n): #dxをもらい n^2 <= dx^2 + dy^2 < (n+1)^2 の範囲で調べreturn pathを返す
            points = deque()
            lower_bound = n**2
            upper_bound = (n+1)**2
    
    # x >= y で探索
            for x in range(0, int(math.sqrt(upper_bound)) + 1):
                for y in range(0, x + 1):  # y は x より大きくならない
                    dist_squared = x**2 + y**2
                    
                    # 条件を満たすかチェック
                    if lower_bound <= dist_squared < upper_bound:
                        points.append((x, y))
            
            # x^2 + y^2 の値でソート
            
            sorted_points = sorted(points, key=lambda point: (point[0]**2 + point[1]**2, -point[0]))
            points = deque(sorted_points)
            return points
        
        dx = [0, -1, 1, 0, -1, 1, -1, 1]
        dy = [-1, 0, 0, 1, -1, -1, 1, 1]    
                
        
            
        def is_valid(nx, ny, maze, visited):
            return 0 <= nx < len(maze[0]) and 0 <= ny < len(maze) and not visited[ny][nx] 

        def extend_directions(dx, dy):
            """Determine expansion directions based on the conditions for dx and dy."""
            if dx != 0 and dy == 0:
                return [(dy, dx), (-dx, dy), (-dy, -dx), (dx, -dy)]
            if dx == dy:
                return [(dy, dx), (-dy, dx), (-dy, -dx), (dy, -dx)]
            else:
                return [(dy, dx), (-dy, dx), (-dy, -dx), (dy, -dx), 
                        (dx, dy), (-dx, dy), (-dx, -dy), (dx, -dy)]

        def bfs(maze, start,neighborhood=8): #波面法を使用し,経路を生成する関数
            que = deque([(start[1], start[0])])  # キューに (x, y, dx, dy) を追加
            visited = [[False for _ in range(len(maze[0]))] for _ in range(len(maze))]  # 訪問済みのフラグ
            parent = [[None for _ in range(len(maze[0]))] for _ in range(len(maze))]  # 親の追跡用
            visited[start[1]][start[0]] = True  # 開始点を訪問済みに設定
            if maze[start[1]][start[0]] == 1:
                                
                                
                if sw == 1: path = search_path((nx,ny),start,neighborhood)
                path = search_path(start,(nx,ny),neighborhood)
                return (nx, ny),path
            dx,dy  = 1,0
            
            que_path = deque()
            
            while que:
                x, y = que.popleft()  # 現在の座標と移動方向を取り出す
                que_path = find_points(dx) #距離順にリストが返ってくる que_path = [(1,0),(1,1)]など
                # 現在の半径 (r^2) を計算
                r_squared = dx**2 + dy**2
                while que_path: #キューがなくなるまで 終わったら dx += 1 and dy = 0
                    x_r,y_r = que_path.popleft() #左端から取り出していく
                    
                    directions = extend_directions(x_r,y_r)
                    for ddx, ddy in directions: 
                        nx, ny = start[0] + ddx, start[1] + ddy
                        
                        if is_valid(nx, ny, maze,visited):
                            visited[ny][nx] = True
                            que.append((nx, ny))  # dx, dyは絶対値で拡張する
                            parent[ny][nx] = (x, y)  # 親を保存
                            
                            if maze[ny][nx] == 1:
                                
                                
                                if sw == 1: path = search_path((nx,ny),start,neighborhood)
                                path = search_path(start,(nx,ny),neighborhood)
                                return (nx, ny),path
                 
                dy = 0
                dx += 1
                
                

            print("境界点に到達しませんでした")
            return None, None

        def search_path(s,min_now,neighborhood):
            """境界点からスタート点までの最短経路を復元する."""
            path = deque([min_now])
            min_x, min_y = min_now[0], min_now[1]

            # 傾きの計算
            if (min_now[0] - s[0]) != 0:
                a = (abs(min_now[1] - s[1])) / (abs(min_now[0] - s[0]))
            else:
                a = 0

            # ゴールが右か左かを判定し、探索方向を決定
            if a == 0:
                
                return straight_search(path, s, min_now, neighborhood)
            
            elif s[0] < min_now[0]:
               
                return slope_search(path, s, min_now, a, neighborhood, katamuki_direction="down")
            
            elif s[0] > min_now[0]:
                
                return slope_search(path, s, min_now, a, neighborhood, katamuki_direction="up")

        def straight_search(path, s, min_now, neighborhood):
            min_x, min_y = min_now
            main = 0
            dis = float('inf')

            while True:
                nowx, nowy = path[0]
                if (nowx, nowy) == s:
                    break
                for i in range(neighborhood):
                    nx = nowx + dx[i]
                    ny = nowy + dy[i]
                    if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H or self.check_overlap[ny, nx] != 0:
                        continue

                    new_dis = calculate_distance(s, (nx, ny))
                    if new_dis < dis:
                        dis = new_dis
                        min_x, min_y = nx, ny
                
                path.appendleft((min_x, min_y))
                main += calculate_distance(path[0], path[1])

            path.pop()
            return min_now, path
        
        def slope_search(path, s, min_now, a, neighborhood, katamuki_direction):
            min_x, min_y = min_now
            main = 0
            
            if katamuki_direction == "up":
                #goalor start境界上の点よりも右下の時or左した
                
                if s[0] > min_now[0]:
                    while True:
                        nowx ,nowy = path[0] 
                        katamuki = 999
                        #print(nowx,nowy)
                        if (nowx, nowy) == s:
                            break;
                        for i in range(neighborhood):
                            nx = nowx + dx[i]
                            ny = nowy + dy[i]
                            if nx < 0 or self.W <= nx or ny < 0 or self.H <= ny or nowx > nx or nx > s[0]:
                                continue; 
                            
                            if (nx,ny) == s: #goal地点に到着
                                min_x,min_y = nx,ny
                                #path.appendleft((min_x,min_y))
                                break
                            
                            if (nx == s[0] and ny != s[1]): #nxは同じだけどgoalではない時
                                continue;
                            
                            b = abs(s[1]- ny) / abs(s[0] - nx) #傾き計算
                            katamuki_tmp = abs(a - b) #差を計算する
                            if katamuki >= katamuki_tmp: #sw==0の時は点s-s'の処理 sw==1の時は点g-g'の処理
                                if s[1] <= nowy and nowy >= ny: #nyが小さければ更新 s,gを中心とすると境界上の点は右下
                                    katamuki = katamuki_tmp
                                    min_x, min_y = nx,ny
                                elif s[1] >= nowy and nowy <= ny:#s,gを中心とすると境界上の点は右上
                                    katamuki = katamuki_tmp
                                    min_x, min_y = nx,ny
                                    #print(path)
                        path.appendleft((min_x,min_y))
                    path.pop()
                    return min_now,path
            elif katamuki_direction == "down":
                if s[0] < min_now[0]:
                    while True:
                        nowx, nowy = path[0]
                        katamuki = 999

                        if (nowx, nowy) == s:
                            
                            break

                        for i in range(neighborhood):
                            nx = nowx + dx[i]
                            ny = nowy + dy[i]
                            if nx < 0 or self.W <= nx or ny < 0 or self.H <= ny or nowx < nx or nx < s[0]:
                                continue;
                            if (s[0] == nx and s[1] == ny):#たどり着いた時
                                min_x,min_y = nx,ny
                                break
                            
                            if (nx == s[0] and ny != s[1]): #nxは同じだけどgoalではない時
                                continue;
                            b = abs(s[1]- ny) / abs(s[0] - nx) #傾き計算
                            katamuki_tmp = abs(a - b)
                            if katamuki >= katamuki_tmp:
                                if s[1] <= nowy and nowy >= ny: #nyが小さければ更新 (sの場合) 左下の時 (gの場合) 左下の時
                                    katamuki = katamuki_tmp
                                    min_x, min_y = nx,ny
                                if s[1] >= nowy and nowy <= ny: #nyが大きければ更新　(s,gの場合) 左上の時 
                                    katamuki = katamuki_tmp
                                    min_x,min_y = nx,ny  
                        path.appendleft((min_x,min_y))
                    path.pop()
                    return min_now, path 
            else:
                print('見つからない')
                
                    
                    
                     
            
        sb, path_sb = bfs(maze, s, neighborhood)
        path_sb = path_sb[1]
        
        for x,y in path_sb:
            if (0 <= y <maze.shape[0] and 0 <= x < maze.shape[1]):
                maze[y,x] = 4
            else:
                print(f"Invalid coordinate: (y={y}, x={x})")
              
            
        
        sw = 1
        
        gb, path_gb = bfs(maze, g, neighborhood)
        path_gb = path_gb[1]
        #for x,y in path_gb:
         #   maze[y,x] = 4
        #print(main)
        for x,y in path_gb:
            if (0 <= y <maze.shape[0] and 0 <= x < maze.shape[1]):
                maze[y,x] = 4
            else:
                print(f"Invalid coordinate: (y={y}, x={x})")
        que = deque([sb])
        parent = [[-1 for i in range(self.W)] for j in range(self.H)]
        dx = [0, -1, 1, 0, -1, 1, -1, 1]
        dy = [-1, 0, 0, 1, -1, -1, 1, 1]
        while que:
            now = que.popleft()
            
            if now == gb:
                break
            
            for i in range(neighborhood):
                nx = now[0] + dx[i]
                ny = now[1] + dy[i]
                if nx < 0 or self.W <= nx or ny < 0 or self.H <= ny:
                    continue;
                if maze[ny, nx] == 0:  # 0 is boundary
                    continue;
                if parent[ny][nx] != -1:
                    continue;
                que.append((nx, ny))
                parent[ny][nx] = now
        
        path_b = deque([gb])
        while True:
            nowx, nowy = path_b[0]
            if (nowx, nowy) == sb:
                break;
            path_b.appendleft(parent[nowy][nowx])
            
        path = deque(path_sb) + deque(path_b) + deque(reversed(path_gb))
        for x,y in path_b:
            maze[y,x] = 4

        return path
    
    def plot_generators(self, RGB=(0, 0, 0), A=100):
        img = np.zeros((self.H, self.W, 4), np.uint8)
        for y in range(self.H):
            for x in range(self.W):
                img[y, x, :3] = RGB
                img[y, x, 3] = self.check_overlap[y, x] * A
        
        plt.imshow(img)

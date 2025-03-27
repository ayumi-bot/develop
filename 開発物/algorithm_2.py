from math import floor, ceil, sqrt
from operator import itemgetter
import heapq
from collections import deque
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
import  math
import time

class Generator:
    """
    生成源のクラス。座標、色(id)、距離の値を保持する。
    
    Attributes
    ----------
    x : int
        生成源のx座標
    y : int
        生成源のy座標
    color : int
        生成源の色(id)
    d : int
        生成源の内側の半径
    """
    
    def __init__(self, x, y, color):
        """
        Parameters
        ----------
        x : int
            生成源のx座標
        y : int
            生成源のy座標
        color : int
            生成源の色(id)
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
        生成源を書きこむ変数
    is_placed : ndarray
        uint8型の2次元配列
        生成源があるかを判断する
        まとめありのときに使う
    check_overlap : ndarray
        uint32型の2次元配列
        生成源の重なり判定に使用する
    counter : int
        生成源の個数
        図形のid, 図形を描くごとにカウントアップしていく
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
    gen_data : list
        生成源のデータ
    start : tuple of int
        経路探索のスタート地点座標 (Yabe)
    goal : tuple of int
        経路探索のゴール地点座標 (Yabe)
    merging : int
        1のとき、生成源をまとめる (Yabe)
    straight : int
        1のとき、ボロノイ図を直線に置き換える (Yabe)
    """
    
    def __init__(self):
        self.H = int()
        self.W = int()
        self.counter = 1
        self.E = set()
        self.Em = set()
        self.V = set()
        self.Vm = set()
        self.gen_data = list()
        self.start = tuple()
        self.goal = tuple()
        self.merging = 0
        self.straight = 0
                
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
        生成源として点を入力する。
        
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
            tmp_plane = np.zeros((self.H, self.W), np.uint8)
            tmp_plane[y, x] = 1

            # 重なっている生成源をまとめる (Yabe)
            if self.merging == 1:
                id = self.merge_generators("point", [x, y], id, tmp_plane)

            self.plane[y, x] = id
            self.check_overlap += tmp_plane
        
            self.counter += 1
    
    def add_points(self, pts):
        """
        生成源として複数の点を入力する。
        
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
                tmp_plane[y, x] = 1

        # 重なっている生成源をまとめる (Yabe)
        if self.merging == 1:
            id = self.merge_generators("points", pts, id, tmp_plane)

        for x, y in pts:
            if 0 <= x < self.W and 0 <= y < self.H:
                self.plane[y, x] = id
        self.check_overlap += tmp_plane
        
        self.counter += 1

    def add_line(self, p1, p2):
        """
        生成源として線分を入力する。
        
        Parameters
        ----------
        p1 : tuple of int
            入力する線分の始点(x_1, y_1)
        p2 : tuple of int
            入力する線分の終点(x_2, y_2)
        """
        
        self.gen_data.append(["line", p1, p2])
        
        id = self.counter2color(self.counter)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.line(tmp_plane, p1, p2, 1)

        # 重なっている生成源をまとめる (Yabe)
        if self.merging == 1:
            id = self.merge_generators("line", [p1, p2], id, tmp_plane)

        cv2.line(self.plane, p1, p2, id)
        self.check_overlap += tmp_plane
        
        self.counter += 1
        
    def add_lines(self, pts):
        """
        生成源として折れ線を入力する。
        
        Parameters
        ----------
        pts : list of tuple
            折れ線を構成する点のリスト
            [(x_1, y_1), ..., (x_n, y_n)]
        """
        
        pts = np.array(pts)
        self.gen_data.append(["lines", pts])
        
        id = self.counter2color(self.counter)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.polylines(tmp_plane, [pts], False, 1)

        # 重なっている生成源をまとめる (Yabe)
        if self.merging == 1:
            id = self.merge_generators("lines", pts, id, tmp_plane)

        cv2.polylines(self.plane, [pts], False, id)
        self.check_overlap += tmp_plane
        
        self.counter += 1
        
    def add_circle(self, p, r):
        """
        生成源として内部が塗り潰された円を入力する。
        
        Parameters
        ----------
        p : tuple of int
            円の中心座標(x, y)
        r : int
            円の半径
        """

        #r *= 2 # クレーターの半径を倍として扱う(s, gがつぶれないよう注意) (Yabe)

        self.gen_data.append(["circle", p, r])

        id = self.counter2color(self.counter)

        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.circle(tmp_plane, p, r, 1, thickness=-1)

        # 重なっている生成源をまとめる (Yabe)
        if self.merging == 1:
            id = self.merge_generators("circle", [p, r], id, tmp_plane)
        
        cv2.circle(self.plane, p, r, id, thickness=-1)
        self.check_overlap += tmp_plane

        self.counter += 1

    def add_ellipse(self, p, axis, angle):
        """
        生成源として内部が塗り潰された楕円を入力する。
        
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
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.ellipse(tmp_plane, box, 1, thickness=-1)

        # 重なっている生成源をまとめる (Yabe)
        if self.merging == 1:
            id = self.merge_generators("ellipse", box, id, tmp_plane)

        cv2.ellipse(self.plane, box, id, thickness=-1)
        self.check_overlap += tmp_plane
        
        self.counter += 1

    def add_polygon(self, pts):
        """
        生成源として内部が塗り潰された多角形を入力する。
        
        Parameters
        ----------
        pts : list of tuple
            多角形を構成する頂点のリスト
            [(x_1, y_1), ..., (x_n, y_n)]
        """
        
        pts = np.array(pts)
        self.gen_data.append(["polygon", pts])
        
        id = self.counter2color(self.counter)
        
        tmp_plane = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(tmp_plane, [pts], 1)

        # 重なっている生成源をまとめる (Yabe)
        if self.merging == 1:
            id = self.merge_generators("polygon", pts, id, tmp_plane)

        cv2.fillPoly(self.plane, [pts], id)
        self.check_overlap += tmp_plane
        
        self.counter += 1
        
    def merge_generators(self, shape, datalist, id, tmp_plane):
        """
        他の生成源と重なっている場合、同じidでまとめる。 (Yabe)

        Parameters
        ----------
        kind : char
            図形の名称
        datalist : list
            各図形を構成する要素のリスト(入力データのリスト)
        id : tuple of int
            生成源のid

        Returns
        ----------
        id : tuple of int
            生成源のid
        """

        if np.any(self.is_placed - tmp_plane == 2):
            already_overlapping = 0
            for y in range(self.H):
                for x in range(self.W):
                    if tmp_plane[y, x] and self.check_overlap[y, x]: # 重なる生成源がある場合
                        R, G, B = self.plane[y, x].tolist()
                        idn = (R, G, B)
                        if not np.all(id == idn):
                            if already_overlapping == 0: # 重なったもののidにまとめる
                                id = idn
                                already_overlapping = 1
                            else: # 2つ以上と重なるとき、自分のidにまとめる
                                for t in range(self.H):
                                    for s in range(self.W):
                                        if self.check_overlap[t, s] and np.all(self.plane[t, s] == idn):
                                            self.plane[t, s] = id

        if shape == "point":
            self.is_placed[datalist[1], datalist[0]] = 3
        elif shape == "points":
            for x, y in datalist:
                if 0 <= x < self.W and 0 <= y < self.H:
                    self.is_placed[y, x] = 3
        elif shape == "line":
            cv2.line(self.is_placed, datalist[0], datalist[1], 3)
        elif shape == "lines":
            cv2.polylines(self.is_placed, [datalist], False, 3)
        elif shape == "circle":
            cv2.circle(self.is_placed, datalist[0], datalist[1], 3, thickness=-1)
        elif shape == "ellipse":
            cv2.ellipse(self.is_placed, datalist, 3, thickness=-1)
        elif shape == "polygon":
            cv2.fillPoly(self.is_placed, [datalist], 3)

        return id
    
    def file_input(self, file_name):
        """
        生成源を入力する
        入力ファイルの書き方はREADME.txt参照(Yabe)

        Parameters
        ----------
        file_name : str
            ファイルの名前、パス
        """
        
        with open(file_name) as f:
            reader = csv.reader(f)
            cnt = 0
            for data in reader:
                if cnt == 0: # 出力スクリーンサイズ倍率 (Yabe)
                    magnification = int(data[0])
                    print('magnification :', magnification)

                elif cnt == 1: # スクリーンサイズ (Yabe)
                    self.W, self.H = int(data[0]) * magnification, int(data[1]) * magnification
                    print('size : ', self.W, 'x', self.H)
                    self.plane = np.zeros((self.H, self.W, 3), np.uint8)
                    self.is_placed = np.zeros((self.H, self.W), np.uint8)
                    self.check_overlap = np.zeros((self.H, self.W), np.uint32)
                    self.diagram = np.zeros((self.H, self.W), np.uint32)

                elif cnt == 2: # 原点座標 (Yabe)
                    xo, yo = float(data[0]) * magnification, float(data[1]) * magnification
                    origin = tuple((xo, yo))
                    print('origin :', origin)

                elif cnt == 3: # スタート (Yabe)
                    self.start = tuple((int(data[0]) * magnification, int(data[1]) * magnification))
                    print('strart :', self.start)

                elif cnt == 4: # ゴール (Yabe)
                    self.goal = tuple((int(data[0]) * magnification, int(data[1]) * magnification))
                    print('goal :', self.goal)

                elif cnt == 5: # 0以外のとき生成源をまとめる (Yabe)
                    self.merging = int(data[0])

                elif cnt == 6: # 0のとき曲線、0以外のとき直線 (Yabe)
                    self.straight = int(data[0])

                else: # 生成源データ
                    if data[0].lower() == "point":
                        x, y = int(float(data[1]) * magnification - xo), int(float(data[2]) * magnification - yo)
                        self.add_point(x, y)
                    
                    if data[0].lower() == "points":
                        pts = []
                        for i in range(len(data) // 2):
                            pts.append((floor(float(data[2*i + 1]) * magnification - xo), floor(float(data[2*(i + 1)]) * magnification - yo)))
                        self.add_points(pts)
                
                    if data[0].lower() == "line":
                        p = []
                        for i in range(2):
                            p.append((int(float(data[2*i + 1]) * magnification - xo), int(float(data[2*(i + 1)]) * magnification - yo)))
                        self.add_line(p[0], p[1])
                    
                    if data[0].lower() == "lines":
                        pts = []
                        for i in range(len(data) // 2):
                            pts.append((floor(float(data[2*i + 1]) * magnification - xo), floor(float(data[2*(i + 1)]) * magnification - yo)))
                        self.add_lines(pts)
                    
                    if data[0].lower() == "circle":
                        # p = (int(float(data[1]) * magnification - xo), int(float(data[2]) * magnification - yo)) # 元のコード
                        p = (int((float(data[1])*1.5) * magnification - xo), self.H - int((float(data[2])*1.5) * magnification - yo)) # y座標を相対的に反転したもの。Yanatoriの入力データは上下逆になっている (Yabe)
                        r = max(1, int(float(data[3])*1.5)) * magnification
                        self.add_circle(p, r)
                    
                    if data[0].lower() == "ellipse":
                        p = ((floor(float(data[1]) * magnification - xo), floor(float(data[2]) * magnification - yo)))
                        axis = ((max(1, floor(float(data[3]) * magnification)), max(1, floor(float(data[4]) * magnification))))
                        angle = floor(float(data[5]) * magnification)
                        self.add_ellipse(p, axis, angle)
                    
                    if data[0].lower() == "polygon":
                        pts = []
                        for i in range(len(data) // 2):
                            pts.append((floor(float(data[2*i + 1]) * magnification - xo), floor(float(data[2*(i + 1)]) * magnification - yo)))
                        self.add_polygon(pts)

                cnt += 1

    def read_file(self, file_name):
        """
        指定されたファイルを読み込み、実行する。経路探索の結果がpltで出力される

        Parameters
        ----------
        file_name : str
            ファイルの名前、パス
            ファイルの書き方はREADME.txt参照

        Returns
        ----------
        E : set of tuple of int
            境界の座標の集合
        V : set of tuple of int
            頂点の座標の集合
        path : deque of tuple of int
            経路のリスト（deque）
            スタート地点からゴール地点までの経路の座標を順に格納してある
        ex_img : uint8型の3次元配列
            境界や経路が描かれている
        """
        
        # 時間計測開始 (Yabe)
        # time_start = time.perf_counter()

        print('inputfile : ', file_name)

        self.file_input(file_name)

        diagram = self.voronoi_tessellation()
        E, _, V, _ = self.extractEandV(diagram)

        # 経路探索をする、sからgまでに辿るピクセルの座標のリストを返す
        path = self.search_path(s=self.start, g=self.goal)
        # path = None # 経路探索しない

        """
        # 生成源入力から経路探索終了までの時間計測終了 (Yabe)
        time_end = time.perf_counter()
        tim = time_end - time_start
        print('time : ', tim)
        """

        ex_img = np.zeros((self.H, self.W, 4), np.uint8)

        # 境界に色をつける
        for x, y, in self.E:
            ex_img[y, x, :] = (0, 0, 255, 255)

        """
        # 頂点に色をつける (Yabe)
        for x, y in self.V:
           cv2.circle(ex_img, (x, y), 1, (255, 0, 0, 255))
        """
        
        if path is not None:
            # 経路のピクセル数を出力
            print('path len : ', len(path) - 1)

            for i in range(1, len(path)):
                cv2.line(ex_img, path[i - 1], path[i], (0, 255, 0, 255), thickness=2)
        
        plt.figure()
        self.plot_generators() # 生成源をpltに載せる
        plt.imshow(ex_img)
        plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False) # 軸目盛を消す (Yabe)
        plt.show()

        return E, V, path, ex_img

    def colloect_seeds(self):
        """
        生成源を構成する点を集める。
        
        Returns
        -------
        generators : list of tuple
            生成源のリスト
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

    def wave_front(self):
        """
        波面法を行う。
        実際には、波面法ではなく、正方形を拡大している。(Yabe)

        Parameters
        ----------
        generators : list of tuple
            生成源のリスト
            [(x_1, y_1, color_1), ..., (x_n, y_n, color_n)]
        
        Returns
        -------
        diagram : ndarray
            uint32型の2次元配列
            ボロノイ図
        """

        # 時間計測開始 (Yabe)
        # time_start = time.perf_counter()

        # 生成源のリストを作成
        generator_list = self.get_generator_list()
 
        # 距離表を作成
        dist_table = self.make_dist_table(self.H, self.W)
        self.seiseigen = generator_list.copy()
        # 波面法
        len_dist_table = len(dist_table)
        diagram = self.diagram.copy()
        W, H = self.W, self.H
        for i in range(len_dist_table):
            for generator in generator_list[:]:
                j = i
                colored = False

                while True:
                    dx, dy, r = dist_table[j]
                    dxs = [+dx, +dx, -dx, -dx, +dy, +dy, -dy, -dy]
                    dys = [+dy, -dy, -dy, +dy, +dx, -dx, -dx, +dx]

                    if dy == 0:
                        if dx == 0:
                            '''
                            (dx, dy)
                            '''
                            diagram[generator.y + dy, generator.x + dx] = generator.color
                            colored = True
                        else:
                            '''
                            (+dy, +dx), (-dx, +dy), (-dy, -dx), (+dx, -dy)
                            '''
                            for k in range(4):
                                kn = (k * 5) % 8
                                xn = generator.x + dxs[kn]
                                yn = generator.y + dys[kn]
                                if 0 <= xn < W and 0 <= yn < H\
                                        and diagram[yn, xn] == 0:
                                    diagram[yn, xn] = generator.color
                                    colored = True
                    else:
                        if dx == dy:
                            '''
                            (+dy, +dx), (-dy, +dx), (-dy, -dx), (+dy, -dx)
                            '''
                            for k in range(4):
                                xn = generator.x + dxs[k]
                                yn = generator.y + dys[k]
                                if 0 <= xn < W and 0 <= yn < H\
                                        and diagram[yn, xn] == 0:
                                    diagram[yn, xn] = generator.color
                                    colored = True
                        else:
                            '''
                            (+dy, +dx), (-dy, +dx), (-dy, -dx), (+dy, -dx),
                            (+dx, +dy), (-dx, +dy), (-dx, -dy), (+dx, -dy)
                            '''
                            for k in range(8):
                                xn = generator.x + dxs[k]
                                yn = generator.y + dys[k]
                                if 0 <= xn < W and 0 <= yn < H\
                                        and diagram[yn, xn] == 0:
                                    diagram[yn, xn] = generator.color
                                    colored = True

                    j += 1
                    if j == len_dist_table or dist_table[j][2] != r: 
                        break;

                if colored:
                    generator.d = r
                else:
                    e = r
                    if e > generator.d + 2*floor(sqrt(generator.d)) + 1:
                        generator_list.remove(generator)

        self.diagram = diagram.copy()

        """
        # 時間計測終了 (Yabe)
        time_end = time.perf_counter()
        tim = time_end - time_start
        print('time for wave front : ', tim)
        """

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
        
        return self.wave_front()

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
            平均のボロノイ境界の集合(理想境界)
            {(x_, y_1), ..., (x_n, y_n)}
        V : set of tuple of int
            ボロノイ頂点の集合
            {(x_1, y_1), ..., (x_n, y_n)}
        Vm : set of tuple of float
            平均のボロノイ頂点の集合(理想頂点)
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

        # ボロノイ図を直線に置き換える (Yabe)
        if self.straight == 1:
            self.E.clear
            self.E = self.edge_straight()

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
    
    def plot_generators(self, RGB=(0, 0, 0), A=100):
        """
        生成源をpltに載せる
        """
        
        img = np.zeros((self.H, self.W, 4), np.uint8)
        for y in range(self.H):
            for x in range(self.W):
                img[y, x, :3] = RGB
                # 重なっている部分を暗くする
                img[y, x, 3] = self.check_overlap[y, x] * A 
                # 重なっている部分を暗くしない
                if self.check_overlap[y, x]: 
                    img[y, x, 3] = A

        plt.imshow(img)

    def get_generator_list(self):
        """
        生成源のリストを返す

        Returns
        -------
        generator_list : list of Generator
            生成源のリスト
        """
        
        generators = self.colloect_seeds()
        generator_list = list()
        for x, y, color in generators:
            generator_list.append(Generator(x, y, color))

        return generator_list



    
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
        time_start = time.perf_counter()
        def find_point(start,diagram=None):
            if diagram  is None:
                diagram = self.diagram
            que_path = deque()
            x,y = start
            region_index = diagram[y,x] 
            maze_1 = np.argwhere(maze==1)
            for j,i in maze_1:
                
                if diagram[j,i] == region_index: #右回り？　maze == 1 かつ regon_indexが一致かつ 右回りになるように
                    que_path.append((i,j)) 
                
            return que_path
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
        
        s = [coord - 1 if coord == self.W or coord == self.H else coord for coord in s]
        g = [coord - 1 if coord == self.W or coord == self.H else coord for coord in g]
        print(s,g)
        s = (s[0],s[1])
        g = (g[0],g[1])
            
        # sまたはgが画面外にある時、経路探索しない (Yabe)
        if s[0] < 0 or self.W < s[0] or \
            s[1] < 0 or self.H < s[1] or \
                g[0] < 0 or self.W < g[0] or \
                    g[1] < 0 or self.H < g[1]:
                    print('start or goal is out of screen')
                    return None
        # sが生成源上にある時、経路探索しない (Yabe)
        if self.check_overlap[s[1], s[0]] != 0:
            print('Start point is on generator')
            return None
        # gが生成源上にある時、経路探索しない (Yabe)
        if self.check_overlap[g[1], g[0]] != 0:
            print('Goal point is on generator')
            return None
        
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

        def bfs(maze, start,goal,neighborhood=8): #波面法を使用し,経路を生成する関数
            que = deque([(start[1], start[0])])  # キューに (x, y, dx, dy) を追加
            visited = [[False for _ in range(len(maze[0]))] for _ in range(len(maze))]  # 訪問済みのフラグ
            parent = [[None for _ in range(len(maze[0]))] for _ in range(len(maze))]  # 親の追跡用
            visited[start[1]][start[0]] = True  # 開始点を訪問済みに設定
            main = 0
            dx,dy  = 1,0
            
            que_path = deque()
            """"
            
            for i in range(len(maze)):
                for j in range(len(maze[0])):
                    if maze[j][i] == 1:
                        que_path.append((i,j))
            """
            
            que_path = find_point(start)
                        
        
            
            sorted_points = sorted(que_path, key=lambda p: math.sqrt((p[0] - goal[0]) ** 2 + (p[1] - goal[1]) **2))
            for sort_point in sorted_points:
                x, y = sort_point[0],sort_point[1] # 現在の座標を取り出す
                min_now,path,main = search_path(start,(x,y),neighborhood)
                #print(path)
                i = 0
                if path is None:
                    
                    i += 1
                    continue
                else:
                    print('pathが見つかりました')
                    break
                
            return (x,y),path,main

            print("境界点に到達しませんでした")
            return None, None,None

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
            xy_list = [(gen.x,gen.y) for gen in self.seiseigen] #生成源のリスト
            while True:
                nowx, nowy = path[0]
                if (nowx, nowy) == s:
                    break
                for i in range(neighborhood):
                    nx = nowx + dx[i]
                    ny = nowy + dy[i]
                    if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
                        continue

                    new_dis = calculate_distance(s, (nx, ny))
                    if new_dis < dis:
                        dis = new_dis
                        min_x, min_y = nx, ny
                
                #if self.check_overlap[ny, nx] != 0:
                if (min_x,min_y) in xy_list:
                    
                    return None,None,None
                
                
                path.appendleft((min_x, min_y))
                main += calculate_distance(path[0], path[1])

            path.pop()
            return min_now, path,main
        
        def slope_search(path, s, min_now, a, neighborhood, katamuki_direction):
            min_x, min_y = min_now
            main = 0
            xy_list = [(gen.x,gen.y) for gen in self.seiseigen] #生成源のリスト　
            
            #ここからは生成源にぶつかると終了する
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
                            if (s[0] == nx and s[1] == ny  ):#たどり着いた時
                                min_x,min_y = nx,ny
                                break
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
                            
                            
                        
                        if (min_x,min_y) in xy_list:
                            
                            return None,None,None
                            
                        path.appendleft((min_x,min_y))
                        main += calculate_distance(path[0], path[1])
                    path.pop()
                    return min_now,path,main
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
                            if (s[0] == nx and s[1] == ny  ):#たどり着いた時
                                min_x,min_y = nx,ny
                                break
                            if nx < 0 or self.W <= nx or ny < 0 or self.H <= ny or nowx < nx or nx < s[0]:
                                continue;
                            #if (s[0] == nx and s[1] == ny  ):#たどり着いた時
                               # min_x,min_y = nx,ny
                               # break
                            
                            
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
                        
                        if (min_x,min_y) in xy_list:
                            
                            return None,None,None
                        

                        path.appendleft((min_x,min_y))
                        main += calculate_distance(path[0], path[1])
                    path.pop()
                    return min_now, path ,main
            else:
                print('見つからない')
                
                    
                    
                     
            
        sb, path_sb,main_s = bfs(maze, s, g,neighborhood)
        #path_sb = path_sb[1]
        
        for x,y in path_sb:
            if (0 <= y <maze.shape[0] and 0 <= x < maze.shape[1]):
                maze[y,x] = 2
            else:
                print(f"Invalid coordinate: (y={y}, x={x})")
              
            
        
        sw = 1
        print('start終了')
        gb, path_gb ,main_g= bfs(maze, g, sb, neighborhood)
        #path_gb = path_gb[1]
        #for x,y in path_gb:
         #   maze[y,x] = 4
        #print(main)
        for x,y in path_gb:
            if (0 <= y <maze.shape[0] and 0 <= x < maze.shape[1]):
                maze[y,x] = 2
            else:
                print(f"Invalid coordinate: (y={y}, x={x})")
        
        print('goal終了')
        que = deque([sb])
        parent = [[-1 for i in range(self.W)] for j in range(self.H)]
        dx = [0, -1, 1, 0, -1, 1, -1, 1]
        dy = [-1, 0, 0, 1, -1, -1, 1, 1]
        main = 0
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
            try:
                nowx, nowy = path_b[0]

            except: # 経路がみつけられないとき。スタートorゴールの最寄りが、画面端と生成源に挟まれていると、経路探索できない (Yabe)
                print('Not found the route.')
                return None

            if (nowx, nowy) == sb:
                break;

            path_b.appendleft(parent[nowy][nowx])
            
            main += calculate_distance(path_b[0], path_b[1])
        main_f = main + main_s + main_g
        print(main_f)
        path = path_sb + path_b + deque(reversed(path_gb))
        time_end = time.perf_counter()
        tim = time_end - time_start
        print('time : ', tim)
        if path is not None:
            path_length = len(path)
            print('Path length:', path_length) 
        for x,y in path_b:
            maze[y,x] = 2
        
        ex_img = np.zeros((self.H, self.W, 4), np.uint8)
        for x, y, in self.E: #黒のボロノイ境界にする
            ex_img[y, x, :] = (0, 0, 0, 255) #(青,緑,赤,透明度) 

        for i in range(1, len(path)):
            start = path[i - 1]
            end = path[i]
            if maze[start[1], start[0]] == 2:
                thickness = 3  # 境界上なら太め
            
            
            else:
                #thickness = 1  # 通路なら細め
                continue
        
            cv2.line(ex_img, start, end, (0, 0, 255, 255), thickness=thickness) #経路を赤にする
        cv2.circle(ex_img,s,radius=5,color=(0,0,255,255),thickness=-1)
        cv2.circle(ex_img,g,radius=5,color=(0,0,255,255),thickness=-1)
        ex_img_rgb = cv2.cvtColor(ex_img, cv2.COLOR_BGRA2RGBA)
        plt.figure()
        self.plot_generators()
        plt.imshow(ex_img_rgb)
        
        plt.show()

        
            
        

        return path
    
    

import math
import random

from sumo.simpleStruct import RoadPosition, Direction


class Road:
    road_id = "E0"
    direction = "forward"
    lane_type = "straight"
    lane_num = 3
    lane_width = 3.6
    speed_limit = 35
    R = 494.6

    @staticmethod
    def in_road(x, y):
        if -750 <= x <= 750 and (500 - Road.lane_num * Road.lane_width) <= y <= 500:
            return RoadPosition.UP_OF_ROAD
        elif -750 <= x <= 750 and -500 <= y <= -(500 - Road.lane_num * Road.lane_width):
            return RoadPosition.DOWN_OF_ROAD
        elif x > 750:
            r = math.sqrt((x - 750) ** 2 + y ** 2)
            if (500 - Road.lane_num * Road.lane_width) <= r <= 500:
                return RoadPosition.RIGHT_OF_ROAD
            else:
                return RoadPosition.ERROR_ARISE
        elif x < -750:
            r = math.sqrt((x + 750) ** 2 + y ** 2)
            if (500 - Road.lane_num * Road.lane_width) <= r <= 500:
                return RoadPosition.LEFT_OF_ROAD
            else:
                return RoadPosition.ERROR_ARISE
        else:
            return RoadPosition.ERROR_ARISE

    @staticmethod
    def get_direction(x ,y):
        dv = Direction(0, 0)
        pos_index = Road.in_road(x, y)

        if pos_index.name == "ERROR_ARISE":
            dv.x = 0
            dv.y = 0
            return dv
        elif pos_index.name == "UP_OF_ROAD":
            dv.x = 1
            dv.y = 0
            return dv
        elif pos_index.name == "DOWN_OF_ROAD":
            dv.x = -1
            dv.y = 0
            return dv
        elif pos_index.name == "RIGHT_OF_ROAD":
            length = math.sqrt((x - 750.0) ** 2 + y ** 2)
            dv.x = y / length
            dv.y = -(x - 750.0) / length
            return dv
        elif pos_index.name == "LEFT_OF_ROAD":
            length = math.sqrt((x + 750.0) ** 2 + y ** 2)
            dv.x = y / length
            dv.y = -(x + 750.0) / length
            return dv
        else:
            return dv

    @staticmethod
    def front_or_behind(x1, y1, x2, y2):
        dv1 = Road.get_direction(x1, y1)
        dv2 = Direction(x2 - x1, y2 - y1)

        ab = dv1.x * dv2.x + dv1.y * dv2.y
        a = math.sqrt(dv1.x ** 2 + dv1.y ** 2)
        b = math.sqrt(dv2.x ** 2 + dv2.y ** 2)

        if a * b == 0:
            return 1

        cosa = ab / (a * b)
        if cosa != 0:
            return abs(cosa) / cosa
        else:
            return 1

    @staticmethod
    def relation_distance(x1, y1, x2, y2):
        signal = Road.front_or_behind(x1, y1, x2, y2)
        pos1 = Road.in_road(x1, y1)
        pos2 = Road.in_road(x2, y2)

        if pos1 == RoadPosition.ERROR_ARISE or pos2 == RoadPosition.ERROR_ARISE:
            return 0

        if 750 >= x1 >= -750:
            if y1 > 0:
                if x2 > 750:
                    length = math.sqrt((x2 - 750.0) ** 2 + y2 ** 2)
                    dis1 = 750.0 - x1
                    dis2 = Road.R * math.acos(1 - ((x2 - 750.0) ** 2 + (y2 - length) ** 2) / (2 * length ** 2))
                    return dis1 + dis2
                elif x2 < -750:
                    length = math.sqrt((x2 + 750.0) ** 2 + y2 ** 2)
                    dis1 = x1 + 750
                    dis2 = Road.R * math.acos(1 - ((x2 + 750.0) ** 2 + (y2 - length) ** 2) / (2 * length ** 2))
                    return -(dis1 + dis2)
                else:
                    if x2 > x1:
                        distance = math.sqrt((x2 - x1) ** 2)
                        return distance
                    else:
                        distance = -math.sqrt((x2 - x1) ** 2)
                        return distance
            else:
                if x2 > 750:
                    length = math.sqrt((x2 - 750.0) ** 2 + y2 ** 2)
                    dis1 = 750.0 - x1
                    dis2 = Road.R * math.acos(1 - ((x2 - 750.0) ** 2 + (y2 + length) ** 2) / (2 * length ** 2))
                    return -(dis1 + dis2)
                elif x2 < -750:
                    length = math.sqrt((x2 + 750.0) ** 2 + y2 ** 2)
                    dis1 = x1 + 750
                    dis2 = Road.R * math.acos(1 - ((x2 + 750.0) ** 2 + (y2 + length) ** 2) / (2 * length ** 2))
                    return dis1 + dis2
                else:
                    if x2 > x1:
                        distance = -math.sqrt((x2 - x1) ** 2)
                        return distance
                    else:
                        distance = math.sqrt((x2 - x1) ** 2)
                        return distance
        else:
            if x1 > 750:
                if x2 > 750:
                    length1 = math.sqrt((x1 - 750) ** 2 + y1 ** 2)
                    length2 = math.sqrt((x2 - 750) ** 2 + y2 ** 2)
                    length3 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    dis2 = Road.R * math.acos((length1 ** 2 + length2 ** 2 - length3 ** 2) / (2 * length1 * length2))
                    return signal * abs(dis2)
                elif y2 > 0:
                    dis1 = 750 - x2
                    length1 = Road.R
                    length2 = math.sqrt((x1 - 750) ** 2 + y1 ** 2)
                    length3 = math.sqrt((x1 - 750) ** 2 + (y1 - Road.R) ** 2)
                    dis2 = Road.R * math.acos((length1 ** 2 + length2 ** 2 - length3 ** 2) / (2 * length1 * length2))
                    return signal * (abs(dis2) + dis1)
                else:
                    dis1 = 750 - x2
                    length1 = Road.R
                    length2 = math.sqrt((x1 - 750) ** 2 + y1 ** 2)
                    length3 = math.sqrt((x1 - 750) ** 2 + (y1 + Road.R) ** 2)
                    dis2 = Road.R * math.acos((length1 ** 2 + length2 ** 2 - length3 ** 2) / (2 * length1 * length2))
                    return dis1 + abs(dis2)
            else:
                if x2 < -750.0:
                    length1 = math.sqrt((x1 + 750) ** 2 + y1 ** 2)
                    length2 = math.sqrt((x2 + 750) ** 2 + y2 ** 2)
                    length3 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    dis2 = Road.R * math.acos((length1 ** 2 + length2 ** 2 - length3 ** 2) / (2 * length1 * length2))
                    return signal * abs(dis2)
                elif y2 > 0:
                    dis1 = x2 + 750
                    length1 = Road.R
                    length2 = math.sqrt((x1 + 750) ** 2 + y1 ** 2)
                    length3 = math.sqrt((x1 + 750) ** 2 + (y1 - Road.R) ** 2)
                    dis2 = Road.R * math.acos((length1 ** 2 + length2 ** 2 - length3 ** 2) / (2 * length1 * length2))
                    return dis1 + abs(dis2)
                else:
                    dis1 = 750 - x2
                    length1 = Road.R
                    length2 = math.sqrt((x1 + 750) ** 2 + y1 ** 2)
                    length3 = math.sqrt((x1 + 750) ** 2 + (y1 + Road.R) ** 2)
                    dis2 = Road.R * math.acos((length1 ** 2 + length2 ** 2 - length3 ** 2) / (2 * length1 * length2))
                    return -(dis1 + abs(dis2))

    @staticmethod
    def get_lanes_distance(x, y, lane):
        pos_index = Road.in_road(x, y)
        leftS = 0.0
        rightS = 0.0

        if pos_index == RoadPosition.ERROR_ARISE:
            return 0.0, 0.0
        elif pos_index == RoadPosition.UP_OF_ROAD:
            leftS = (500 - (Road.lane_num - 1 - lane) * Road.lane_width) - y
            rightS = Road.lane_width - leftS
        elif pos_index == RoadPosition.DOWN_OF_ROAD:
            leftS = y - (-500 + (Road.lane_num - 1 - lane) * Road.lane_width)
            rightS = Road.lane_width - leftS
        elif pos_index == RoadPosition.RIGHT_OF_ROAD:
            ss_x = x - 750
            ss_y = y
            leftS = (500 - Road.lane_width * (Road.lane_num - 1 - lane)) - math.sqrt(ss_x ** 2 + ss_y ** 2)
            rightS = Road.lane_width - leftS
        elif pos_index == RoadPosition.LEFT_OF_ROAD:
            ss_x = x + 750
            ss_y = y
            leftS = (500 - Road.lane_width * (Road.lane_num - 1 - lane)) - math.sqrt(ss_x ** 2 + ss_y ** 2)
            rightS = Road.lane_width - leftS

        return  leftS, rightS

    @staticmethod
    def check_collision(vehicle1, vehicle2):
        """
        检测两车是否发生碰撞（考虑车辆朝向和环形道路特性）
        参数:
            vehicle1: 车辆对象，需有x,y,length,width,heading属性
            vehicle2: 车辆对象，需有x,y,length,width,heading属性
        返回:
            bool: True表示发生碰撞，False表示未碰撞
        """

        def get_rotated_vertices(x, y, length, width, heading):
            """计算车辆四个顶点的坐标（考虑旋转）"""
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            half_len = length / 2
            half_wid = width / 2

            # 四个顶点的相对坐标（未旋转）
            corners = [
                (half_len, half_wid),  # 右前
                (half_len, -half_wid),  # 右后
                (-half_len, -half_wid),  # 左后
                (-half_len, half_wid)  # 左前
            ]

            # 旋转并转换为绝对坐标
            rotated = []
            for dx, dy in corners:
                rx = x + dx * cos_h - dy * sin_h
                ry = y + dx * sin_h + dy * cos_h
                rotated.append((rx, ry))
            return rotated

        def project(poly, axis):
            """将多边形投影到轴上，返回投影的最小值和最大值"""
            min_proj = max_proj = None
            for x, y in poly:
                proj = x * axis[0] + y * axis[1]
                if min_proj is None or proj < min_proj:
                    min_proj = proj
                if max_proj is None or proj > max_proj:
                    max_proj = proj
            return min_proj, max_proj

        def overlaps(a, b):
            """检查两个投影是否重叠"""
            return not (a[1] < b[0] or b[1] < a[0])

        # 获取两车的顶点坐标（考虑朝向）
        poly1 = get_rotated_vertices(
            vehicle1.x, vehicle1.y,
            vehicle1.length, vehicle1.width,
            vehicle1.heading
        )
        poly2 = get_rotated_vertices(
            vehicle2.x, vehicle2.y,
            vehicle2.length, vehicle2.width,
            vehicle2.heading
        )

        # 获取所有需要检测的分离轴（两车的边法线）
        edges = []
        for i in range(len(poly1)):
            x1, y1 = poly1[i]
            x2, y2 = poly1[(i + 1) % 4]
            edge = (x2 - x1, y2 - y1)
            normal = (-edge[1], edge[0])  # 法向量
            edges.append(normal)

        for i in range(len(poly2)):
            x1, y1 = poly2[i]
            x2, y2 = poly2[(i + 1) % 4]
            edge = (x2 - x1, y2 - y1)
            normal = (-edge[1], edge[0])  # 法向量
            edges.append(normal)

        # 标准化所有轴并检查分离轴
        for axis in edges:
            # 跳过零向量
            if axis[0] == 0 and axis[1] == 0:
                continue

            # 标准化轴
            length = math.sqrt(axis[0] ** 2 + axis[1] ** 2)
            norm_axis = (axis[0] / length, axis[1] / length)

            # 投影两个多边形
            proj1 = project(poly1, norm_axis)
            proj2 = project(poly2, norm_axis)

            # 如果发现分离轴，则没有碰撞
            if not overlaps(proj1, proj2):
                return False

        # 没有找到分离轴，说明发生了碰撞
        return True

    @staticmethod
    def locate_lane(x, y):
        if Road.in_road(x, y) == RoadPosition.ERROR_ARISE:
            return -1

        if -750 < x < 750:
            index = (500 - math.sqrt(y ** 2)) // Road.lane_width
            lane = Road.lane_num - int(index) - 1
        else:
            index = (500 - math.sqrt(y ** 2 + (abs(x) - 750.0) ** 2)) // Road.lane_width
            lane = Road.lane_num - int(index) - 1

        return lane

    @staticmethod
    def spawn_vehicles_ahead(traci_client, ego_vehicle, count, ev_num=10, min_gap=50, max_gap=300):
        """
        在ego车辆前方100-300米范围内生成环境车辆
        参数：
            traci_client: traci连接对象
            ego_vehicle: 主车对象
            count: 批次计数标识
            min_gap: 最小生成距离（米）
            max_gap: 最大生成距离（米）
        """
        # 1. 预定义所有edge长度
        EDGE_LENGTHS = {
            "121to0": 1500,
            "60to61": 1500
        }
        # 设置默认长度（其他edge）
        DEFAULT_LENGTH = 26.18

        # 2. 获取主车信息
        current_edge = traci_client.vehicle.getRoadID(ego_vehicle.id)
        ego_pos = traci_client.vehicle.getLanePosition(ego_vehicle.id)

        # 3. 基于命名规则的edge推算
        def get_next_edge(edge):
            if edge == "120to121":  # 环形道路连接处
                return "121to0"
            try:
                from_node, to_node = map(int, edge.split("to"))
                return f"{to_node}to{to_node + 1}"
            except:
                return None

        # 4. 计算生成区域
        spawn_edges = []
        remaining_distance = max_gap
        current_e = current_edge
        start_pos = ego_pos + min_gap

        while remaining_distance > 0 and current_e:
            # 获取edge长度（优先特殊edge，否则使用默认值）
            e_len = EDGE_LENGTHS.get(current_e, DEFAULT_LENGTH)

            if start_pos < e_len:
                end_pos = min(start_pos + remaining_distance, e_len)
                spawn_edges.append({
                    "edge": current_e,
                    "start": start_pos,
                    "end": end_pos,
                    "length": e_len
                })
                remaining_distance -= (end_pos - start_pos)
                start_pos = 0
            else:
                start_pos -= e_len

            current_e = get_next_edge(current_e)

        # 5. 生成车辆参数（保持安全距离）
        ev_num = ev_num
        placements = []

        for _ in range(ev_num * 2):  # 尝试次数放宽
            if not spawn_edges: break

            target = random.choice(spawn_edges)
            pos = random.uniform(target["start"], target["end"])
            lane_count = 3  # 固定3车道（根据您之前的设定）
            lane = random.randint(0, lane_count - 1)

            # 同车道20米间距检测
            conflict = any(
                abs(p["pos"] - pos) < 20
                and p["lane"] == lane
                and p["edge"] == target["edge"]
                for p in placements
            )

            if not conflict:
                placements.append({
                    "edge": target["edge"],
                    "pos": pos,
                    "lane": lane
                })
                if len(placements) >= ev_num: break

        # 6. 生成车辆和路线
        for i, place in enumerate(placements):
            ev_id = f"env_{count}_{i}"

            # 构建10条edge的路线
            route_edges = []
            current_e = place["edge"]
            for _ in range(15):
                route_edges.append(current_e)
                current_e = get_next_edge(current_e)
                if not current_e: break
            car_type = random.choice(["Car1", "Car2", "Car3"])
            # 添加车辆到SUMO
            try:
                traci_client.route.add(f"route_{ev_id}", route_edges)
                traci_client.vehicle.add(
                    ev_id,
                    routeID=f"route_{ev_id}",
                    typeID=car_type,
                    departSpeed="15",
                    departLane=str(place["lane"]),
                    departPos=str(place["pos"])
                )
            except Exception as e:
                print(f"车辆生成失败 {ev_id}: {str(e)}")

    @staticmethod
    def vehicles_ahead_num(ego_vehicle, evs, detection_range=500):
        near_front_evs_num = 0
        for ev in evs:
            dis = Road.relation_distance(ego_vehicle.x, ego_vehicle.y, ev.x, ev.y)
            if 0 < dis < detection_range:
                near_front_evs_num += 1

        return near_front_evs_num

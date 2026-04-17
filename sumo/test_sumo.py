import random

import math

import traci
from traci.exceptions import TraCIException
from convoyVehicle import ConvoyVehicle
from road import Road
from vehicle import Vehicle

# 假设已经定义了 Direction, ControlCar, Vehicle, DecisionMaker 类和其他相关变量
SIMULATION_DURATION = 300  # 仿真总时长，秒

all_cvs = []  # 所有编队车辆
all_evs = []  # 所有环境车辆


def spawn_vehicles_ahead(traci_client, ego_vehicle, count, ev_num=10, min_gap=50, max_gap=300):
    """
    在ego车辆前方100-300米范围内生成环境车辆
    参数：
        traci_client: traci连接对象
        ego_vehicle: 主车对象（需包含id,x,y属性）
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

        # 添加车辆到SUMO
        try:
            traci_client.route.add(f"route_{ev_id}", route_edges)
            traci_client.vehicle.add(
                ev_id,
                routeID=f"route_{ev_id}",
                typeID="Car3",
                departSpeed="15",
                departLane=str(place["lane"]),
                departPos=str(place["pos"])
            )
        except Exception as e:
            print(f"车辆生成失败 {ev_id}: {str(e)}")


def has_vehicles_ahead(ego_vehicle, evs, detection_range=500):
    near_front_evs_num = 0
    for ev in evs:
        dis = Road.relation_distance(ego_vehicle.x, ego_vehicle.y, ev.x, ev.y)
        if 0 < dis < detection_range:
            near_front_evs_num += 1

    return near_front_evs_num


def start_sumo():
    try:
        # 定义 SUMO 可执行文件路径
        sumo_binary = "sumo-gui"  # 如果你想用GUI模式，使用 "sumo-gui"，否则使用 "sumo"
        # 定义网络和配置文件
        sumo_config = "../configs/convoy.sumocfg"  # SUMO 配置文件，包含地图和仿真参数
        # 启动 SUMO 模拟
        traci.start([sumo_binary, "-c", sumo_config, "--random", "--lanechange.duration", "1.5"])
    except TraCIException:
        input("无法连接到SUMO服务器，请检查配置并按回车键重试。")
        return


def main():
    # 所有受控车辆的id
    cv_names = ["veh1", "veh2", "veh3", "veh4", "veh5", "veh6", "veh7", "veh8"]
    desired_lanes = [1, 0, 1, 0, 1, 0, 1, 0]

    for i, cv_id in enumerate(cv_names):
        temp_v = ConvoyVehicle(cv_id, desired_lanes[i])
        all_cvs.append(temp_v)

    all_cvs[0].state = 1

    count = 0

    start_sumo()
    simulation_time = 0
    while simulation_time < SIMULATION_DURATION:
        count += 1
        traci.simulationStep()
        simulation_time = traci.simulation.getTime()
        traci.gui.trackVehicle("View #0", "veh8")

        vehicles_list = traci.vehicle.getIDList()
        for c in vehicles_list:
            if c not in cv_names:
                temp_v = Vehicle(c)
                temp_v.update_state(traci)
                # print(f"{temp_v.id}'s speed is {temp_v.speed}")
                all_evs.append(temp_v)

        front_evs_num = has_vehicles_ahead(all_cvs[7], all_evs)
        if simulation_time % 10 == 0 and front_evs_num < 10:
            spawn_vehicles_ahead(traci, all_cvs[7], count, 10-front_evs_num)

        for c in all_cvs:
            c.update_state(traci)
            # c.show_state()

        for c in all_cvs:
            c.find_surround_evs(all_evs)
            c.find_neighborhoods(all_cvs)
            # c.show_neighborhoods()
            c.void_obstacles(all_evs)
            c.planning(all_cvs)
        for c in all_cvs:
            traci.vehicle.moveToXY(c.id, "", c.lane, c.new_x, c.new_y, angle=90 - c.heading * 180 / math.pi, keepRoute=2)
            for ev in all_evs:
                if Road.check_collision(c, ev):
                    print(f"{c.id} and {ev.id} collide!")

        all_evs.clear()

    traci.close()


if __name__ == "__main__":
    main()

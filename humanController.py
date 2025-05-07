import json
import os
import time
import keyboard
from typing import Dict, List
from sumo.mutilEnv import MutilEnv
from reasoning.scenarioDescription import ScenarioDescription
from sumo.road import Road
import traci


class HumanController:
    def __init__(self, env: MutilEnv, output_file: str = "human_decisions.json"):
        """
        人类驾驶员控制器（支持数字+控制键组合指定车辆）
        参数：
            env: MutilEnv 环境实例
            output_file: 决策数据保存路径
        """
        self.env = env
        self.output_file = output_file
        self.decision_data = []

        # 数字键映射到车辆ID
        self.vehicle_mapping = {
            '1': 'veh1',
            '2': 'veh2',
            '3': 'veh3',
            '4': 'veh4',
            '5': 'veh5',
            '6': 'veh6',
            '7': 'veh7',
            '8': 'veh8'
        }

        # 控制键映射到动作
        self.action_mapping = {
            'up': 3,  # 加速
            'w': 3,  # 加速
            'down': 4,  # 减速
            's': 4,  # 减速
            'left': 0,  # 左变道
            'a': 0,  # 左变道
            'right': 2,  # 右变道
            'd': 2  # 右变道
        }

        # 当前选中的车辆ID（默认为None表示未选择）
        self.selected_vehicle = None
        self.last_key_time = 0
        self.key_combination_timeout = 1.0  # 组合键超时时间（秒）

    def get_vehicle_by_id(self, vehicle_id: str):
        """根据车辆ID获取车辆对象"""
        for cv in self.env.convoy_vehicles:
            if cv.id == vehicle_id:
                return cv
        return None

    def get_human_action(self) -> Dict[str, int]:
        """
        获取键盘输入并解析为车辆-动作组合
        返回：
            Dict: {vehicle_id: action_id} 字典
        """
        current_time = time.time()
        actions = {}

        # 检测数字键（车辆选择）
        for num_key, vehicle_id in self.vehicle_mapping.items():
            if keyboard.is_pressed(num_key):
                self.selected_vehicle = vehicle_id
                self.last_key_time = current_time
                print(f"已选择车辆: {vehicle_id}")

        # 检查组合键是否超时
        if current_time - self.last_key_time > self.key_combination_timeout:
            self.selected_vehicle = None

        # 检测控制键（动作）
        if self.selected_vehicle:
            for control_key, action in self.action_mapping.items():
                if keyboard.is_pressed(control_key):
                    actions[self.selected_vehicle] = action
                    print(f"控制 {self.selected_vehicle} 执行动作: {action}")
                    self.selected_vehicle = None  # 执行后清除选择
                    break

        return actions

    def get_scenario_description(self, cv) -> str:
        """
        生成场景描述信息（与decisionMaker.py格式一致）
        参数：
            cv: 当前控制的车辆
        返回：
            str: 格式化的场景描述
        """
        sce = ScenarioDescription(self.env, cv)
        scenario_desc = sce.describe()
        avail_actions = sce.availableActionsDescription()

        return f"""\
Driving scenario description:
{scenario_desc}
Driving Intentions:
1.Driving carefully and void collision.
2.The safe distance for changing lanes is at least 15 meters.
3.Keep a safe distance of at least 15 meters from the ahead of the surrounding vehicles.
4.Try to maintain the desired speed of 25 m/s when there is no vehicle ahead in the same lane.
5.Limit speed is 15~30 m/s.
Available actions:
{avail_actions}"""

    def save_decision(self, cv, action: int):
        """
        保存决策数据到JSON文件
        参数：
            cv: 当前控制的车辆
            action: 采取的动作ID
        """
        system_msg = """You are a large language model. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex highway driving scenarios.
You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.
Your response should use the following format:
Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to IDLE, then output `1`>"""
        scenario_desc = self.get_scenario_description(cv)

        decision_record = {
            "instruction": system_msg,
            "input": scenario_desc,
            "output": f"Response to user:### {action}"
        }

        self.decision_data.append(decision_record)

        # 实时写入文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.decision_data, f, indent=2, ensure_ascii=False)

    def run(self):
        """
        主控制循环
        """
        print("=== 组合键控制模式 ===")
        print("控制方式：")
        print(" 1. 先按数字键(1-8)选择车辆（如按'1'选择veh1）")
        print(" 2. 在1秒内按控制键执行动作：")
        print("    - 加速: ↑ 或 W")
        print("    - 减速: ↓ 或 S")
        print("    - 左变道: ← 或 A")
        print("    - 右变道: → 或 D")
        print("示例：按'1'然后按'←' = 控制veh1左变道")
        print("=====================")

        try:
            count = 0
            simulation_time = 0
            while True:
                # 获取人类控制指令
                vehicle_actions = self.get_human_action()

                # 准备动作列表（默认所有车辆保持）
                action_list = [1] * len(self.env.convoy_vehicles)

                # 应用人类控制指令
                for i, cv in enumerate(self.env.convoy_vehicles):
                    if cv.id in vehicle_actions:
                        action_list[i] = vehicle_actions[cv.id]
                        self.save_decision(cv, action_list[i])

                # 环境车辆生成逻辑
                control_vehicles = self.env.convoy_vehicles
                all_evs = self.env.get_all_evs()
                front_evs_num = Road.vehicles_ahead_num(control_vehicles[7], all_evs)
                if simulation_time % 10 == 0 and front_evs_num < 10:
                    Road.spawn_vehicles_ahead(traci, control_vehicles[7], count, 10 - front_evs_num)

                # 执行动作
                self.env.step(action_list)
                count += 1
                simulation_time = traci.simulation.getTime()
                print("simulation_time:",simulation_time)
                print("front_evs_num:",front_evs_num)

        except KeyboardInterrupt:
            print("\n=== 退出控制模式 ===")
            print(f"决策数据已保存到: {self.output_file}")


if __name__ == "__main__":
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    env = MutilEnv(render_mode="human", result_folder=result_folder)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    controller = HumanController(env, output_file=f"datasets/human_decisions_{current_time}.json")

    # 重置环境
    obs, _ = env.reset(seed=42)

    # 运行控制器
    controller.run()

    # 关闭环境
    env.close()
import json
import os
import time
import keyboard
import traci
from sumo.mutilEnv import MutilEnv
from sumo.road import Road
from reasoning.scenarioDescription import ScenarioDescription


class SingleVehicleController:
    def __init__(self, output_file: str = "human_decisions.json"):
        """
        单车控制器（基于修改后的MutilEnv）
        按键改成 on_release_key 事件驱动
        """
        self.output_file = output_file
        self.decision_data = []
        self.step_count = 0
        self.last_save_step = 0

        # 初始化环境（仅包含veh1）
        self.env = MutilEnv(render_mode="human", result_folder="results")
        self.ego_id = "veh1"

        # 控制键映射
        self.action_mapping = {
            'up': 3, 'w': 3,
            'down': 4, 's': 4,
            'left': 0, 'a': 0,
            'right': 2, 'd': 2
        }

        # 用来在回调里临时存一次按键动作，主循环每步取一次
        self.next_action = None

        # 注册按键抬起回调
        for key in self.action_mapping:
            keyboard.on_release_key(key, self._on_key_release)

        os.makedirs("datasets", exist_ok=True)

    def _on_key_release(self, event):
        """抬起时触发一次，将 action 存到队列里"""
        act = self.action_mapping.get(event.name)
        if act is not None:
            self.next_action = act

    def get_ego_vehicle(self):
        if self.env.convoy_vehicles and self.env.convoy_vehicles[0].id == self.ego_id:
            return self.env.convoy_vehicles[0]
        return None

    def get_scenario_description(self) -> str:
        ego = self.get_ego_vehicle()
        if not ego:
            return "No vehicle available"
        sce = ScenarioDescription(self.env, ego)
        return (
f"Driving scenario description:\n{sce.describe()}\n"
f"Driving Intentions:\n"
"1. Driving carefully and avoid collision.\n"
"2. The safe distance for changing lanes is at least 15 meters.\n"
"3. Keep a safe distance of at least 15 meters from the ahead of the surrounding vehicles.\n"
"4. Try to maintain the desired speed of 25 m/s when there is no vehicle ahead in the same lane.\n"
"5. Limit speed is 15~30 m/s.\n"
f"Available actions:\n{sce.availableActionsDescription()}"
        )

    def should_save_data(self, action: int) -> bool:
        # 非默认动作总要存
        if action != 1:
            return True
        # 默认动作每60步存一次
        if self.step_count - self.last_save_step >= 60:
            self.last_save_step = self.step_count
            return True
        return False

    def save_decision(self, action: int):
        if not self.should_save_data(action):
            return
        system_msg = """You are a large language model. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex highway driving scenarios.
You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.
Your response should use the following format:
Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to IDLE, then output `1`>"""
        scenario_desc = self.get_scenario_description()
        record = {
            "instruction": system_msg,
            "input": scenario_desc,
            "output": f"Response to user:### {action}"
        }
        self.decision_data.append(record)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.decision_data, f, indent=2, ensure_ascii=False)

    def spawn_environment_vehicles(self, ev_num=10):
        ego = self.get_ego_vehicle()
        if not ego:
            return
        print("veh1.speed:",ego.speed)
        sim_time = traci.simulation.getTime()
        if sim_time % 10 == 0:
            all_evs = self.env.get_all_evs()
            front_evs_num = Road.vehicles_ahead_num(ego, all_evs)
            if front_evs_num < ev_num:
                Road.spawn_vehicles_ahead(traci, ego, sim_time,
                                          ev_num - front_evs_num,
                                          min_gap=20, max_gap=300)

    def run(self):
        print("=== 单车控制模式 ===")
        print("控制键位：")
        print(" 加速: ↑/W")
        print(" 减速: ↓/S")
        print(" 左变道: ←/A")
        print(" 右变道: →/D")
        print("===================")
        print("数据保存策略：")
        print(" - 键盘控制动作立即保存")
        print(" - 自动保持动作每60步保存一次")
        print("===================")
        test_list_seed = [52, 5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 5326,
                          4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31, 8317,
                          123, 456, 789, 101, 202, 303, 404, 505, 606, 707,
                          808, 909, 111, 222, 333, 444, 555, 666, 777, 888,
                          999, 121, 232, 343, 454, 565, 676, 787, 898, 919]
        obs, _ = self.env.reset(seed=test_list_seed[1])
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                self.step_count += 1
                self.spawn_environment_vehicles()

                # 如果回调里有动作，就用它；否则保持 (1)
                action = self.next_action if self.next_action is not None else 1
                # 用过之后清空，保证只触发一次
                self.next_action = None

                self.save_decision(action)
                self.env.step([action])

        except KeyboardInterrupt:
            print("\n=== 退出控制模式 ===")
            print(f"总步数: {self.step_count}")
            print(f"数据集已保存到: {self.output_file}")
            self.env.close()


if __name__ == "__main__":
    current_time = time.strftime("%Y%m%d-%H%M%S")
    path = f"datasets/human_decisions_{current_time}.json"
    controller = SingleVehicleController(path)
    controller.run()

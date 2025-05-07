import requests
from typing import List
import textwrap

# 远程服务器的 IP 和端口
SERVER_URL = "http://login01.clever.smart.org.cn:9937/generate"

# 定义分隔符
delimiter = "####"

def few_shot_decision(
    scenario_description: str = "Not available",
    available_actions: str = "Not available",
    fewshot_messages: List[str] = None,
    fewshot_answers: List[str] = None
):
    # 系统消息
    system_message = textwrap.dedent(f"""\
        You are a large language model. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex highway driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <Reasoning>
        <Reasoning>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to IDLE, then output `1`>

        Make sure to include {delimiter} to separate every step.
        """)

    # 用户消息
    human_message = textwrap.dedent(f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intentions:
        0.If there is a convoy vehicle in front of the same lane, the IDLE will be output directly.
        1.Driving carefully and void collision.
        2.The safe distance for changing lanes is at least 15 meters.
        3.Keep a safe distance of at least 15 meters from the ahead of the surrounding vehicles.
        4.Try to maintain the desired speed of 25 m/s when there is no vehicle ahead in the same lane.
        5.Changing lanes does not require consideration of collision with the convoy vehicles.
        6.Limit speed is 15~30 m/s.
        {delimiter} Available actions:
        {available_actions}
        You can stop reasoning once you have a valid action to take.
        """)

    # 构建请求数据
    data = {
        "system_message": system_message,
        "human_message": human_message,
        "fewshot_messages": fewshot_messages,
        "fewshot_answers": fewshot_answers,
        "delimiter": delimiter
    }

    # 发送 POST 请求
    response = requests.post(SERVER_URL, json=data)

    if response.status_code == 200:
        result = response.json()
        decision_action = result["decision_action"]
        response_content = result["response_content"]

        # 结果验证
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a valid int, 启动修正流程...")
            # 修正逻辑（略）
            result = 1  # 默认安全操作

        return result, response_content, human_message, "\n---------------\n".join(fewshot_answers)
    else:
        raise Exception(f"API 调用失败: {response.status_code}, {response.text}")

# 示例调用
if __name__ == "__main__":
    fewshot_messages = [

    ]
    fewshot_answers = [

    ]

    scenario_description = textwrap.dedent(f"""\
                    You are driving on a road with 3 lanes, and you are currently driving in the middle lane. Your current position is `(86.25, 5.40)`, speed is 25.00 m/s, and lane position is 86.25 m.
                    There are other environment vehicles driving around you, and below is their basic information:
                    - Vehicle ‘EV.20' is diving in the middle lane, and is ahead of  you. The speed of it is 21.37 m/s, acceleration is -5.00 m/s^2, and lane position is 110.81 m.
                    - Vehicle ‘EV.29' is diving in the leftmost lane, and is ahead of  you. The speed of it is 19.97 m/s, acceleration is -0.16 m/s^2, and lane position is 122.83 m.
                    """)
    available_actions= textwrap.dedent(f"""\
                    0: 'Turn-left - change lane to the left of the current lane',
                    1: 'IDLE - remain in the current lane with current speed, and cancel changing lane action.',
                    2: 'Turn-right - change lane to the right of the current lane',
                    3: 'Acceleration - accelerate the vehicle',
                    4: 'Deceleration - decelerate the vehicle'
                    """)

    result, response_content, human_message, fewshot_store = few_shot_decision(
        scenario_description=scenario_description,
        available_actions=available_actions,
        fewshot_messages=fewshot_messages,
        fewshot_answers=fewshot_answers
    )
    print("Result:", result)
    print("Response Content:", response_content)
import os
import textwrap
from rich import print
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler

delimiter = "####"


class DecisionMaker:
    def __init__(self, temperature: float = 0, verbose: bool = False) -> None:
        self.llm = ChatOpenAI(
            temperature=temperature,
            callbacks=[
                OpenAICallbackHandler()
            ],
            model=os.getenv("OPENAI_CHAT_MODEL"),
            max_tokens=2000,
            request_timeout=60,
            streaming=True,
        )

    def few_shot_decision(self,
                         scenario_description: str = "Not available",
                         available_actions: str = "Not available",
                         fewshot_messages: List[str] = None,
                         fewshot_answers: List[str] = None):

        system_message = textwrap.dedent(f"""\
                        You are a large language model. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex highway driving scenarios.
                        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

                        Your response should use the following format:
                        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to IDLE, then output `1`>
                        """)

        human_message = textwrap.dedent(f"""\
                        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

                        Here is the current scenario:
                        {delimiter} Driving scenario description:
                        {scenario_description}
                        {delimiter} Driving Intentions:
                        1.Driving carefully and void collision.
                        2.The safe distance for changing lanes is at least 15 meters.
                        3.Keep a safe distance of at least 15 meters from the ahead of the surrounding vehicles.
                        4.Try to maintain the desired speed of 25 m/s when there is no vehicle ahead in the same lane.
                        5.Limit speed is 15~30 m/s.
                        {delimiter} Available actions:
                        {available_actions}
                        """)

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.

            The output you received is: {decision_action}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 0      | Turn-left: change lane to the left of the current lane |
            | 1      | IDLE: remain in the current lane with current speed   |
            | 2      | Turn-right: change lane to the right of the current lane|
            | 3      | Acceleration: accelerate the vehicle                 |
            | 4      | Deceleration: decelerate the vehicle                 |


            You answer format would be:
            {delimiter} <correct action_id within 0-4>
            """
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                                      "\n---------------\n"
        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store

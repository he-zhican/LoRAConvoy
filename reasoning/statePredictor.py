import os
import textwrap
from typing import List, Tuple, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import OpenAICallbackHandler

# Delimiter for prompt structuring
delimiter = "####"


class StatePredictor:
    def __init__(self,
                 model_name: str = None,
                 temperature: float = 0.0,
                 history_length: int = 5,
                 verbose: bool = False) -> None:
        """
        A predictor that uses a large language model to infer the current state of a vehicle
        from its recent history of states.

        Args:
            model_name: the OpenAI model to use (e.g., "gpt-4").
            temperature: sampling temperature for the LLM.
            history_length: number of past timesteps to include in the prompt.
            verbose: whether to print debug information.
        """
        self.history_length = history_length
        self.verbose = verbose
        self.llm = ChatOpenAI(
            model=model_name or os.getenv("OPENAI_CHAT_MODEL"),
            temperature=temperature,
            callbacks=[OpenAICallbackHandler()],
            max_tokens=500,
            request_timeout=60,
            streaming=False,
        )

        # System instruction for the LLM
        self.system_message = textwrap.dedent(f"""
            You are an expert driving state predictor. Given a vehicle's historical states (position x,y, speed, lane) over the last few timesteps,
            you must predict its current state (x, y, speed, lane) at the next timestep. Only output a JSON object with fields: x, y, speed, lane.
            Do not add any extra text or explanation.
            All sections are delimited by {delimiter}.
        """)

    def predict_state(self, vehicle_id: str, history: List[Tuple[float, float, float, int]]) -> Optional[dict]:
        """
        Predict the current state of a vehicle given its history.

        Args:
            vehicle_id: unique identifier of the vehicle.
            history: list of past states [(x, y, speed, lane), ...], most recent last.

        Returns:
            A dict with keys 'x', 'y', 'speed', 'lane', or None if prediction fails.
        """
        # Truncate history to the configured length
        trimmed = history[-self.history_length:]

        # Format history for the prompt
        history_str = []
        for t, (x, y, speed, lane) in enumerate(trimmed, start=1):
            history_str.append(f"t-{len(trimmed)-t+1}: x={x:.2f}, y={y:.2f}, speed={speed:.2f}, lane={lane}")
        history_block = "\n".join(history_str)

        # Build prompt
        human_message = textwrap.dedent(f"""
            {delimiter} Vehicle ID: {vehicle_id}
            {delimiter} History (most recent last):
            {history_block}
            {delimiter}
        """)

        messages = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=human_message),
        ]

        if self.verbose:
            print("=== Predict Prompt ===")
            print(self.system_message)
            print(human_message)

        # Call the LLM
        response = self.llm(messages)
        text = response.content.strip()

        if self.verbose:
            print("=== Raw LLM Output ===")
            print(text)

        # Expecting pure JSON
        try:
            import json
            result = json.loads(text)
            # Validate keys
            assert all(k in result for k in ['x', 'y', 'speed', 'lane'])
            return result
        except Exception as e:
            if self.verbose:
                print(f"Failed to parse LLM output: {e}")
            return None

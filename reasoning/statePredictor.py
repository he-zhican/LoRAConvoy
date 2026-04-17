import os
import textwrap
import math
from typing import List, Tuple, Optional, Dict
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks import OpenAICallbackHandler

# Delimiter for prompt structuring
delimiter = "####"


class StatePredictor:
    def __init__(self,
                 temperature: float = 0.0,
                 history_length: int = 5,
                 verbose: bool = False) -> None:
        """
        A predictor that uses a large language model to infer the current state of a vehicle
        from its recent history of states.

        Args:
            temperature: sampling temperature for the LLM.
            history_length: number of past timesteps to include in the prompt.
            verbose: whether to print debug information.
        """
        self.history_length = history_length
        self.verbose = verbose
        self.llm = ChatOpenAI(
            model=os.getenv("PREDICT_MODEL"),
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
        """)
        
        # 用于存储预测值和真实值，用于计算精度
        # 格式: [{"predicted": {x, y, speed, lane}, "actual": {x, y, speed, lane}, "vehicle_id": str}, ...]
        self.prediction_records: List[Dict] = []

    def predict_state(self, vehicle_id: str, history: List[Tuple[float, float, float, int]], 
                     actual_state: Optional[Dict] = None) -> Optional[dict]:
        """
        Predict the current state of a vehicle given its history.

        Args:
            vehicle_id: unique identifier of the vehicle.
            history: list of past states [(x, y, speed, lane), ...], most recent last.
            actual_state: optional dict with actual state {x, y, speed, lane} for accuracy calculation.

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
            
            # 如果提供了真实值，记录预测结果用于精度计算
            if actual_state is not None:
                self.prediction_records.append({
                    "vehicle_id": vehicle_id,
                    "predicted": result.copy(),
                    "actual": actual_state.copy()
                })
            
            return result
        except Exception as e:
            if self.verbose:
                print(f"Failed to parse LLM output: {e}")
            return None
    
    def calculate_accuracy_metrics(self) -> Dict[str, float]:
        """
        计算预测精度指标。
        
        Returns:
            包含各种精度指标的字典
        """
        if len(self.prediction_records) == 0:
            return {
                "total_predictions": 0,
                "position_errors": [],
                "x_errors": [],
                "y_errors": [],
                "speed_errors": [],
                "speed_relative_errors": [],
                "lane_correct": 0
            }
        
        position_errors = []
        x_errors = []
        y_errors = []
        speed_errors = []
        speed_relative_errors = []
        lane_correct = 0
        
        for record in self.prediction_records:
            pred = record["predicted"]
            actual = record["actual"]
            
            # 位置误差（欧氏距离）
            x_err = abs(pred['x'] - actual['x'])
            y_err = abs(pred['y'] - actual['y'])
            pos_err = math.sqrt(x_err ** 2 + y_err ** 2)
            position_errors.append(pos_err)
            x_errors.append(x_err)
            y_errors.append(y_err)
            
            # 速度误差
            speed_err = abs(pred['speed'] - actual['speed'])
            speed_errors.append(speed_err)
            
            # 速度相对误差（百分比）
            if actual['speed'] > 0:
                speed_rel_err = (speed_err / actual['speed']) * 100
            else:
                speed_rel_err = speed_err * 100 if speed_err > 0 else 0
            speed_relative_errors.append(speed_rel_err)
            
            # 车道准确率
            if pred['lane'] == actual['lane']:
                lane_correct += 1
        
        return {
            "total_predictions": len(self.prediction_records),
            "position_errors": position_errors,
            "x_errors": x_errors,
            "y_errors": y_errors,
            "speed_errors": speed_errors,
            "speed_relative_errors": speed_relative_errors,
            "lane_correct": lane_correct
        }
    
    def print_accuracy_report(self, output_file: Optional[str] = None):
        """
        计算并输出预测精度报告。
        
        Args:
            output_file: 可选的文件路径，如果提供则将报告写入文件
        """
        metrics = self.calculate_accuracy_metrics()
        
        if metrics["total_predictions"] == 0:
            report = "=== State Predictor Accuracy Report ===\n"
            report += "No predictions recorded yet.\n"
        else:
            pos_errors = metrics["position_errors"]
            x_errors = metrics["x_errors"]
            y_errors = metrics["y_errors"]
            speed_errors = metrics["speed_errors"]
            speed_rel_errors = metrics["speed_relative_errors"]
            lane_correct = metrics["lane_correct"]
            total = metrics["total_predictions"]
            
            # 计算统计量
            def calc_stats(values):
                if len(values) == 0:
                    return 0.0, 0.0, 0.0, 0.0
                mean = sum(values) / len(values)
                sorted_vals = sorted(values)
                median = sorted_vals[len(sorted_vals) // 2] if len(sorted_vals) > 0 else 0.0
                min_val = min(values)
                max_val = max(values)
                return mean, median, min_val, max_val
            
            pos_mean, pos_median, pos_min, pos_max = calc_stats(pos_errors)
            x_mean, x_median, x_min, x_max = calc_stats(x_errors)
            y_mean, y_median, y_min, y_max = calc_stats(y_errors)
            speed_mean, speed_median, speed_min, speed_max = calc_stats(speed_errors)
            speed_rel_mean, speed_rel_median, speed_rel_min, speed_rel_max = calc_stats(speed_rel_errors)
            lane_accuracy = (lane_correct / total) * 100 if total > 0 else 0.0
            
            report = "=" * 60 + "\n"
            report += "State Predictor Accuracy Report\n"
            report += "=" * 60 + "\n"
            report += f"Total Predictions: {total}\n"
            report += "\n"
            
            report += "1. Position Error (Euclidean Distance)\n"
            report += f"   Mean:   {pos_mean:.4f} m\n"
            report += f"   Median: {pos_median:.4f} m\n"
            report += f"   Min:    {pos_min:.4f} m\n"
            report += f"   Max:    {pos_max:.4f} m\n"
            report += "\n"
            
            report += "2. X Coordinate Error\n"
            report += f"   Mean:   {x_mean:.4f} m\n"
            report += f"   Median: {x_median:.4f} m\n"
            report += f"   Min:    {x_min:.4f} m\n"
            report += f"   Max:    {x_max:.4f} m\n"
            report += "\n"
            
            report += "3. Y Coordinate Error\n"
            report += f"   Mean:   {y_mean:.4f} m\n"
            report += f"   Median: {y_median:.4f} m\n"
            report += f"   Min:    {y_min:.4f} m\n"
            report += f"   Max:    {y_max:.4f} m\n"
            report += "\n"
            
            report += "4. Speed Error\n"
            report += f"   Mean:   {speed_mean:.4f} m/s\n"
            report += f"   Median: {speed_median:.4f} m/s\n"
            report += f"   Min:    {speed_min:.4f} m/s\n"
            report += f"   Max:    {speed_max:.4f} m/s\n"
            report += "\n"
            
            report += "5. Speed Relative Error\n"
            report += f"   Mean:   {speed_rel_mean:.4f}%\n"
            report += f"   Median: {speed_rel_median:.4f}%\n"
            report += f"   Min:    {speed_rel_min:.4f}%\n"
            report += f"   Max:    {speed_rel_max:.4f}%\n"
            report += "\n"
            
            report += "6. Lane Prediction Accuracy\n"
            report += f"   Correct Predictions: {lane_correct} / {total}\n"
            report += f"   Accuracy: {lane_accuracy:.2f}%\n"
            report += "\n"
            
            report += "=" * 60 + "\n"
        
        # 输出到控制台
        # print(report)
        
        # 如果指定了输出文件，写入文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Accuracy report saved to: {output_file}")
    
    def reset_accuracy_records(self):
        """清空预测记录，用于新的评估周期"""
        self.prediction_records.clear()

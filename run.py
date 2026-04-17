import random

import yaml
import os
from rich import print
import time

from reasoning.scenarioDescription import ScenarioDescription
from reasoning.decisionMaker import DecisionMaker
from sumo.mutilEnv import MutilEnv

test_list_seed = [52, 5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 5326,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31, 8317,
                  123, 456, 789, 101, 202, 303, 404, 505, 606, 707,
                  808, 909, 111, 222, 333, 444, 555, 666, 777, 888,
                  999, 121, 232, 343, 454, 565, 676, 787, 898, 919]


def setup_env(config):
    os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
    os.environ["DECISION_MODEL"] = config['DECISION_MODEL']
    os.environ["PREDICT_MODEL"] = config['PREDICT_MODEL']
    os.environ["OPENAI_API_BASE"] = config['OPENAI_API_BASE']


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    setup_env(config)

    sumo_home = config["SUMO_HOME"]
    all_vehicles_decision = config["all_vehicles_decision"]
    result_folder = config["result_folder"]

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    episode = 33
    env = MutilEnv("human", sumo_home=sumo_home, result_folder=result_folder)
    # 用于收集所有episode的state_predictor，用于最终精度报告
    all_state_predictors = []
    
    while episode < config["episodes_num"]:
        success = False
        envType = f'sumo-highway-multi'
        result_prefix = f"sumo_highway_multi_{episode}"
        # seed = random.choice(test_list_seed)
        seed = test_list_seed[episode]
        obses, _ = env.reset(seed=seed, episode=episode)
        env.initialize_video_writer(result_folder + "/" + result_prefix + ".mp4")

        # Create a ScenarioDescription class for each vehicle in the convoy and record {"vhe 1":sce} with a dictionary
        sce_dict = {}
        for i, cv in enumerate(env.convoy_vehicles):
            sce = ScenarioDescription(env, cv)
            sce_dict[env.cv_names[i]] = sce

        decisionMaker = DecisionMaker()
        
        # 收集当前episode中所有车辆的state_predictor（如果存在）
        episode_predictors = []
        for cv in env.convoy_vehicles:
            if cv.state_predictor is not None and cv.state_predictor not in episode_predictors:
                episode_predictors.append(cv.state_predictor)
                if cv.state_predictor not in all_state_predictors:
                    all_state_predictors.append(cv.state_predictor)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1
        try:
            already_decision_steps = 0
            total_decision_time = 0  # Track total decision time
            decision_count = 0       # Track number of decisions
            for i in range(0, config["simulation_duration"]):
                if not all_vehicles_decision:
                    all_lead_cvs = env.get_all_lead_cvs()
                action_list = []
                for cv in env.convoy_vehicles:
                    if not all_vehicles_decision:
                        if cv not in all_lead_cvs:
                            action_list.append(1)
                            continue
                            
                    sce_descrip = sce_dict[cv.id].describe()
                    avail_action = sce_dict[cv.id].availableActionsDescription()
                    print(f'[cyan]{cv.id} Scenario description: [/cyan]\n', sce_descrip)

                    start_time = time.time()
                    action, response, human_question = decisionMaker.make_decision(
                        scenario_description=sce_descrip, available_actions=avail_action
                    )
                    end_time = time.time()

                    # Accumulate decision time statistics
                    decision_time = end_time - start_time
                    total_decision_time += decision_time
                    decision_count += 1

                    action_list.append(action)
                    already_decision_steps += 1

                obses, rewards, dones, truncateds, _ = env.step(action_list)

                print("=================================")

                if True in dones:
                    print("[red]Simulation crash after running steps: [/red] ", i)
                    collision_frame = i
                    break
                if True in truncateds:
                    success = True
                    print("[green]Simulation finnish successfully! [/green] ")
                    break
        finally:
            avg_decision_time = total_decision_time / decision_count if decision_count > 0 else 0

            with open(result_folder + "/" + 'log.txt', 'a') as f:
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | success: {} | Avg decision time: {:.4f}s | aver_speeds: {} | File prefix: {} \n"
                    .format(episode, seed, already_decision_steps, success, avg_decision_time, env.get_aver_speeds(), result_prefix))

            # 输出当前episode的预测精度报告
            if len(episode_predictors) > 0:
                print(f"\n[cyan]=== Episode {episode} State Predictor Accuracy Report ===[/cyan]")
                accuracy_file = os.path.join(result_folder, f"accuracy_episode_{episode}.txt")
                for predictor in episode_predictors:
                    predictor.print_accuracy_report(output_file=accuracy_file)
                print()

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close_video_writer()
    
    # 输出所有episode的总体精度报告
    if len(all_state_predictors) > 0:
        print("\n[bold cyan]" + "=" * 60)
        print("Overall State Predictor Accuracy Report (All Episodes)")
        print("=" * 60 + "[/bold cyan]")
        overall_accuracy_file = os.path.join(result_folder, "accuracy_overall.txt")
        
        # 合并所有predictor的记录到第一个predictor中
        main_predictor = all_state_predictors[0]
        for predictor in all_state_predictors[1:]:
            main_predictor.prediction_records.extend(predictor.prediction_records)
        
        main_predictor.print_accuracy_report(output_file=overall_accuracy_file)
        print()
    
    env.close()

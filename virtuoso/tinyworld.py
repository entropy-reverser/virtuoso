from typing import List, Dict, Any, Optional
import time
from abc import ABC, abstractmethod
from openai import OpenAI
import re
from .baseagent import*
from .tinyperson import*


class TinyWorld:
    """对话世界模拟器"""
    def __init__(self, agents: List[BaseAgent], scene: str, memory_window: int = 3):
        self.agents = agents
        self.scene = scene
        self.memory_window = memory_window
        self.rounds = 0
        self.context = {
            "scene": scene,
            "history": [],
            "shared_memory": {}
        }
        
    def run(self, num_rounds: int) -> List[Dict[str, Any]]:
        """运行指定轮数的对话"""
        results = []
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*20} {self.scene} Round {round_num} {'='*20}")
            
            # 构建当前轮次上下文
            current_context = {
                "scene": self.scene,
                "round": round_num,
                "shared_memory": self.context["shared_memory"],
                "recent_history": self._get_recent_history()
            }
            
            round_results = []
            for agent in self.agents:
                response = agent.act(round_num, self.scene, current_context)
                round_results.append(response)
                print(f"{agent.name}:  {response['thought']}")
                print(f"{agent.name}:  {response['speech']}")
                print(f"{agent.name}:  {response['action']}")
                
                # 更新共享记忆
                self.context["shared_memory"].update({
                    f"{agent.name}_contribution_{round_num}": response["speech"],
                    f"{agent.name}_action_{round_num}": response["action"]
                })
            
            # 保存本轮结果
            self.context["history"].append({
                "round": round_num,
                "results": round_results
            })
            results.extend(round_results)
            self.rounds += 1
            time.sleep(1)  # 模拟自然对话间隔
            
        return results
    
    def _get_recent_history(self) -> List[Dict[str, Any]]:
        """获取最近几轮的对话历史"""
        start_idx = max(0, len(self.context["history"]) - self.memory_window)
        return self.context["history"][start_idx:]
    
 
    def save_conversation_to_file(self, filename: str = "conversation_log.txt"):
        """将对话记录保存为文本文件"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"场景：{self.scene}\n")
            f.write(f"总对话轮数：{self.rounds}\n\n")
            
            for round_data in self.context["history"]:
                f.write(f"{'='*20} 第{round_data['round']}轮 {'='*20}\n")
                for result in round_data["results"]:
                    f.write(f"[{result['agent']}的思考]\n{result['thought']}\n")
                    f.write(f"[{result['agent']}的对话]\n{result['speech']}\n\n")
                    f.write(f"[{result['agent']}的行为]\n{result['action']}\n\n")
                f.write("\n")
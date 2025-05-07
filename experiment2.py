

from typing import List, Dict, Any, Optional
import time
from abc import ABC, abstractmethod
from openai import OpenAI

"""
"""

class BaseAgent(ABC):
    """智能体基类（抽象类）"""
    def __init__(self, name: str, role: str, traits: List[str]):
        self.name = name
        self.role = role
        self.traits = traits
        self.api_client = self._initialize_api_client()
        self.conversation_history = []  # 保存对话历史
        
    def _initialize_api_client(self) -> OpenAI:
        """初始化LLM客户端"""
        return OpenAI(
            api_key="sk-08e683555efd433b8ef0d346a0cf5fa8",#选取合适的API密钥，如deepseek
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    @abstractmethod
    def think(self, scene: str, context: Dict[str, Any]) -> str:
        """根据场景和上下文生成思考过程"""
        pass
    
    @abstractmethod
    def speak(self, scene: str, context: Dict[str, Any]) -> str:
        """根据场景和上下文生成对话内容"""
        pass
    
    def act(self, round_num: int, scene: str, context: Dict[str, Any]) -> Dict[str, str]:
        """执行完整动作流程"""
        thought = self.think(scene, context)
        speech = self.speak(scene, context)
        
        # 构建完整响应
        response = {
            "agent": self.name,
            "round": round_num,
            "thought": thought,
            "speech": speech
        }
        
        # 更新对话历史
        self.conversation_history.append(response)
        return response


class TinyPerson(BaseAgent):
    """"""
    def __init__(self, name: str, role: str, traits: List[str], personality: Dict[str, Any]):
        super().__init__(name, role, traits)
        self.personality = personality
        
    def _build_prompt(self, scene: str, context: Dict[str, Any]) -> str:
        """构建LLM提示词"""
        return f"""你正在扮演{self.name}，一位{self.role}。你的核心特质是：{', '.join(self.traits)}。
        
        当前场景是：{scene}
        上下文记忆：{context}
        
        请按照以下格式进行思考和回应：
        [思考] 先进行内部思考，考虑当前对话场景和对话上下文
        [行动] 然后输出符合角色设定的对话内容，要求符合当前对话主题
        保持专业性和自然对话风格，结合你的{self.personality['style']}沟通方式。
        """
    
    def think(self, scene: str, context: Dict[str, Any]) -> str:
        """通过LLM生成思考过程"""
        prompt = self._build_prompt(scene, context)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.api_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def speak(self, scene: str, context: Dict[str, Any]) -> str:
        """通过LLM生成对话内容"""
        prompt = self._build_prompt(scene, context)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.api_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content


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
                print(f"{agent.name}: [思考] {response['thought']}")
                print(f"{agent.name}: [行动] {response['speech']}")
                
                # 更新共享记忆
                self.context["shared_memory"].update({
                    f"{agent.name}_contribution_{round_num}": response["speech"]
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


# 示例用法
if __name__ == "__main__":
    # 创建智能体
    lisa = LLMEnhancedAgent(
        name="Lisa",
        role="数据科学家",
        traits=["分析型思维", "技术热情", "跨部门协作"],
        personality={
            "style": "专业且亲和力强",
            "expertise": ["机器学习", "数据可视化", "产品设计"]
        }
    )
    
    oscar = LLMEnhancedAgent(
        name="Oscar",
        role="建筑师",
        traits=["空间想象力", "可持续发展意识", "技术创新"],
        personality={
            "style": "热情且富有创意",
            "expertise": ["绿色建筑", "智能设计", "项目管理"]
        }
    )
    
    # 创建对话世界
    world = TinyWorld(
        agents=[lisa, oscar],
        scene="产品发布会：AI写作助手讨论",
        memory_window=2  # 保留最近两轮的对话记忆
    )
    
    # 运行对话
    results = world.run(4)
    
    # 输出完整对话记录
    print("\n完整对话记录：")
    for record in results:
        print(f"{record['agent']}: [THOUGHT] {record['thought']}")
        print(f"{record['agent']}: [SPEECH] {record['speech']}")
        print("-" * 50)
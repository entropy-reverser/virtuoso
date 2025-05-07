# tiny_simulation.py

import os
import json
from typing import List, Dict, Union, Optional
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ------------------ 数据模型定义 ------------------
class PersonaModel(BaseModel):
    """角色核心属性验证模型"""
    name: str
    age: int
    gender: str
    nationality: str
    occupation: Dict[str, str]
    long_term_goals: List[str]
    personality: Dict[str, Union[List[str], Dict]]
    # 其他字段根据实际需要扩展...

class AgentConfig(BaseModel):
    """智能体配置验证模型"""
    type: str
    persona: PersonaModel

# ------------------ 核心类实现 ------------------
class TinyPerson:
    """角色智能体实现类（支持define方法初始化）"""
    def __init__(self, config: Union[Dict, str, Path] = None):
        """
        初始化方式：
        1. 通过define方法编程定义角色
        2. 保留兼容性：可继续使用JSON文件或字典配置
        """
        self._config = {}
        if config is not None:
            try:
                if isinstance(config, (str, Path)):
                    with open(config, 'r', encoding='utf-8') as f:
                        raw_config = json.load(f)
                    validated_config = AgentConfig(**raw_config).model_dump()
                else:
                    validated_config = AgentConfig(**config).model_dump()
                self._config.update(validated_config["persona"])
            except ValidationError as e:
                raise ValueError(f"配置验证失败: {e}")
        self._init_llm_client()

    @classmethod
    def define(cls, **kwargs) -> 'TinyPerson':
        """类方法：通过编程方式定义角色（替代JSON文件）"""
        instance = cls()  # 创建空实例
        # 处理标准字段
        instance._config.update({
            "name": kwargs.get("name", "Unnamed"),
            "age": kwargs.get("age", 30),
            "gender": kwargs.get("gender", "Unknown"),
            "nationality": kwargs.get("nationality", "Unknown"),
        })
        
        # 处理职业信息
        if "occupation" in kwargs:
            occupation = kwargs["occupation"]
            instance._config["occupation"] = {
                "title": occupation.get("title", "Occupation"),
                "organization": occupation.get("organization", "Organization"),
                "description": occupation.get("description", "")
            }
        
        # 处理长期目标
        instance._config["long_term_goals"] = kwargs.get("long_term_goals", [])
        
        # 处理性格特征
        if "personality" in kwargs:
            personality = kwargs["personality"]
            instance._config["personality"] = {
                "traits": personality.get("traits", []),
                "style": personality.get("style", "Standard")
            }
        
        # 处理偏好
        if "preferences" in kwargs:
            instance._config["preferences"] = {
                "interests": kwargs["preferences"].get("interests", [])
            }
        
        # 处理行为
        if "behaviors" in kwargs:
            instance._config["behaviors"] = {
                "routines": kwargs["behaviors"].get("routines", [])
            }
        
        return instance

    def _init_llm_client(self):
        """初始化LLM客户端"""
        self.llm_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", "#"),
            base_url="#",
        )
        if not self.llm_client.api_key:
            raise EnvironmentError("未找到DASHSCOPE_API_KEY环境变量")

    def _build_system_prompt(self) -> str:
        """构建角色设定系统提示词"""
        traits = "\n".join(self._config["personality"]["traits"])
        return f"""
        你正在扮演{self._config['name']}，一位{self._config['age']}岁的{self._config['occupation']['title']}。
        角色设定：
        - 国籍：{self._config['nationality']}
        - 性别：{self._config['gender']}
        - 职业：{self._config['occupation']['title']}（{self._config['occupation']['organization']}）
        - 核心特质：{traits}
        - 说话风格：{self._config['personality']['style']}
        - 长期目标：{"，".join(self._config['long_term_goals'])}
        - 兴趣爱好：{", ".join(self._config.get('preferences', {}).get('interests', []))}
        """

    def listen_and_act(self, stimulus: str) -> str:
        """处理输入并生成符合角色的响应"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        response = self.llm_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

# ------------------ 工厂类增强 ------------------
class TinyPersonFactory:
    """角色生成工厂类（支持混合输入）"""
    def __init__(self, scenario: str):
        if not isinstance(scenario, str):
            raise TypeError("场景描述必须为字符串")
        self.scenario = scenario
        self._init_llm_client()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        self.llm_client = OpenAI(
            api_key="#",
            base_url="#",
        )

    def generate_from_define(self, definition: Dict) -> TinyPerson:
        """通过编程定义生成角色"""
        return TinyPerson.define(**definition)

# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 方式1：使用define方法编程定义角色
    lisa = TinyPerson.define(
        name="Lisa",
        age=28,
        nationality="Canadian",
        occupation={
            "title": "Data Scientist",
            "organization": "Microsoft",
            "description": """
            You are a data scientist. You work at Microsoft, in the M365 Search team. Your main role is to analyze 
            user behavior and feedback data, and use it to improve the relevance and quality of the search results. 
            You also build and test machine learning models for various search scenarios, such as natural language 
            understanding, query expansion, and ranking. You care a lot about making sure your data analysis and 
            models are accurate, reliable and scalable. Your main difficulties typically involve dealing with noisy, 
            incomplete or biased data, and finding the best ways to communicate your findings and recommendations to 
            other teams. You are also responsible for making sure your data and models are compliant with privacy and 
            security policies.
            """
        },
        behaviors={
            "routines": ["Every morning, you wake up, do some yoga, and check your emails."]
        },
        personality={
            "traits": [
                "You are curious and love to learn new things.",
                "You are analytical and like to solve problems.",
                "You are friendly and enjoy working with others.",
                "You don't give up easily, and always try to find a solution. However, sometimes you can get frustrated when things don't work as expected."
            ],
            "style": "Professional yet approachable"
        },
        preferences={
            "interests": [
                "Artificial intelligence and machine learning.",
                "Natural language processing and conversational agents.",
                "Search engine optimization and user experience.",
                "Cooking and trying new recipes.",
                "Playing the piano.",
                "Watching movies, especially comedies and thrillers."
            ]
        },
        long_term_goals=[
            "推动零碳建筑设计标准",
            "开发模块化生态住宅系统"
        ]
    )

    # 方式2：通过工厂生成新角色
    factory = TinyPersonFactory("A hospital in São Paulo.")
    oscar = factory.generate_from_define({
        "name": "Oscar",
        "age": 35,
        "nationality": "Brazilian",
        "occupation": {
            "title": "Doctor",
            "organization": "Hospital São Paulo",
            "description": "You are a passionate doctor who loves both his patients and his pets. You have a special interest in nature conservation and enjoy listening to heavy metal music in your free time."
        },
        "personality": {
            "traits": [
                "Empathetic and patient",
                "Adventurous and curious",
                "Loves animals and nature",
                "Expressive through music"
            ],
            "style": "Warm and enthusiastic"
        }
    })

    # 创建世界实例并运行
    class TinyWorld:
        def __init__(self, environment: str, agents: List[TinyPerson]):
            self.environment = environment
            self.agents = agents

        def run(self, rounds: int = 3):
            print(f"\n{'='*20} {self.environment} {'='*20}")
            for round_num in range(1, rounds+1):
                print(f"\nRound {round_num}:")
                for agent in self.agents:
                    response = agent.listen_and_act(f"Start conversation in {self.environment}")
                    print(f"{agent._config['name']} --> {response}")

    world = TinyWorld("Hospital Conference Room", [lisa, oscar])
    world.run(4)
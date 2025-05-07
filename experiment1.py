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
    """角色智能体实现类（支持JSON/define两种初始化方式）"""
    def __init__(self, config: Union[Dict, str, Path]):
        """
        初始化方式：
        1. 字典配置：直接传入配置字典
        2. 文件路径：传入json文件路径（str或Path对象）
        """
        try:
            if isinstance(config, (str, Path)):
                # 修复编码问题：显式指定utf-8编码
                with open(config, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                validated_config = AgentConfig(**raw_config).model_dump()
            else:
                validated_config = AgentConfig(**config).model_dump()
        except ValidationError as e:
            raise ValueError(f"配置验证失败: {e}")

        self.persona = validated_config["persona"]
        self.memory = []
        self._init_llm_client()
    
    @classmethod
    def define(cls, **kwargs) -> 'TinyPerson':
        """类方法：通过编程方式定义角色"""
        base_structure = {
            "type": "TinyPerson",
            "persona": {
                "name": kwargs.get("name", "Unnamed"),
                "age": kwargs.get("age", 30),
                "gender": kwargs.get("gender", "Unknown"),
                # 其他字段根据参数自动填充...
            }
        }
        return cls(base_structure)

    def _init_llm_client(self):
        """初始化LLM客户端"""
        self.llm_client = OpenAI(
            api_key="sk-08e683555efd433b8ef0d346a0cf5fa8",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if not self.llm_client.api_key:
            raise EnvironmentError("未找到DASHSCOPE_API_KEY环境变量")

    # 原有方法保持不变，此处省略...
    def _build_system_prompt(self) -> str:
        """构建角色设定系统提示词"""
        persona = self.persona
        traits = "\n".join(persona["personality"]["traits"])
        beliefs = "\n".join(persona["beliefs"])
        return f"""
        你正在扮演{persona['name']}，一位{persona['age']}岁的{persona['occupation']['title']}。
        角色设定：
        - 性格特点：{persona['style']}
        - 核心特质：{traits}
        - 关键信念：{beliefs}
        - 说话风格：{persona['style']}
        当前目标：{"，".join(persona['long_term_goals'])}
        """

    def listen_and_act(self, stimulus: str) -> str:
        """处理输入并生成符合角色的响应"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *self._build_memory_context(),
            {"role": "user", "content": stimulus}
        ]
        
        response = self.llm_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        action = response.choices[0].message.content
        self._update_memory(stimulus, action)
        return action
    
    def _build_memory_context(self) -> List[Dict]:
        """构建对话历史上下文"""
        return [
            {"role": "assistant" if msg["sender"] == "self" else "user", 
             "content": msg["content"]}
            for msg in self.memory[-3:]  # 保留最近3轮对话
        ]
    
    def _update_memory(self, stimulus: str, response: str):
        """更新对话记忆"""
        self.memory.extend([
            {"sender": "other", "content": stimulus},
            {"sender": "self", "content": response}
        ])
    
# ------------------ 工厂类增强 ------------------
class TinyPersonFactory:
    """角色生成工厂类（支持混合输入）"""
    def __init__(self, scenario: str):
        if not isinstance(scenario, str):
            raise TypeError("场景描述必须为字符串")
        self.scenario = scenario
        self._init_llm_client()
    
    def generate_from_define(self, definition: Dict) -> TinyPerson:
        """通过编程定义生成角色"""
        return TinyPerson(definition)
    
    def generate_from_json(self, json_path: Union[str, Path]) -> TinyPerson:
        """从JSON文件生成角色"""
        return TinyPerson(json_path)

    # 原有生成方法保持不变...

    def generate_person(self, instruction: str) -> TinyPerson:
        """根据指令生成角色配置"""
        prompt = f"""
        根据以下场景和指令生成角色JSON配置：
        场景：{self.scenario}
        指令：{instruction}
        输出格式需严格遵循Friedrich_Wolf.agent.json的结构
        """
        
        response = self.llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            config = json.loads(response.choices[0].message.content)
            return TinyPerson(config)
        except json.JSONDecodeError:
            raise ValueError("生成的角色配置格式错误")

# ------------------ 世界类强化 ------------------
class TinyWorld:
    """多智能体交互环境（增强版）"""
    def __init__(self, environment: str, agents: List[TinyPerson]):
        if len(agents) < 2:
            raise ValueError("至少需要两个智能体进行交互")
        
        self.environment = environment
        self.agents = self._validate_agents(agents)
        self.conversation_log = []
    
    def _validate_agents(self, agents) -> Dict[str, TinyPerson]:
        """验证智能体列表"""
        validated = {}
        for agent in agents:
            if not isinstance(agent, TinyPerson):
                raise TypeError(f"无效的智能体类型: {type(agent)}")
            name = agent.persona["name"]
            if name in validated:
                raise ValueError(f"角色名称重复: {name}")
            validated[name] = agent
        return validated
    
    # 其他方法保持不变...
    def run(self, rounds: int = 3):
        """执行多轮对话"""
        for _ in range(rounds):
            for name, agent in self.agents.items():
                context = self._get_context_for(name)
                response = agent.listen_and_act(context)
                self._log_interaction(name, response)
    
    def _get_context_for(self, agent_name: str) -> str:
        """构建当前对话上下文"""
        return "\n".join([
            f"{log['sender']}: {log['content']}" 
            for log in self.conversation_log[-2:]
        ])
    
    def _log_interaction(self, sender: str, content: str):
        """记录交互信息"""
        self.conversation_log.append({
            "sender": sender,
            "content": content,
           #"timestamp": datetime.now().isoformat()
        })

# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 方式1：使用JSON文件（显式指定编码）
    try:
        wolf = TinyPerson("tinyperson.json")
    except FileNotFoundError:
        print("错误：配置文件不存在")
    except json.JSONDecodeError:
        print("错误：配置文件格式无效")

    # 方式2：编程定义角色
    custom_agent = TinyPerson.define(
        name="Emma",
        age=28,
        gender="Female",
        occupation={
            "title": "Sustainable Architect",
            "organization": "GreenBuild Inc."
        },
        long_term_goals=[
            "推动零碳建筑设计标准",
            "开发模块化生态住宅系统"
        ]
    )

    # 创建世界实例
    world = TinyWorld(
        environment="Sustainable Design Forum",
        agents=[wolf, custom_agent]
    )

    # 运行对话
    try:
        world.run(rounds=5)
    except KeyboardInterrupt:
        print("\n对话被用户中断")
    except Exception as e:
        print(f"运行时错误: {str(e)}")
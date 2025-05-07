import os
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, ValidationError, validator

# ------------------ 数据验证模型 ------------------
class RoleDefinition(BaseModel):
    """角色定义验证模型"""
    name: str
    age: int
    gender: str = "unknown"
    occupation: str
    traits: List[str]
    goals: List[str]
    knowledge_base: Dict[str, List[str]]
    conversation_style: str
    definition_source: str  # 记录定义文本
    
    @validator('age')
    def validate_age(cls, v):
        if not 0 < v < 150:
            raise ValueError('年龄必须在1-149之间')
        return v

# ------------------ 核心角色类 ------------------
class TinyAgent:
    """纯文本定义的角色智能体"""
    _definition_log = []  # 类级别定义记录
    
    def __init__(self):
        self.definition = None
        self.memory = []
        self._init_llm_client()
    
    @classmethod
    def define(cls, definition_text: str) -> 'TinyAgent':
        """定义新角色并记录原始文本"""
        # 解析定义文本
        params = cls._parse_definition(definition_text)
        
        # 创建实例并验证
        agent = cls()
        try:
            agent.definition = RoleDefinition(
                **params,
                definition_source=definition_text  # 记录原始文本
            )
        except ValidationError as e:
            raise ValueError(f"定义验证失败: {e.errors()}")
        
        # 记录定义
        cls._definition_log.append({
            "timestamp": datetime.now().isoformat(),
            "definition": definition_text,
            "params": params
        })
        return agent
    
    @staticmethod
    def _parse_definition(text: str) -> Dict:
        """解析自然语言定义文本"""
        # 示例解析逻辑（实际应使用LLM增强解析）
        params = {
            "name": "Unnamed",
            "age": 30,
            "occupation": "未知职业",
            "traits": [],
            "goals": [],
            "knowledge_base": {},
            "conversation_style": "中立"
        }
        
        # 简单关键词提取（实际需扩展）
        if "建筑师" in text:
            params.update({
                "occupation": "建筑师",
                "knowledge_base": {
                    "专业领域": ["可持续设计", "模块化建筑"]
                }
            })
        if "激进" in text:
            params["conversation_style"] = "激进"
        return params
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        self.llm_client = OpenAI(
            api_key="sk-08e683555efd433b8ef0d346a0cf5fa8",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if not self.llm_client.api_key:
            raise EnvironmentError("未配置DASHSCOPE_API_KEY环境变量")
    
    def respond(self, input_text: str) -> str:
        """生成符合角色设定的响应"""
        prompt = f"""
        角色设定：{self.definition.json(exclude={'definition_source'})}
        对话历史：{self.memory[-3:]}
        当前输入：{input_text}
        请生成符合角色设定的自然响应：
        """
        
        response = self.llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        output = response.choices[0].message.content
        self._update_memory(input_text, output)
        return output
    
    def _update_memory(self, input_text: str, output_text: str):
        """更新对话记忆"""
        self.memory.extend([
            {"type": "input", "text": input_text, "time": datetime.now()},
            {"type": "output", "text": output_text, "time": datetime.now()}
        ])
    
    @classmethod
    def get_definitions(cls) -> List[Dict]:
        """获取所有定义记录"""
        return cls._definition_log

# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 定义角色
    design_agent = TinyAgent.define("""
    定义一个名为艾玛的激进派可持续建筑师，年龄28岁
    特征：
    - 坚持使用可再生材料
    - 反对传统建筑范式
    - 擅长模块化设计
    目标：
    - 推动零碳建筑标准
    - 改革建筑行业规范
    知识库：
    - 材料科学：竹纤维复合材料、再生混凝土
    - 设计理论：生物仿生学、参数化设计
    """)
    
    # 查看定义记录
    print("当前定义记录：")
    for record in TinyAgent.get_definitions():
        print(f"[{record['timestamp']}] {record['definition'][:50]}...")
    
    # 进行对话
    try:
        response = design_agent.respond("如何看待传统钢筋混凝土建筑？")
        print(f"\n角色响应：{response}")
    except Exception as e:
        print(f"对话失败：{str(e)}")
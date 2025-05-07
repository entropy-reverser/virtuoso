from typing import List, Dict, Any, Optional
import time
from abc import ABC, abstractmethod
from openai import OpenAI
import re

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
            api_key="sk-08e683555efd433b8ef0d346a0cf5fa8",#选取合适的API密钥，如deepseek，通义千问，azure openai...
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    @abstractmethod
    def think(self, scene: str, context: Dict[str, Any]) -> str:
        """根据场景和上下文生成思考过程"""
        pass
    
    @abstractmethod
    def speak1(self, scene: str, context: Dict[str, Any]) -> str:
        """根据场景和上下文生成对话内容"""
        pass

    def behavior(self, scene: str, context: Dict[str, Any]) -> str:

        pass
    
    def act(self, round_num: int, scene: str, context: Dict[str, Any]) -> Dict[str, str]:
        """执行完整动作流程"""
        thought = self.think(scene, context)
        speech = self.speak1(scene, context)
        action = self.behavior(scene, context)
        
        # 构建完整响应
        response = {
            "agent": self.name,
            "round": round_num,
            "thought": thought,
            "speech": speech,
            "action": action
        }
        
        # 更新对话历史
        self.conversation_history.append(response)
        return response
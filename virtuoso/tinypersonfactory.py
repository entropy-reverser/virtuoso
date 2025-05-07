from typing import List, Dict, Any, Optional
import time
from abc import ABC, abstractmethod
from openai import OpenAI
import re
from .baseagent import*
from .tinyperson import*




class TinyPersonFactory:
    """提供了一种通过TinyPersonFactory类使用LLM为您生成新代理规范的聪明方法。
    
    根据此类可扩展用例：
    - 生成不同场景下的TinyPerson实例
    - 使用生成的一系列TinyPerson实例进行对话模拟
    - 自动化对话调研"""
    def __init__(self, base_scene: str):
        self.base_scene = base_scene
        self.api_client = OpenAI(
            api_key="#",
            base_url="#",
        )
    
    def _parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """使用LLM解析生成指令"""
        prompt = f"""请根据以下指令和主题场景生成符合指令主题以及适合出现在当前场景下的模拟角色，并按照下面要求提取人物属性：
        指令：{instruction}
        场景：{self.base_scene}
        
        按JSON格式返回包含以下字段的结构：
        - name（根据场景生成合理名字）
        - role（职业身份）
        - traits（3个性格特质）
        - personality（包含style沟通风格和expertise专业领域）
        - interests（3个兴趣爱好）
        - goals（2个当前目标）"""
        
        response = self.api_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500,
            #stream=True

        )
        
        # 提取并清理JSON内容
        raw_json = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL).group()
        return eval(raw_json)
    
    def generate_person(self, instruction: str) -> TinyPerson:
        """生成TinyPerson实例"""
        attributes = self._parse_instruction(instruction)
        
        return TinyPerson(
            name=attributes["name"],
            role=attributes["role"],
            traits=attributes["traits"],  
            personality={
                "style": attributes["personality"]["style"],
                "expertise": attributes["personality"]["expertise"],
                "interests": attributes["interests"],
                "goals": attributes["goals"]
            }
        )
from typing import List, Dict, Any, Optional
import time
from abc import ABC, abstractmethod
from openai import OpenAI
import re
from .baseagent import*

class TinyPerson(BaseAgent):
    """
    TinyPerson是一个具有特定个性特征、兴趣和目标的模拟人。 
    随着每个这样的模拟试剂在其生命中的发展，
    它从环境中接收刺激并对其采取行动。通过听来接收刺激，
    参见和其他类似方法，
    并且通过act方法执行动作。 
    还提供了listen_and_act等便利方法。"""
    def __init__(self, name: str, role: str, traits: List[str], personality: Dict[str, Any]):
        super().__init__(name, role, traits)
        self.personality = personality
        self.interests = personality.get("interests", [])
        self.goals = personality.get("goals", [])

    def listen_and_act(self, stimulus: str) -> str:
        """接收环境刺激并生成回应"""
        context = {
            "current_stimulus": stimulus,
            "personality": self.personality,
            "interests": self.interests,
            "goals": self.goals
        }
        return self.speak(scene="互动对话", context=context)
    
    def _build_prompt(self, scene: str, context: Dict[str, Any]) -> str:
        # 扩展提示模板
        prompt = f"""你正在扮演{self.name}，一位{self.role}。核心特质：{', '.join(self.traits)}
        个性风格：{self.personality['style']}
        专业领域：{', '.join(self.personality.get('expertise', []))}
        兴趣爱好：{', '.join(self.interests)}
        当前目标：{', '.join(self.goals)}
        
        当前场景：{scene}
        收到的提问：{context.get('current_stimulus', '')}
        需要结合{self.personality['style']}的风格进行回应，并自然融入专业领域知识。另外保持自然的说话方式，别太刻意按照模板来说话。
        """
        return prompt
        
    def _build_prompt_think(self, scene: str, context: Dict[str, Any]) -> str:
        """构建LLM提示词"""
        return f"""你正在扮演{self.name}，一位{self.role}。你的核心特质是：{', '.join(self.traits)}。
        
        当前场景是：{scene}
        上下文记录：{context}
        
        请按照以下格式输出角色当前事件下的思考：
        [思考中....] 进行内部思考，以及对其他人的看法，考虑当前对话场景和对话上下文历史，以及其他人物的行动和你的行动，符合角色设定和{self.goals}目标。
        你可以结合你的{self.personality['style']}风格进行思考，保持自然的思考方式，别太刻意按照模板来。

        """
    
    def _build_prompt_speak(self, scene: str, context: Dict[str, Any]) -> str:
        """构建LLM提示词"""
        return f"""你正在扮演{self.name}，一位{self.role}。你的核心特质是：{', '.join(self.traits)}。
        
        当前场景是：{scene}
        上下文记录：{context}
        
        请按照以下格式输出角色当前事件下的对话：
        [对话中....]  输出符合角色设定的对话内容，要求符合当前对话主题，结合上下文信息，
        保持上下文连贯和自然的对话风格，结合你的{self.personality['style']}沟通方式，不要刻意强调你的角色设定，不要模板化，更专注于当前场景事件。
        你的对话内容应该符合角色设定和{self.goals}目标。
 
        """
    
    def _build_prompt_act(self, scene: str, context: Dict[str, Any]) -> str:
        """构建LLM提示词"""
        return f"""你正在扮演{self.name}，一位{self.role}。你的核心特质是：{', '.join(self.traits)}。
        
        当前场景是：{scene}
        上下文记录：{context}
        
        请按照以下格式输出角色当下的行动：
        [行动中....]  输出符合角色设定的行动内容，角色接下来具体的行动和将要做的事，要求符合当前对话主题，结合上下文信息，符合角色{self.goals}目标，
        保持上下文连贯和自然的行为逻辑，不要太刻意模板化，同时不要上文做了什么事，下一步突然做别的事了。
        """
    
    def think(self, scene: str, context: Dict[str, Any]) -> str:
        """通过LLM生成思考过程"""
        prompt = self._build_prompt_think(scene, context)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.api_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
           
        )
        
        return response.choices[0].message.content
    
    def speak1(self, scene: str, context: Dict[str, Any]) -> str:
        """通过LLM生成对话内容"""
        prompt = self._build_prompt_speak(scene, context)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.api_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
            
        )
        
        return response.choices[0].message.content
    
    def behavior(self, scene: str, context: Dict[str, Any]) -> str:
        """通过LLM生成行为内容"""
        prompt = self._build_prompt_act(scene, context)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.api_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
          
        )
        
        return response.choices[0].message.content
    
    def speak(self, scene: str, context: Dict[str, Any]) -> str:
        """通过LLM生成日常对话内容"""
        prompt = self._build_prompt(scene, context)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.api_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
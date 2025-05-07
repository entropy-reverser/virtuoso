from typing import List, Dict, Any, Optional
import time
from abc import ABC, abstractmethod
from openai import OpenAI
import re
from virtuoso.tinyperson import *
from virtuoso.tinypersonfactory import*
from virtuoso.tinyworld import*

"""
"""
# 示例用法
if __name__ == "__main__":

    # 使用工厂创建人物
    factory = TinyPersonFactory("在一场权威有声望的科技产品展览会，一家初创公司研发出了一套虚拟现实设备，让人们可以在AI生成的物理空间中进行各种交互和游戏（比如枪战）" )
    person1 = factory.generate_person(
        "你是该初创公司的产品首席技术官，名为musk,一位经验丰富的工程师，专注于产品研发与技术创新，你复杂了该虚拟设备的设计和实现。"
    )
    
    # 测试listen_and_act方法
    print(person1.listen_and_act("介绍下你自己并且能告诉我这个虚拟现实设备的工作原理吗？"))

    person2 = factory.generate_person(
        "你是参观本场科技展会的一名来自伯克利大学的学生，名为jack，学习的是计算机科学专业，专注于人工智能和虚拟现实技术，你对该虚拟设备产品非常感兴趣。"
    )

    
  
    person3 = factory.generate_person(
        "你是参观本场科技展会的一名投资者，名为lisa，专注于投资初创公司和新兴科技领域，你对该虚拟设备产品非常感兴趣。"
    )
    
    # 创建智能体
    """
    lisa = TinyPerson(
        name="Lisa",
        role="母亲",
        traits=["易怒易暴燥", "疼爱孩子","有限的耐心"],
        personality={
            "style": "一切以孩子为中心",
            #"expertise": ["机器学习", "数据可视化", "产品设计"]
        }
    )
    
    oscar = TinyPerson(
        name="Oscar",
        role="军人",
        traits=["冷静", "头脑清晰", "果断"],
        personality={
            "style": "专业且顾全大局",
            "expertise": ["使用枪械", "杀丧尸", "项目管理"]
        }
    )
    """
    
    # 创建对话世界
    world = TinyWorld(
        agents=[person1, person2, person3],
        scene="在一场权威有声望的科技产品展览会，一家初创公司研发出了一套虚拟现实设备，让人们可以在AI生成的物理空间中进行各种交互和游戏（比如枪战），musk向jack和lisa介绍该产品，并欲求得投资和潜在消费者的兴趣", 
        memory_window=2  # 保留最近两轮的对话记忆
    )
    
    # 运行对话
    results = world.run(3)

    # 运行对话后添加：
    world.save_conversation_to_file("投资产品讨论.txt")
    
    # 输出完整对话记录
    #print("\n完整对话记录：")
    #for record in results:
        #print(f"{record['agent']}: [思考] {record['thought']}")
        #print(f"{record['agent']}: [行动] {record['speech']}")
       # print("-" * 50)
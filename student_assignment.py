import json
import base64
import requests

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def get_llm():
    return AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

holiday_json_format = """
    {
        "Result": [
            {
                "date": "2024-10-10",
                "name": "國慶日"
            }
        ]
    }
"""

add_holiday_json_format = """
{
    "Result": 
        {
            "add": true,
            "reason": "描述理由"
        }
}
"""

score_json_format = """
{
    "Result": 
        {
            "score": 5498
        }
}
"""

def format_json(data):
    return json.dumps(data, indent=4, ensure_ascii=False)

@tool
def get_holiday_tool(country, year, month) -> json:
    '根據問句擷取以下資訊:year為年份、month為月份、country為國家。格式例如:year=2025、month=10、country=TW'
    response = requests.get('https://calendarific.com/api/v2/holidays', 
                            params={
                                'api_key': 'CGphSV08hRKMEjT1Om9tp5NJeHoQvFd0',
                                'country': country,
                                'year': year,
                                'month': month
                            })
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Request error：{response.status_code}\n{response.text}')
        return json.loads('{"Result": {}}')

history = {}
def get_history_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history:
        history[session_id] = InMemoryChatMessageHistory()
    return history[session_id]

def generate_hw01(question):
    prompt = f'你是一個熟知各國節假日及紀念日的人，請根據問句以json格式回答，參考以下格式:{holiday_json_format}'
    message = (HumanMessage(content=prompt+question))
    response = get_llm().invoke([message])
    return format_json(JsonOutputParser().invoke(response))
    
def generate_hw02(question):
    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是一個熟知各國節假日及紀念日的人'),
        ('human', '{input}'),
        ('human', '{agent_scratchpad}'),
    ])
    tool=[get_holiday_tool]
    agent=create_tool_calling_agent(get_llm(), tool, prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tool)
    data = agent_executor.invoke({'input': question})
    response = get_llm().invoke(f'請把{data}以json格式印出，參考以下格式:{holiday_json_format}')
    return format_json(JsonOutputParser().invoke(response))
    
def generate_hw03(question2, question3):
    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是一個熟知各國節假日及紀念日的人'),
        ('human', '{input}'),
        ('human', '{agent_scratchpad}'),
    ])
    tool=[get_holiday_tool]
    agent=create_tool_calling_agent(get_llm(), tool, prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tool)

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_history_by_session_id,
        input_messages_key = 'input',
        history_messages_key = 'history',
    )

    config = {'configurable': {'session_id': 'hw03'}}

    agent_with_history.invoke({'input': question2}, config = config)
    response = agent_with_history.invoke(
        {'input': question3 + '是否需要將節日新增到節日清單中。' +
        '根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false' +
        '，並描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容' +
        f'，以json格式回答，參考以下格式:{add_holiday_json_format}'},
        config=config
    )
    return format_json(JsonOutputParser().invoke(response['output']))

def generate_hw04(question):
    # Read and encode the image file
    with open('baseball.png', "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    prompt = f'請根據問句以json格式回答，參考以下格式:{score_json_format}'
    message = HumanMessage(
        content=[
            {'type': 'text', 'text': question + prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_encoded_data}'}},
        ]
    )
    response = get_llm().invoke([message])
    return format_json(JsonOutputParser().invoke(response))
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

# print(generate_hw01('2024年台灣10月紀念日有哪些?'))
# print(generate_hw02('2024年台灣10月紀念日有哪些?'))
# print(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？'))
# print(generate_hw04('請問日本的積分是多少'))
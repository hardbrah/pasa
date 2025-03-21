from openai import OpenAI
# from data_gen import generate_selector_prompt
import sys
import os
import json 
import logging
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)

# Set up logging
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"./log/process_log_{timestamp}.txt", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8' 
)

right_predict = 0
all_predict = 0
def process_queries():
    try:
        with open('../prompt/selector_p2.txt', 'r', encoding='utf-8') as f:
            raw_prompt = f.read().strip()
    except FileNotFoundError:
        logging.error("Unable to read the prompt file selector_p2.txt!")
        return None

    with open("../decision/q_true.txt", "r") as f:
        true_file = f.read().splitlines()

    for i in range(1, 2):
        data = json.loads(true_file[i])
        input_text = data["input"]
        output_text = data["output"]
        user_query = input_text.split("User Query: ")[1].split("\n")[0]
        title = input_text.split("Title: ")[1].split("\n")[0]
        abstract = input_text.split("Abstract: ")[1].strip()
        decision = output_text.split("**Decision:** ")[1].split("\\n")[0].strip()
        # print("User Query:", user_query)
        # print("Title:", title)
        # print("Abstract:", abstract)
        # print("Decision:", decision)

        prompt = raw_prompt.format(
            user_query=user_query,
            title=title,
            abstract=abstract
        )
        messages = [{
            'role': 'user',
            'content': prompt
        }]

        resp = client.chat.completions.create(
            model='Qwen2.5-7B-Instruct', # 注意这里的default-lora，代表使用lora进行推理，也可以使用qwen2-7b-instruct，即使用原模型了，下同
            messages=messages,
            seed=42)
        response = resp.choices[0].message.content

        with open(f"./response/qwen_response_{timestamp}.txt", 'w') as f:
            f.write(prompt + '\n' + response + '\n' + true_file[i])
        

        # 找到 "**Decision:**" 后面的值
        first_line = response.split("\n")[0]  # 获取第一行
        decision_value = first_line.split("**Decision:**")[1].strip()  # 提取 "True"
        if(decision_value == decision.strip()):
            right_predict += 1
            print("True")
        all_predict += 1

if __name__ == "__main__":
    process_queries()    
from openai import OpenAI
# from data_gen import generate_selector_prompt
import sys
import os
import json 

# # 获取目标模块所在的绝对路径
# module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_module_directory"))
# sys.path.append(module_path)

# 现在可以导入该目录中的模块
# import my_module


# client = OpenAI(
#     api_key='EMPTY',
#     base_url='http://localhost:8000/v1',
# )

def process_queries():
    try:
        with open('./prompt/selector_p2.txt', 'r', encoding='utf-8') as f:
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
            'content': query
        }]

        resp = client.chat.completions.create(
            model='Qwen2.5-7B-Instruct', # 注意这里的default-lora，代表使用lora进行推理，也可以使用qwen2-7b-instruct，即使用原模型了，下同
            messages=messages,
            seed=42)
        response = resp.choices[0].message.content

        # 找到 "**Decision:**" 后面的值
        first_line = response.split("\n")[0]  # 获取第一行
        decision_value = first_line.split("**Decision:**")[1].strip()  # 提取 "True"
        if(decision_value == decision.strip()) print("True")

if __name__ == "__main__":
    process_queries()    

# query = 'You are an elite researcher in the field of AI, conducting research on Can you provide works that used Graph Neural Networks (GNNs) to integrate node representations between graphs or construct an instance graph for multi-view data?. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent. Searched Paper: Title: Low Data Drug Discovery with One-shot Learning Abstract: Recent advances in machine learning have made significant contributions to drug discovery. Deep neural networks in particular have been demonstrated to provide significant boosts in predictive power when inferring the properties and activities of small-molecule compounds. However, the applicability of these techniques has been limited by the requirement for large amounts of training data. In this work, we demonstrate how one-shot learning can be used to significantly lower the amounts of data required to make meaningful predictions in drug discovery applications. We introduce a new architecture, the residual LSTM embedding, that, when combined with graph convolutional neural networks, significantly improves the ability to learn meaningful distance metrics over small-molecules. We open source all models introduced in this work as part of DeepChem, an open-source framework for deep-learning in drug discovery. User Query: Can you provide works that used Graph Neural Networks (GNNs) to integrate node representations between graphs or construct an instance graph for multi-view data? Output format: Decision: True/False Reason:... Decision:'
# messages = [{
#     'role': 'user',
#     'content': query
# }]
# resp = client.chat.completions.create(
#     model='Qwen2.5-7B-Instruct', # 注意这里的default-lora，代表使用lora进行推理，也可以使用qwen2-7b-instruct，即使用原模型了，下同
#     messages=messages,
#     seed=42)
# response = resp.choices[0].message.content
# print(f'query: {query}')
# print(f'response: {response}')
# # available_models = client.models.list()
# # print([model.id for model in available_models])
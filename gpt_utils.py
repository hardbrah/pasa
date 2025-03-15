import requests
from openai import OpenAI
import bs4
import json
import re
from dotenv import load_dotenv
import os

load_dotenv()


debug = False

def call_LLM(prompt,model,key,url):
    client = OpenAI(
        api_key = key,
        base_url = url)
    try:
        # 将prompt写入文件
        # 用时间戳记录输入
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = f"./text/prompt_{timestamp}.txt"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        if debug:
            return None
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        # 用时间戳记录输出
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"./text/gpt_{timestamp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)

        return response.choices[0].message.content

    except Exception as e:
        print(f"[Error] GPT 调用失败: {e}")
        return None

def get_arxiv_html(entry_id):
    """ 从 ar5iv 获取 HTML 论文内容 """
    assert re.match(r'^\d+\.\d+$', entry_id)
    url = f"https://ar5iv.labs.arxiv.org/html/{entry_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("./text/test.txt", "w", encoding="utf-8") as f:
                f.write(response.text)
            return response.text
        else:
            print(f"[Error] ar5iv 页面返回 {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"[Error] ar5iv 访问失败: {e}")
        return None

def call_gpt4o(prompt):
    """
    Call GPT-4o and return the response
    :param prompt: The prompt to send to GPT
    :return: GPT response
    """
    return call_LLM(
        prompt,
        model="gpt-4o-2024-11-20",
        key=os.getenv("GPT_API_KEY"),
        url=os.getenv("GPT_BASE_URL")
    )
def call_ds32b(prompt):
    return call_LLM(
        prompt,
        model=os.getenv("DS_R1_MODEL_32B"),
        key=os.getenv("DS_API_KEY"),
        url=os.getenv("DS_API_BASE_URL"))
def search_section_by_arxiv_id_by_LLM(arxiv_id):
    """
    通过 arXiv ID 获取论文章节信息，并调用 LLM 解析正确的引用论文标题
    """
    # # 1. 获取 HTML
    html_content = get_arxiv_html(arxiv_id)
    if not html_content:
        return None

    # 2. 解析 HTML 结构
    soup = bs4.BeautifulSoup(html_content, "lxml")

    # 3. 提取章节与参考文献
    sections = {}
    limit = 3000

    # 找到所有的大章节
    for section in soup.find_all("section", class_="ltx_section"):
        # 获取大章节标题
        title = section.find(class_="ltx_title")
        if title:
            section_title = title.get_text(strip=True)
            # 获取章节文本
            section_text = section.get_text(" ", strip=True)[:limit]
            sections[section_title] = section_text
            subsections = {}

            # 提取子章节
            for subsection in section.find_all("section", class_="ltx_subsection"):
                subsection_title = subsection.find(class_="ltx_title")
                if subsection_title:
                    # 拼接父章节名和子章节名
                    full_subsection_title = f"{section_title} {subsection_title.get_text(strip=True)}"
                    subsection_text = subsection.get_text(" ", strip=True)[:limit]
                    subsections[full_subsection_title] = subsection_text

            # 如果该章节有子章节，则记录子章节
            if subsections:
                sections.update(subsections)

    bib_section = soup.find("section", class_="ltx_bibliography")
    if bib_section:
        sections["References"] = bib_section.get_text(" ", strip=True)[:5000]

    # 4. 生成 GPT-4o 提示词
    # 读取提示词模板
    try:
        with open("./prompt/p1.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except Exception as e:
        print(f"[Error] 读取提示词模板失败: {e}")
        return None
        
    # 拼接提示词和论文内容
    prompt = prompt_template.replace("[在这里插入完整论文内容]", json.dumps(sections, indent=2, ensure_ascii=False))
    
    gpt_response = call_ds32b(prompt)
    
    if not gpt_response:
        return None
    
    try:
        extracted_references = extract_json_from_response(gpt_response)
        return extracted_references
    except json.JSONDecodeError:
        print("[Error] LLM 返回的 JSON 解析失败")
        return None

def extract_json_from_response(response):
    # 使用正则表达式匹配 JSON 数据，并去掉 ```json 和 ```
    json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    
    if json_match:
        # 提取并加载 JSON 数据
        json_data = json.loads(json_match.group(1).strip())  # 使用 .strip() 去除前后的空格
        return json_data
    else:
        print("没有找到 JSON 数据")
        return None
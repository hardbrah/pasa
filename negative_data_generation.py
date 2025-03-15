import random
from gpt_utils import get_arxiv_html,call_gpt4o
from utils import search_arxiv_id_by_title
import bs4
import json
import re
def get_cited_arxiv_id(arxiv_id):
    # 获取该论文引用的论文列表
    cited_papers = get_cited_papers(arxiv_id)
    
    if not cited_papers:
        return None
    
    while cited_papers:
        # 随机选择一篇论文
        random_title = random.choice(cited_papers)
        
        # 获取该论文的arxiv_id
        cited_arxiv_id = search_arxiv_id_by_title(random_title)
        
        if cited_arxiv_id:  # 成功获取到 arxiv_id
            # print(f"Successfully found arxiv_id for {random_title}: {cited_arxiv_id}")
            return cited_arxiv_id
        
        # 失败则移除该论文，继续尝试
        cited_papers.remove(random_title)
    
    return None

def extract_paper_titles_from_llm_response(llm_response):
    """
    Parse the LLM response and extract the titles of the papers.
    This assumes that the LLM response is a list of paper titles.
    """
    # 去除可能的包裹字符 ''' 或 """
    # 1. 去除 ```json 及 ``` 代码块标记
    cleaned_json = re.sub(r"^```json", "", llm_response.strip(), flags=re.IGNORECASE)  # 移除 ```json
    cleaned_json = cleaned_json.strip("`")  # 再次去除 ``` 反引号
    cleaned_json = cleaned_json.strip("'''").strip('"""')  # 处理三引号 ''' 或 """

    # 解析 JSON 数据
    try:
        cited_papers = json.loads(cleaned_json)
        return cited_papers
    except json.JSONDecodeError as e:
        # Warning.warning(f"JSON 解析错误: {e}")
        return None



# Function to get cited papers from arXiv paper by parsing HTML and calling LLM
def get_cited_papers(arxiv_id):
    # 1. 获取 HTML
    html_content = get_arxiv_html(arxiv_id)
    if not html_content:
        return None

    # 2. 解析 HTML 结构
    soup = bs4.BeautifulSoup(html_content, "lxml")

    # Find references section
    bib_section = soup.find("section", class_="ltx_bibliography")
    if bib_section:
        sections = {"References": bib_section.get_text(" ", strip=True)[:3000]}
    else:
        # print(f"No references section found in paper {arxiv_id}")
        return None

    # 读取提示词模板
    try:
        with open("./prompt/get_reference.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except Exception as e:
        # print(f"[Error] 读取提示词模板失败: {e}")
        return None

    # Replace placeholder in prompt template with the references section text
    prompt = prompt_template.replace("[在这里插入论文文献]", json.dumps(sections, indent=2, ensure_ascii=False))

    # 3. Send the prompt to LLM and retrieve the response
    llm_response = call_gpt4o(prompt)

    # 4. Extract titles of cited papers from LLM's response
    cited_papers = extract_paper_titles_from_llm_response(llm_response)
    if not cited_papers:  # 如果解析失败，则返回 None
        # Warning.warning("Failed to extract cited papers from LLM response.")
        return None

    return cited_papers

if __name__ == "__main__":
    arxiv_id = "2201.11903"  # Example arXiv paper ID
    cited_arxiv_id = get_cited_arxiv_id(arxiv_id)
    if cited_arxiv_id:
        print("Cited Papers:", cited_arxiv_id)
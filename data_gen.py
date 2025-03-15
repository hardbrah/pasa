import json
import random
import logging
import os
from utils import search_paper_by_arxiv_id
from gpt_utils import call_LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"./log/process_log_{timestamp}.txt", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8' 
)


def random_entry():
    """
    Randomly select a query and the associated arXiv paper ID.
    :return: tuple (question, answer_arxiv_id_list) or None
    """
    data = []
    try:
        with open('./data/AutoScholarQuery/train.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))  # Parse JSON line by line
                except json.JSONDecodeError as e:
                    logging.error(f"JSON Decode Error: {e}")
    except FileNotFoundError:
        logging.error("File train.jsonl not found!")
        return None

    if data:
        entry = random.choice(data)
        question = entry.get("question", "No question found")
        answer_arxiv_id = entry.get("answer_arxiv_id", [])
        return question, answer_arxiv_id
    else:
        logging.warning("No valid data found!")
        return None


def generate_selector_prompt(question, arxiv_id):
    """
    Generate prompts for GPT-4o based on the query and arXiv ID list.
    :param question: The user query
    :param arxiv_id: The list of arXiv paper IDs
    :return: tuple (prompt, paper_info)
    """
    try:
        with open('./prompt/selector_p2.txt', 'r', encoding='utf-8') as f:
            raw_prompt = f.read().strip()
    except FileNotFoundError:
        logging.error("Unable to read the prompt file selector_p2.txt!")
        return None

    paper_info = search_paper_by_arxiv_id(arxiv_id)
    if paper_info is None:
        logging.warning(f"Paper {arxiv_id} search failed, skipping.")
        return None

    prompt = raw_prompt.format(
        user_query=question,
        title=paper_info['title'],
        abstract=paper_info['abstract']
    )

    return prompt, paper_info

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


def process_queries(num_queries=10):
    """
    Process multiple queries, call GPT-4o for paper evaluation, and store the results.
    :param num_queries: Number of queries to process
    """
    for i in range(1, num_queries + 1):
        query_data = random_entry()
        if query_data is None:
            continue  # Skip current query and move to the next one

        question, answer_arxiv_id_list = query_data
        answer_arxiv_id = random.choice(answer_arxiv_id_list)
        answer_arxiv_id = re.sub(r'v\d+$', '', answer_arxiv_id)
        prompt_data = generate_selector_prompt(question, answer_arxiv_id)

        if prompt_data is None:
            logging.info(f"Query {i}: `{question}` skipped due to failure in retrieving paper.")
            continue

        true_file = f"./decision/q_true.txt"
        false_file = f"./decision/q_false.txt"

        prompt, paper_info = prompt_data
        response = call_gpt4o(prompt)
        if response is None:
            logging.error(f"Query {i}: `{paper_info['title']}` GPT evaluation failed!")
            continue

        # Parse GPT response
        decision_line = response.split("\n")[0].strip()
        if decision_line.startswith("**Decision:** True"):
            decision = "True"
            output_file = true_file
        elif decision_line.startswith("**Decision:** False"):
            decision = "False"
            output_file = false_file
        else:
            logging.warning(f"Query {i}: `{paper_info['title']}` GPT evaluation format error, skipping.")
            continue
            
        # Structure the data in the desired JSON format
        result = {
            "input": f"User Query: {question}\nTitle: {paper_info['title']}\nAbstract: {paper_info['abstract']}",
            "output": f"{response}"
        }

        # Write the result as a JSON line to the output file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logging.info(f"Query {i}: `{paper_info['title']}` evaluation result saved to {output_file}")

# Run the code
if __name__ == "__main__":
    output_dir = './decision'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)        
    process_queries(num_queries=1000)

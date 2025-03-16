import json
import random
import logging
import os
from utils import search_paper_by_arxiv_id
from gpt_utils import call_gpt4o
from dotenv import load_dotenv
from negative_data_generation import get_cited_arxiv_id
import re
import threading
# Load environment variables
load_dotenv()

file_lock = threading.Lock()

# Set up logging
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"./log/process_log_{timestamp}.txt", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [Thread-%(thread)d] (%(threadName)s) - %(message)s",
    encoding='utf-8' 
)

@staticmethod
def do_parallel(func, args, num):
    threads = []
    for _ in range(num):
        thread = threading.Thread(target=func, args=args,name=f"Thread-{_}")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

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


def process_queries(num_queries=10):
    """
    Process multiple queries, call GPT-4o for paper evaluation, and store the results.
    :param num_queries: Number of queries to process
    """
    true_file = "./decision/q_true_pos.txt"
    false_file = "./decision/q_false_pos.txt"
    true_file_neg = "./decision/q_true_neg.txt"
    false_file_neg = "./decision/q_false_neg.txt"

    for i in range(1, num_queries + 1):
        query_data = random_entry()
        if not query_data:
            logging.info(f"Query {i}: No data found, skipping.")
            continue  # Skip current query and move to the next one

        question, answer_arxiv_id_list = query_data
        answer_arxiv_id = choose_arxiv_id(answer_arxiv_id_list)

        if not answer_arxiv_id:
            logging.info(f"Query {i}: `{question}` skipped due to failure in retrieving answer paper.")
            continue

        # Retrieve cited paper ID
        cited_arxiv_id = find_cited_arxiv_id(answer_arxiv_id, answer_arxiv_id_list.copy())
        if not cited_arxiv_id or cited_arxiv_id in answer_arxiv_id_list:
            logging.info(f"Query {i}: `{question}` skipped due to failure in retrieving cited paper.")
            continue

        # Generate prompts and evaluate papers
        pos_prompt_data = generate_selector_prompt(question, answer_arxiv_id)
        neg_prompt_data = generate_selector_prompt(question, cited_arxiv_id)
        if not pos_prompt_data or not neg_prompt_data:
            logging.info(f"Query {i}: `{question}` skipped due to failure in generating prompts.")
            continue

        pos_prompt, pos_paper_info = pos_prompt_data
        neg_prompt, neg_paper_info = neg_prompt_data

        # Process positive response
        pos_response = process_gpt_response(pos_prompt,question, pos_paper_info, true_file, false_file, i)
        if not pos_response:
            continue

        # Process negative response
        neg_response = process_gpt_response(neg_prompt,question, neg_paper_info, true_file_neg, false_file_neg, i, neg=True)
        if not neg_response:
            continue

        logging.info(f"Query {i}: `{pos_paper_info['title']}` and `{neg_paper_info['title']}` evaluation completed.")

def choose_arxiv_id(answer_arxiv_id_list):
    """
    Select a valid arXiv ID from the list.
    """
    answer_arxiv_id = random.choice(answer_arxiv_id_list)
    return re.sub(r'v\d+$', '', answer_arxiv_id)

def find_cited_arxiv_id(answer_arxiv_id, answer_arxiv_id_list):
    """
    Search for cited arXiv ID by title.
    """
    cited_arxiv_id = get_cited_arxiv_id(answer_arxiv_id)
    while cited_arxiv_id is None and len(answer_arxiv_id_list) > 1:
        if answer_arxiv_id in answer_arxiv_id_list:
            answer_arxiv_id_list.remove(answer_arxiv_id)
        answer_arxiv_id = random.choice(answer_arxiv_id_list)
        cited_arxiv_id = get_cited_arxiv_id(answer_arxiv_id)
    return cited_arxiv_id

def process_gpt_response(prompt, question, paper_info, true_file, false_file, query_idx, neg=False):
    """
    Process the GPT response and save the result.
    """
    response = call_gpt4o(prompt)
    if response is None:
        logging.error(f"Query {query_idx}: `{paper_info['title']}` GPT evaluation failed!")
        return None

    decision_line = response.split("\n")[0].strip()
    if decision_line.startswith("**Decision:** True") \
    or decision_line.startswith("**Decision**: True") \
    or decision_line.startswith("True") \
    or decision_line.startswith("Decision: True"):
        decision = "True"
        output_file = true_file
    elif decision_line.startswith("**Decision:** False") \
    or decision_line.startswith("**Decision**: False") \
    or decision_line.startswith("False") \
    or decision_line.startswith("Decision: False"):
        decision = "False"
        output_file = false_file
    else:
        logging.warning(f"Query {query_idx}: `{paper_info['title']}` GPT evaluation format error, skipping.")
        return None

    result = {
        "input": f"User Query: {question}\nTitle: {paper_info['title']}\nAbstract: {paper_info['abstract']}",
        "output": f"{response}"
    }

    # Write the result as a JSON line to the output file
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return response

# Run the code
if __name__ == "__main__":
    output_dir = './decision'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)        
    # process_queries(num_queries=200)
    do_parallel(process_queries, (50,),10)

from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import argparse
import json
import os
import numpy as np
import faiss
import time
import logging

logging.basicConfig(
    filename='agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

# Document store for storing query, summary and raw results from web & faiss index file to store faiss index for similarity search
DOCUMENT_STORE = "document_store.json"
FAISS_INDEX_FILE = "faiss_index.index"

emb_model="text-embedding-ada-002"
chat_model="gpt-4"

# Load or create FAISS index.. OpenAI model is used for embeddings (1536 dimensions)
def load_faiss(dim=1536):
    try:
        if os.path.exists(FAISS_INDEX_FILE):
            return faiss.read_index(FAISS_INDEX_FILE)
        return faiss.IndexFlatL2(dim)
    except Exception as e:
        logger.error(f"Error loading or creating FAISS index file: {e}")
        raise

def save_faiss(index):
    try:
        faiss.write_index(index, FAISS_INDEX_FILE)
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise

def load_docstore():
    try:
        if os.path.exists(DOCUMENT_STORE):
            with open(DOCUMENT_STORE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading document store: {e}")
        raise

def save_docstore(store):
    try:
        with open(DOCUMENT_STORE, "w") as f:
            json.dump(store, f)
    except Exception as e:
        logger.error(f"Error saving document store: {e}")
        raise

#Generates an embedding for a given text
def generate_embedding(text, model=emb_model):
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

#Generates a response to a prompt
def generate_thought(prompt, model=chat_model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating thought: {e}")
        return None

#creates agent with name, state and memory
def create_agent(name):
    return {
        "name": name,
        "state": "idle",
        "memory": []
    }

#executes an action function and logs the result to agent's memory
def perform_action(agent, action, action_function, *args):
    try:
        logger.info(f"{agent['name']} is performing action: {action}")
        result = action_function(*args)
        agent["memory"].append(f"Performed action: {action}. Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error performing action: {e}")
        agent["memory"].append(f"Failed to perform action: {action}. Error: {e}")
        return None

#receiving feedback for a task
def receive_feedback(agent, task_id, feedback):
    try:
        logger.info(f"{agent['name']} received feedback for task {task_id}: {feedback}")
        if "feedback" not in agent:
            agent["feedback"] = {}
        agent["feedback"][task_id] = feedback
    except Exception as e:
        logger.error(f"Error receiving feedback: {e}")

#analyse the feedback..mark it as positive or negative
def analyze_feedback(agent):
    try:
        if "feedback" not in agent or not agent["feedback"]:
            return "No feedback received yet."
        
        positive = sum(1 for f in agent["feedback"].values() if "good" in f.lower())
        negative = sum(1 for f in agent["feedback"].values() if "bad" in f.lower())
        
        return f"Feedback analysis: {positive} positive, {negative} negative."
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}")
        return None

#adapt the behavior based on the feedback 
def adapt_behavior(agent):
    try:
        if "feedback" not in agent:
            return
        
        for task_id, feedback in agent["feedback"].items():
            if "short" in feedback.lower():
                logger.info(f"Adjusting summarization prompt for task {task_id}.")
                agent["summarization_prompt"] = (
                    "Summarize the following content in 4-5 sentences, providing detailed insights:\n\n{content}"
                )
            elif "irrelevant" in feedback.lower():
                logger.info(f"Refining search query for task {task_id}.")
    except Exception as e:
        logger.error(f"Error adapting behavior: {e}")

#searches web with duckduckgo and extracts top 3 results based on the query
def search_web(query):
    logger.info(f"Searching DuckDuckGo for: {query}")
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.find_all("div", class_="result"):
            title = result.find("h2", class_="result__title")
            if title:
                title_text = title.get_text(strip=True)
            else:
                title_text = "No title"
            
            snippet = result.find("a", class_="result__snippet")
            if snippet:
                snippet_text = snippet.get_text(strip=True)
            else:
                snippet_text = "No snippet"
            
            results.append(f"{title_text}: {snippet_text}")

        return results[:3]
    except Exception as e:
        logger.error(f"Error searching the web: {e}")
        return []

#summarizes the content, which will be presented to the user
def summarize_content(content):
    try:
        prompt = f"Summarize the following content in 2-3 sentences:\n\n{content}"
        return generate_thought(prompt)
    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return None

#executes the task by generating thought, performing actions and summarizing the results
def execute_task(task_description, agent, actions):
    logger.info(f"Task: {task_description}")
    thought = generate_thought(f"How should I accomplish the task: {task_description}?")
    logger.info(f"{agent['name']} is thinking: {thought}")
    results = []
    for action_name, action_function, *args in actions:
        logger.info(f"Performing action: {action_name}")
        result = perform_action(agent, action_name, action_function, *args)
        results.append(result)
    
    # Flatten the results list
    flattened_results = []
    for item in results:
        if isinstance(item, list):
            flattened_results.extend(item)
        else:
            flattened_results.append(item)

    logger.info("Summarizing results...")
    
    summary = summarize_content("\n".join(flattened_results))

    logger.info(f"Task completed..")

    print(f"Task Summary:\n{summary}\n")
    task_id = str(int(time.time()))
    receive_feedback(agent, task_id, f"Task completed. Summary: {summary}")

    user_feedback = collect_user_feedback()
    if user_feedback:
        logger.info(f"User feedback received: {user_feedback}")
        receive_feedback(agent, task_id, f"User feedback: {user_feedback}")

    logger.info("Analyzing feedback...")

    feedback_analysis = analyze_feedback(agent)
    print(feedback_analysis)

    return summary, flattened_results

def collect_user_feedback():
    feedback = input("Please provide your feedback (or press Enter to skip): ")
    return feedback

#scheduler function coordinates the execution of a task by creating an agent and running the task
def run_scheduler(task_description, agent, actions):
    logger.info("Starting scheduler...")
    summary, flattened_results = execute_task(task_description, agent, actions)
    logger.info(f"Task summary: {summary}")
    logger.info("Scheduler finished.")
    return summary, flattened_results

def main():
    parser = argparse.ArgumentParser(description="Tiny Agentic Framework")
    parser.add_argument("--task", type=str, required=True, help="Task description")
    parser.add_argument("--agent", type=str, default="Agent", help="Agent name")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    args = parser.parse_args()

    global client
    client = OpenAI(api_key=args.api_key)

    document_store = load_docstore()
    faiss_index = load_faiss()

    normalized_query = args.task.lower().strip()

    if normalized_query in document_store:
        logger.info(f"Exact query found: {normalized_query}")
        logger.info(f"Retrieved summary: {document_store[normalized_query]['summary']}")
        return

    query_embedding = np.array([generate_embedding(normalized_query)]).astype("float32")
    logger.info(f"Query embedding: {query_embedding}")

    k = 1
    distances, indices = faiss_index.search(query_embedding, k)
    logger.info(f"Distances: {distances}, Indices: {indices}")

    if indices[0][0] != -1 and distances[0][0] < 0.35:
        similar_query = list(document_store.keys())[indices[0][0]]
        if similar_query in document_store:
            logger.info(f"Similar query found: {similar_query}")
            logger.info(f"Retrieved summary: {document_store[similar_query]['summary']}")
            return
        else:
            logger.info(f"Similar query found but not in document store: {similar_query}")

    agent = create_agent(args.agent)
    actions = [
        ("search_web", search_web, args.task),
    ]

    summary, flattened_results = run_scheduler(args.task, agent, actions)

    document_store[normalized_query] = {
        "summary": summary,
        "raw_results": flattened_results
    }
    save_docstore(document_store)

    faiss_index.add(query_embedding)
    save_faiss(faiss_index)

if __name__ == "__main__":
    main()

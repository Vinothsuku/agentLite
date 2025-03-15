# main.py

import requests
from bs4 import BeautifulSoup
import argparse
import json
import os
import numpy as np
import faiss
import time
import logging
from config import load_config, Config
from llm_options import get_provider

logging.basicConfig(
    filename='agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

class AgentFramework:
    def __init__(self, config: Config):
        self.config = config
        self.llm_provider = get_provider(config.llm)
        self.document_store = self.load_docstore()
        self.faiss_index = self.load_faiss()
        
    def load_faiss(self, dim=1536):
        try:
            if os.path.exists(self.config.agent.faiss_index):
                return faiss.read_index(self.config.agent.faiss_index)
            return faiss.IndexFlatL2(dim)
        except Exception as e:
            logger.error(f"Error loading or creating FAISS index file: {e}")
            raise

    def save_faiss(self):
        try:
            faiss.write_index(self.faiss_index, self.config.agent.faiss_index)
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def load_docstore(self):
        try:
            if os.path.exists(self.config.agent.document_store):
                with open(self.config.agent.document_store, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading document store: {e}")
            raise

    def save_docstore(self):
        try:
            with open(self.config.agent.document_store, "w") as f:
                json.dump(self.document_store, f)
        except Exception as e:
            logger.error(f"Error saving document store: {e}")
            raise

    def generate_embedding(self, text):
        try:
            return self.llm_provider.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def generate_thought(self, prompt):
        try:
            return self.llm_provider.generate_completion(prompt)
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return None

    def create_agent(self, name):
        return {
            "name": name,
            "state": "idle",
            "memory": []
        }

    def perform_action(self, agent, action, action_function, *args):
        try:
            logger.info(f"{agent['name']} is performing action: {action}")
            result = action_function(*args)
            agent["memory"].append(f"Performed action: {action}. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error performing action: {e}")
            agent["memory"].append(f"Failed to perform action: {action}. Error: {e}")
            return None

    def receive_feedback(self, agent, task_id, feedback):
        try:
            logger.info(f"{agent['name']} received feedback for task {task_id}: {feedback}")
            if "feedback" not in agent:
                agent["feedback"] = {}
            agent["feedback"][task_id] = feedback
        except Exception as e:
            logger.error(f"Error receiving feedback: {e}")

    def analyze_feedback(self, agent):
        try:
            if "feedback" not in agent or not agent["feedback"]:
                return "No feedback received yet."
            
            positive = sum(1 for f in agent["feedback"].values() if "good" in f.lower())
            negative = sum(1 for f in agent["feedback"].values() if "bad" in f.lower())
            
            return f"Feedback analysis: {positive} positive, {negative} negative."
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return None

    def adapt_behavior(self, agent):
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

    def search_web(self, query):
        logger.info(f"Searching {self.config.search_provider} for: {query}")
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

            return results[:self.config.max_search_results]
        except Exception as e:
            logger.error(f"Error searching the web: {e}")
            return []

    def summarize_content(self, content):
        try:
            prompt = f"Summarize the following content in 2-3 sentences:\n\n{content}"
            return self.generate_thought(prompt)
        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            return None

    def execute_task(self, task_description, agent, actions):
        logger.info(f"Task: {task_description}")
        thought = self.generate_thought(f"How should I accomplish the task: {task_description}?")
        logger.info(f"{agent['name']} is thinking: {thought}")
        results = []
        for action_name, action_function, *args in actions:
            logger.info(f"Performing action: {action_name}")
            result = self.perform_action(agent, action_name, action_function, *args)
            results.append(result)
        
        flattened_results = []
        for item in results:
            if isinstance(item, list):
                flattened_results.extend(item)
            else:
                flattened_results.append(item)

        logger.info("Summarizing results...")
        
        summary = self.summarize_content("\n".join(flattened_results))

        logger.info(f"Task completed..")

        print(f"Task Summary:\n{summary}\n")
        task_id = str(int(time.time()))
        self.receive_feedback(agent, task_id, f"Task completed. Summary: {summary}")

        user_feedback = self.collect_user_feedback()
        if user_feedback:
            logger.info(f"User feedback received: {user_feedback}")
            self.receive_feedback(agent, task_id, f"User feedback: {user_feedback}")

        logger.info("Analyzing feedback...")

        feedback_analysis = self.analyze_feedback(agent)
        print(feedback_analysis)

        return summary, flattened_results

    def collect_user_feedback(self):
        feedback = input("Please provide your feedback (or press Enter to skip): ")
        return feedback

    def run_scheduler(self, task_description, agent, actions):
        logger.info("Starting scheduler...")
        summary, flattened_results = self.execute_task(task_description, agent, actions)
        logger.info(f"Task summary: {summary}")
        logger.info("Scheduler finished.")
        return summary, flattened_results

    def process_task(self, task_description):
        normalized_query = task_description.lower().strip()

        if self.config.cache_enabled and normalized_query in self.document_store:
            logger.info(f"Exact query found: {normalized_query}")
            logger.info(f"Retrieved summary: {self.document_store[normalized_query]['summary']}")
            return self.document_store[normalized_query]['summary']

        query_embedding = np.array([self.generate_embedding(normalized_query)]).astype("float32")
        logger.info(f"Query embedding shape: {query_embedding.shape}")

        if self.config.cache_enabled:
            k = 1
            distances, indices = self.faiss_index.search(query_embedding, k)
            logger.info(f"Distances: {distances}, Indices: {indices}")

            if indices[0][0] != -1 and distances[0][0] < 0.35:
                similar_query = list(self.document_store.keys())[indices[0][0]]
                if similar_query in self.document_store:
                    logger.info(f"Similar query found: {similar_query}")
                    return self.document_store[similar_query]['summary']

        agent = self.create_agent(self.config.agent.name)
        actions = [
            ("search_web", self.search_web, task_description),
        ]

        summary, flattened_results = self.run_scheduler(task_description, agent, actions)

        if self.config.cache_enabled:
            self.document_store[normalized_query] = {
                "summary": summary,
                "raw_results": flattened_results
            }
            self.save_docstore()

            self.faiss_index.add(query_embedding)
            self.save_faiss()

        return summary

def main():
    parser = argparse.ArgumentParser(description="Tiny Agentic Framework")
    parser.add_argument("--task", type=str, required=True, help="Task description")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    framework = AgentFramework(config)
    
    result = framework.process_task(args.task)
    print(f"\nFinal Result:\n{result}")

if __name__ == "__main__":
    main()
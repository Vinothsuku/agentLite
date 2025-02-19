# agentLite
A Simple Modular Agentic Framework, that can be extended for multiple use cases, leveraging tools like OpenAI — GPT-4, FAISS for similary search and DuckDuckGo for web scraping for real-time data retrieval

Blog post [link] 

[link]: https://vinothsuku.medium.com/agentlite-a-simple-modular-agentic-framework-for-ai-apps-a04823a52777

# Key Components of agentLite
1. **Agent Core**:
  Agent is the central entity that performs tasks. It has a name, state (idle, active..) and memory to store its actions and results

2. **Task Execution**:
  Execute_task function coordinates the agent’s actions. It generates thought (using LLM), performance actions (eg — web search) and summarizes results (which can be customized)

3. **Memory & Document Store**:
  Agent stores results/output in a document_store (a json file) for future reference so that only when the user requested content is not available in this file, a web search is performed
FAISS is used to index embeddings of queries, enabling similarity search to avoid redundant work

4. **Web Search**:
  search_web function uses DuckDuckGo to retrieve real-time information from the web

5. **Summarization**:
  summarize_content function uses LLM to condense large amount of information into concise summaries

6. **Feedback mechanism**:
  Agent can receive feedback, which is logged to memory for future improvements

# agentLite - Tiny Agentic Toolkit

A lightweight and modular toolkit for prototyping AI agents that can perform tasks like web searching, content summarization, and feedback analysis. Designed for developers who want to experiment with intelligent agents without the overhead of a full-fledged framework.

**Key Features:**
- **Web Search Integration**: Searches the web using DuckDuckGo and extracts relevant results.
- **Content Summarization**: Uses OpenAI's GPT-4 to generate concise summaries of retrieved content.
**Feedback-Driven Adaptation**: Agents can analyze user feedback and adapt their behavior over time.
**Memory and Logging**: Agents maintain a memory of actions and results, with detailed logging for debugging and analysis.
**Modular Design**: Easily add new actions, modify existing ones, or integrate with other APIs.

**Use Cases:**
Automating research tasks.
Building conversational AI assistants.
Prototyping intelligent agents for specific workflows.

**Get Started:**
Clone the repository.
Install dependencies: pip install openai requests beautifulsoup4 faiss-cpu numpy.
Set your OpenAI API key and run the agent: python main.py --task "Your task description" --api_key "your_openai_api_key".
Contribute:
Feel free to fork, star, or contribute to this project! Suggestions, bug reports, and feature requests are welcome

Blog post [link] 

[link]: https://vinothsuku.medium.com/agentlite-a-simple-modular-agentic-framework-for-ai-apps-a04823a52777

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import os
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    LOCAL = "local"  # For local models like LlamaCpp

@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: str
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = None

@dataclass
class AgentConfig:
    name: str
    memory_type: str = "simple"  # simple, vector, persistent
    max_memory_items: int = 1000
    embedding_model: Optional[str] = None
    document_store: str = "document_store.json"
    faiss_index: str = "faiss_index.index"

@dataclass
class Config:
    llm: LLMConfig
    agent: AgentConfig
    logging_level: str = "INFO"
    cache_enabled: bool = True
    search_provider: str = "duckduckgo"
    max_search_results: int = 3

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file with environment variable support."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Support environment variables for sensitive data
    if 'api_key' in config_data['llm']:
        env_var = config_data['llm']['api_key']
        if env_var.startswith('${') and env_var.endswith('}'):
            env_var_name = env_var[2:-1]
            config_data['llm']['api_key'] = os.getenv(env_var_name)
    
    llm_config = LLMConfig(
        provider=LLMProvider(config_data['llm']['provider']),
        api_key=config_data['llm']['api_key'],
        model_name=config_data['llm']['model_name'],
        temperature=config_data['llm'].get('temperature', 0.7),
        max_tokens=config_data['llm'].get('max_tokens'),
        additional_params=config_data['llm'].get('additional_params', {})
    )
    
    agent_config = AgentConfig(
        name=config_data['agent']['name'],
        memory_type=config_data['agent'].get('memory_type', 'simple'),
        max_memory_items=config_data['agent'].get('max_memory_items', 1000),
        embedding_model=config_data['agent'].get('embedding_model'),
        document_store=config_data['agent'].get('document_store', 'document_store.json'),
        faiss_index=config_data['agent'].get('faiss_index', 'faiss_index.index')
    )
    
    return Config(
        llm=llm_config,
        agent=agent_config,
        logging_level=config_data.get('logging_level', 'INFO'),
        cache_enabled=config_data.get('cache_enabled', True),
        search_provider=config_data.get('search_provider', 'duckduckgo'),
        max_search_results=config_data.get('max_search_results', 3)
    )
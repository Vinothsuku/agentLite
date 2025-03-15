from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from config import LLMConfig
import anthropic
import cohere
from openai import OpenAI

class LLMProvider(ABC):
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def generate_completion(self, prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.client = OpenAI(api_key=config.api_key)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
    
    def generate_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    
    def generate_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

class AnthropicProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.client = anthropic.Client(api_key=config.api_key)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
    
    def generate_embedding(self, text: str) -> List[float]:
        raise NotImplementedError("Anthropic doesn't provide embedding API yet")
    
    def generate_completion(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.content.strip()

class CohereProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.client = cohere.Client(api_key=config.api_key)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
    
    def generate_embedding(self, text: str) -> List[float]:
        response = self.client.embed(
            texts=[text],
            model='embed-english-v3.0'
        )
        return response.embeddings[0]
    
    def generate_completion(self, prompt: str) -> str:
        response = self.client.generate(
            prompt=prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.generations[0].text.strip()

def get_provider(config: LLMConfig) -> LLMProvider:
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "cohere": CohereProvider
    }
    
    provider_class = providers.get(config.provider.value)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {config.provider}")
    
    return provider_class(config)
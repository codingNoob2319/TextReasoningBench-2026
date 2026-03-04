import os
import re
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_openai import ChatOpenAI
from openai import OpenAI

class GPT5ChatModel(BaseChatModel):
    """
    Custom GPT-5 wrapper to interface with the v1/responses endpoint.
    """
    client: Any = None
    model_name: str = "gpt-5"
    reasoning_effort: str = "medium"
    temperature: Optional[float] = None
    enable_thinking: bool = False

    def __init__(self, model_name: str, temperature: float, enable_thinking: bool):
        super().__init__()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.reasoning_effort = "high" if enable_thinking else "low"
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        input_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage): role = "developer"
            elif isinstance(m, AIMessage): role = "assistant"
            input_messages.append({"role": role, "content": m.content})
            
        reasoning_config = {"effort": self.reasoning_effort}
        if self.enable_thinking:
            reasoning_config["summary"] = "detailed"

        try:
            response = self.client.responses.create(
                model=self.model_name,
                reasoning=reasoning_config,
                input=input_messages,
                max_output_tokens=10000 
            )
            
            final_content = ""
            reasoning_summary = ""
            
            if hasattr(response, 'output'):
                for item in response.output:
                    if item.type == 'reasoning' and hasattr(item, 'summary'):
                        for sum_item in item.summary:
                            if sum_item.type == 'summary_text':
                                reasoning_summary += sum_item.text + "\n"
                    elif item.type == 'message' and hasattr(item, 'content'):
                        for content_item in item.content:
                            if content_item.type == 'output_text':
                                final_content += content_item.text

            msg = AIMessage(
                content=final_content,
                additional_kwargs={"reasoning_content": reasoning_summary.strip()}
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        except Exception as e:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])

    @property
    def _llm_type(self) -> str:
        return "gpt-5-custom"
    
class GeminiProxyChatModel(BaseChatModel):
    model_name: str
    temperature: float
    enable_thinking: bool
    api_key: str
    base_url: str
    _client: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        url = self.base_url
        if url and not url.endswith('/v1'):
            if not url.endswith('/'): url += '/'
            if not url.endswith('v1/'): url += 'v1'
        
        self._client = OpenAI(
            api_key=self.api_key, 
            base_url=url,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Thesis Experiment",
            }
        )

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        input_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage): role = "system"
            elif isinstance(m, AIMessage): role = "assistant"
            input_messages.append({"role": role, "content": m.content})

        try:
            payload = {
                "model": self.model_name,
                "messages": input_messages,
                "temperature": self.temperature,
                "max_tokens": 4500, 
            }

            if self.enable_thinking:
                payload["extra_body"] = {
                    "include_reasoning": True,
                    "reasoning": {
                        "max_tokens": 4096
                    }
                }

            response = self._client.chat.completions.create(**payload)
            
            message = response.choices[0].message
            thoughts = getattr(message, 'reasoning', None) or ""
            final_content = getattr(message, 'content', "") or ""

            msg = AIMessage(
                content=final_content,
                additional_kwargs={"reasoning_content": thoughts}
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        except Exception as e:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])

    @property
    def _llm_type(self) -> str:
        return "gemini-openrouter-native"

def get_llm(model_name: str, temperature: float = 0.7, max_retries: int = 3, enable_thinking: bool = False):
    model_name_lower = model_name.lower()
    
    api_key = None
    base_url = None
    model_kwargs = {}
    should_stream = False

    print(f"🔄 正在初始化模型: {model_name} (Thinking Mode: {enable_thinking})")
    
    if "gpt-5" in model_name_lower:
        print(f"⚡ [System] Activating GPT-5 Native Wrapper (Effort={'High' if enable_thinking else 'Low'})")
        return GPT5ChatModel(
            model_name=model_name,
            temperature=temperature,
            enable_thinking=enable_thinking
        )
    
    elif 'gemini' in model_name_lower:

        return GeminiProxyChatModel(
            model_name="google/" + model_name,
            temperature=temperature,
            enable_thinking=enable_thinking,
            api_key=os.environ.get("GOOGLE_API_KEY"),
            base_url=os.environ.get("GOOGLE_BASE_URL")
        )
    
    elif 'qwen' in model_name_lower:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        base_url = os.environ.get("QWEN_BASE_URL")
        
        if enable_thinking:
            model_kwargs["extra_body"] = {"enable_thinking": True}
            should_stream = True
        else:
            model_kwargs["extra_body"] = {"enable_thinking": False}
            should_stream = False
    
    elif 'deepseek' in model_name_lower:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL")
        if 'reasoner' in model_name_lower or enable_thinking:
             should_stream = True
    
    elif 'kimi' in model_name_lower or 'moonshot' in model_name_lower:
        api_key = os.environ.get("MOONSHOT_API_KEY")
        base_url = os.environ.get("MOONSHOT_BASE_URL")

    elif 'grok' in model_name_lower:
        api_key = os.environ.get("XAI_API_KEY")
        base_url = os.environ.get("XAI_BASE_URL")

    elif 'gpt' in model_name_lower or 'o1' in model_name_lower:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if 'o1' in model_name_lower:
            temperature = 1.0

    elif 'llama-3.1' in model_name_lower:
        api_key = os.environ.get("LLAMA_API_KEY")
        base_url = os.environ.get("LLAMA_BASE_URL")
        
        if not api_key:
            raise ValueError("lease check LLAMA_API_KEY in .env.")

        actual_model_name = "meta-llama/llama-3.1-8b-instruct"
        
        if "extra_body" in model_kwargs: 
            del model_kwargs["extra_body"]

        try:
            return ChatOpenAI(
                model=actual_model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_retries=max_retries,
                request_timeout=600,
                default_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Thesis Experiment",
                }
            )
        except Exception as e:
            raise ValueError(f"Failed to connect: {e}")

    # Local vLLM API Mode (For Gemma, etc.)
    elif 'gemma-3' in model_name_lower or 'local' in model_name_lower:
        print(f"[Routing] Local deployment model detected: {model_name}")
        
        base_url = "http://localhost:8000/v1"
        api_key = "EMPTY"
        
        if 'gemma-3-4b-it' in model_name_lower:
            model_name = "gemma-3-4b-it"
            
        if "extra_body" in model_kwargs: 
            del model_kwargs["extra_body"]

        try:
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_retries=max_retries,
                request_timeout=600
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to local vLLM service: {e}")

    else:
        raise ValueError(f"❌ API routing not configured for this model: {model_name}")

    if not api_key:
        raise ValueError(f"❌ Missing API Key! Please verify the corresponding key in the .env file.")
    
    extra_body = model_kwargs.pop("extra_body", None)

    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_retries=max_retries,
            request_timeout=600,
            streaming=should_stream,
            
            extra_body=extra_body, 
            model_kwargs=model_kwargs
        )
        return llm
    except Exception as e:
        raise ValueError(f"Model initialization failed: {e}")
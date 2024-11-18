from langchain.llms.base import LLM
from typing import Dict, Any, Mapping
from pydantic import Field

class Self_LLM(LLM):
    # Custom LLM
    #Inherited from langchain.llms.base.LLM
    # Native interface address
    url : str = None
    # The GPT-3.5 model is selected by default, which is currently commonly referred to as GPT.
    model_name: str = "gpt-3.5-turbo"
    # Access delay upper limit
    request_timeout: float = None
    # Temperature coefficient
    temperature: float = 0.1
    #API_Key
    api_key: str = None
    # Required optional parameters
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Define a method that returns default parameters
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for the call."""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}
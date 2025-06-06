from pydantic import BaseModel, Field
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig

import os


class Configuration(BaseModel):

    query_generator_model: str = Field(
        default="",
        metadata={
            "description": "The name of the language model to use for the agent's description"
        }
    )
    reflection_model: str = Field(
        default="",
        metadata={
            "description": "The name of the model to use for the agent's answer"
        }
    )

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        # get raw values from environment of config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_field.keys()
        }
        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)
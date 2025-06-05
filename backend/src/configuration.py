from pydantic import BaseModel, Field

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
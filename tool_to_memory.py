"""
Make a tool for memorizing information (save user input to the vector database).
"""
import uuid
from llama_index import (
    ServiceContext,
    VectorStoreIndex
)
from llama_index.tools import FunctionTool, ToolMetadata
from pydantic import BaseModel
import tool_for_my_notes
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAILike
from llama_index.schema import TextNode 

def make_tool(service_context:ServiceContext):
    print("tool_to_memorize.py: make_tool !!!!!!!!!!!!!!!!!!!")

    def memorize(input: str):
        """
        Memorize the user's input.

        Args:
        - input: The user's input.
        """

        print(f"Memorize this !!!!!!!!!!!!!!!!!!!! {input}")
        with open(f"{tool_for_my_notes.PATH_TO_NOTES}/memory.md", 'a') as file:
            # Append the string in a new line
            file.write(input + '\n')
        tool_for_my_notes.SHOULD_IGNORE_PERSISTED_INDEX = True

        formatted_time = "2022-01-01T00:00:00Z"

        node = TextNode(
            text=input,
            id_=str(uuid.uuid4()),
            metadata={"when": formatted_time},
        )
        # VectorStoreIndex.build_index_from_nodes(,nodes=[node], service_context=service_context)
        VectorStoreIndex.insert_nodes(self=VectorStoreIndex, nodes=[node])


    class MemorizeToolSchema(BaseModel):
        input: str

    return FunctionTool(
            memorize,
            metadata=ToolMetadata(
                name="memorize_information",
                description="""If the user's input is not a question, but a statement, pass the exact user input as the input for this function tool.""",
                fn_schema=MemorizeToolSchema,
            ),
        )

if __name__ == "__main__":
    local_llm = OpenAILike(
        api_base="http://localhost:1234/v1",
        timeout=600,  # secs
        temperature=0.01,
        api_key="loremIpsum",
        # Honestly, this model name can be arbitrary.
        # I'm using this: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta .
        model="mistralai_mistral-7b-instruct-v0.2",
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=32768,
    )

    service_context = ServiceContext.from_defaults(
        # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#local-embedding-models
        # HuggingFaceEmbedding requires transformers and PyTorch to be installed.
        # Run `pip install transformers torch`.
        embed_model="local",
        # https://docs.llamaindex.ai/en/stable/examples/llm/localai.html
        # But, instead of LocalAI, I'm using "LM Studio".
        llm=local_llm,
    )
    tool = make_tool(service_context)
    result = tool.call(input="My name is ABC")
    print("Using just the tool itself:", result)
    agent = ReActAgent.from_tools(
        tools=[tool],
        llm=local_llm,
        verbose=True,
    )
    result = agent.query("What's my name?")
    print("Using the tool via an agent:", result)
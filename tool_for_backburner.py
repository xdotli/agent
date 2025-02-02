"""
Agent-themed idea No. 1 - “when time gets right”
------------------------------------------------
**Problem statement.** Humans have “back burners”, but AI doesn’t.
You can ask someone “shall we do X?”, and they may say “hmm, potentially?? Y has to come true first before we can do X.” And you keep on your conversation. When Y becomes true in a moment, you human friend will say, “hey, now the time is right, and I think we can get our hands on X now.”

With a AI, if you ask “shall we do X?” and it thinks “Y is a prerequisite“, then it would either utterly refuse you, or enter an infinite loop of checking whether Y has become true yet. In the first case, the AI won’t raise an eyebrow even when the time gets right. In the 2nd case, you are stuck waiting for the AI to respond.

**Proposal.** Implement a meta-tool that, if it deems the time isn’t right to do something X yet, schedule a periodical task in the background. The task should check for the precondition (“Is Y true now?”). When the time has become right, perform the action X that was held off, and mention it in the conversation (“hey btw, Y has just become true, so I went ahead and did X as you asked.”)

**Here’s the tricky part:**
Y can become true as the conversation evolves. (“Hey, did you just say Z is true? You know what, that actually implies that Y is true, so I’ll go ahead and do X now.“)
This means a traditional, static cron job won’t cut it. The AI has to somehow update the context of each desired action X that was held off.
This also means the AI needs to know when to give up and remove X from its back burner. (“Dang it! Now that we realized that Y will never be the case, let’s forget about doing X for good.”)
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

from llama_index import Response, ServiceContext
from llama_index.llms import OpenAILike
from llama_index.tools import FunctionTool, ToolMetadata
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)


@dataclass
class BackburnerEntry:
    condition: str
    action: str
    action_input: str


backburner = []


class ConditionStatus(Enum):
    MET = 1
    NOT_YET_MET = 2
    WILL_NEVER_BE_MET = 3


from llama_index.agent import ReActAgent


def __tool_for_checking_the_weather(*args, **kwargs):
    return "It is sunny outside."


def create_agent_for_evaluating_conditions(
    service_context: ServiceContext,
) -> ReActAgent:
    all_tools = [
        FunctionTool(
            __tool_for_checking_the_weather,
            metadata=ToolMetadata(
                name="check_weather",
                description="Checks the weather.",
            ),
        )
    ]
    from llama_index.agent.react.formatter import ReActChatFormatter

    # Override the default system prompt for ReAct chats.
    with open("system_prompt_for_condition_evaluator.md") as f:
        MY_SYSTEM_PROMPT = f.read()

    class MyReActChatFormatter(ReActChatFormatter):
        system_header = MY_SYSTEM_PROMPT

    chat_formatter = MyReActChatFormatter()
    return ReActAgent.from_tools(
        tools=all_tools,
        llm=service_context.llm,
        verbose=True,
        react_chat_formatter=chat_formatter,
        callback_manager=service_context.callback_manager,
    )


def make_tools(service_context: ServiceContext) -> List:
    agent_for_evaluating_conditions = create_agent_for_evaluating_conditions(
        service_context
    )

    def put_on_backburner(condition: str, action: str, action_input: str):
        """
        When the time isn't yet right to do something, put it on the back burner by using this tool.

        Args:
        - condition: The condition that has to be true before the action can be performed.
        - action: The action that has to be performed.
        - action_input: The input to the action.
        """
        backburner.append(BackburnerEntry(condition, action, action_input))

    def evaluate_condition(condition: str) -> ConditionStatus:
        """
        Evaluate the condition and return its status.

        Args:
        - condition: The condition to be evaluated.

        Returns:
        - The status of the condition.
        """
        logger = logging.getLogger("evaluate_condition")
        # Invoke the Agent to use whatever tool it needs in order to evaluate the condition.
        response = agent_for_evaluating_conditions.query(
            f"Evaluate this condition: {condition}"
        )
        response = response.response
        logger.info(f"Condition evaluation response: {response}")
        if "will never be met" in response:
            return ConditionStatus.WILL_NEVER_BE_MET
        elif "not yet met" in response:
            return ConditionStatus.NOT_YET_MET
        else:
            return ConditionStatus.MET

    def perform_action(action: str, action_input: str):
        """
        Perform the action.

        Args:
        - action: The action to be performed.
        - action_input: The input to the action.

        Returns:
        - Whether the action was successful.
        """
        pass

    async def check_backburner():
        """
        Check if the conditions of any action on the back burner have become true. If so, perform the action and remove it from the back burner.
        """
        while True:
            for entry in backburner:
                condition_status = evaluate_condition(entry.condition)
                if condition_status == ConditionStatus.MET:
                    result = perform_action(entry.action, entry.action_input)
                    if result:
                        backburner.remove(entry)
                elif condition_status == ConditionStatus.WILL_NEVER_BE_MET:
                    backburner.remove(entry)
            await asyncio.sleep(1)

    class ConditionEvaluatingToolSchema(BaseModel):
        input: str

    return [
        FunctionTool(
            evaluate_condition,
            metadata=ToolMetadata(
                name="evaluate_condition",
                description="""Evaluates whether a given statement ("condition") has become true or not.""",
                fn_schema=ConditionEvaluatingToolSchema,
            ),
        )
    ]


if __name__ == "__main__":
    local_llm = OpenAILike(
        api_base="http://localhost:1234/v1",
        timeout=600,  # secs
        temperature=0.01,
        api_key="loremIpsum",
        # Honestly, this model name can be arbitrary.
        # I'm using this: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta .
        model="zephyr beta 7B q5_k_m gguf",
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
    tools = make_tools(service_context)
    result = tools[0].call("Is it sunny here in Santa Clara, CA?")
    print(result)

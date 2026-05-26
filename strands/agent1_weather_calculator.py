import asyncio

from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator

# Define a custom weather tool
@tool
def weather():
    """Get the weather"""
    return "sunny"


bedrock_model = BedrockModel(
    model_id="amazon.nova-micro-v1:0",
    region_name="us-east-1",
    temperature=0.3,
)

# Initialize the agent with tools
agent = Agent(
    tools=[calculator, weather],
    model=bedrock_model,
    system_prompt="You're a helpful assistant. You can perform simple math and tell the weather."
)

# response1 = agent("What is the weather today?")
# response2 = agent("how much is 2 + 2?")
# print(response1)
# print(response2)


async def main():
    stream = agent.stream_async(
        "What is 2 + 2 and what's the weather?"
    )

    async for event in stream:
        print(event)


asyncio.run(main())
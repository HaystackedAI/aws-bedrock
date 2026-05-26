import logging
from strands import Agent
from strands.models import BedrockModel

bedrock_model = BedrockModel(
    model_id="amazon.nova-micro-v1:0",
    region_name="us-east-1",
    temperature=0.3,
)

# Enable debug logging for Strands
logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# Create the Subject Expert agent
subject_expert = Agent(
    model=bedrock_model,
    system_prompt="""You are a Computer Science Subject Expert..."""
)

# Ask a question
response = subject_expert("Explain the concept of recursion in programming.")
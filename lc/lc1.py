import boto3
from langchain_aws import ChatBedrock

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

llm = ChatBedrock(
    client=bedrock_client,
    model="amazon.nova-lite-v1:0",
    model_kwargs={
        "temperature": 0.3,
        "maxTokens": 512,
    },
)

response = llm.invoke("What is the largest city in Vermont?")

print(response.content)
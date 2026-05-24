import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

prompt_data = """
        Command: Write an email from Bob, Customer Service Manager, AnyCompany to the customer "John Doe" 
        who provided negative feedback on the service provided by our customer support 
        engineer"""

response = client.converse(
    modelId="amazon.nova-micro-v1:0",
    messages=[
        {
            "role": "user",
            "content": [{"text": prompt_data}]
        }
    ],
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0
    }
)

print(
    response["output"]["message"]["content"][0]["text"]
)
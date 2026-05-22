import boto3
import json

prompt_data = """ write a poem on Generative AI"""

# Initialize the Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Define the payload for the model
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,  # Maximum length of generated text
    "temperature": 0.5,  # Controls the creativity of the output
    "top_p": 0.9         # Controls the diversity of the output
}

# Convert payload to JSON format
body = json.dumps(payload)

# Define the model ID (Meta's LLaMA 2 model in this case)
model_id = "meta.llama2-70b-chat-v1"

# Invoke the Bedrock model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Extract the response and print the generated text
response_body = json.loads(response.get("body").read())
response_text = response_body['generation']
print(response_text)
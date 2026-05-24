import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-east-1"
MODEL_IDS = [
    "amazon.nova-micro-v1:0",
    "us.amazon.nova-micro-v1:0",
]
PROMPT = "Reply with exactly: ok"


def main():
    client = boto3.client("bedrock-runtime", region_name=REGION)

    for model_id in MODEL_IDS:
        try:
            response = client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": PROMPT}],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 20,
                    "temperature": 0,
                },
            )
        except NoCredentialsError:
            print("FAIL: No AWS credentials found.")
            return 1
        except ClientError as error:
            details = error.response.get("Error", {})
            print(f"{model_id}: {details.get('Code')}: {details.get('Message')}")
            continue

        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content)

        print("PASS")
        print(f"Model: {model_id}")
        print(f"Response: {text}")
        return 0

    print("FAIL: No test model worked.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

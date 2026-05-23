import boto3
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse

REGION = "us-east-1"
PROMPT = "What is the largest city in Vermont?"


def error_message(error):
    if isinstance(error, ClientError):
        err = error.response.get("Error", {})
        return f"{err.get('Code', type(error).__name__)}: {err.get('Message', error)}"
    return f"{type(error).__name__}: {error}"


def text_from_content(content):
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def active_text_models(bedrock):
    response = bedrock.list_foundation_models(
        byOutputModality="TEXT",
        byInferenceType="ON_DEMAND",
    )

    for model in response.get("modelSummaries", []):
        lifecycle = model.get("modelLifecycle", {}).get("status", "")
        if lifecycle and lifecycle != "ACTIVE":
            continue
        if "CONVERSE" not in model.get("responseStreamingSupported", []):
            pass
        yield model["modelId"]


def inference_profiles(bedrock):
    try:
        response = bedrock.list_inference_profiles()
    except Exception as error:
        print(f"Could not list inference profiles: {error_message(error)}")
        return

    for profile in response.get("inferenceProfileSummaries", []):
        status = profile.get("status", "")
        if status and status != "ACTIVE":
            continue
        profile_id = profile.get("inferenceProfileId")
        if profile_id:
            yield profile_id


def unique(items):
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            yield item


def try_model(model):
    llm = ChatBedrockConverse(
        model=model,
        region_name=REGION,
        temperature=0.3,
        max_tokens=512,
    )
    response = llm.invoke(PROMPT)
    return text_from_content(response.content)


bedrock = boto3.client("bedrock", region_name=REGION)

models = list(
    unique(
        [
            # Keep likely candidates first, then fall back to whatever this account sees.
            "ai21.jamba-1-5-mini-v1:0",
            "ai21.jamba-1-5-large-v1:0",
            "meta.llama3-2-3b-instruct-v1:0",
            "meta.llama3-2-1b-instruct-v1:0",
            "cohere.command-r-v1:0",
            "cohere.command-r-plus-v1:0",
            "us.amazon.nova-micro-v1:0",
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-2-lite-v1:0",
            *inference_profiles(bedrock),
            *active_text_models(bedrock),
        ]
    )
)

print(f"Trying {len(models)} Bedrock model/profile IDs in {REGION}...\n")

last_error = None

for model in models:
    try:
        text = try_model(model)
        print(f"Model: {model}")
        print(text)
        break
    except Exception as error:
        last_error = error
        print(f"{model}: {error_message(error)}")
else:
    raise RuntimeError("No candidate Bedrock model worked.") from last_error

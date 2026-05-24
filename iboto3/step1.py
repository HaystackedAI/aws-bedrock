import boto3
from botocore.exceptions import ClientError

PROMPT = "What is the largest city in Vermont?"
REGIONS = ["us-east-1", "us-east-2", "us-west-2"]
MAX_FAILURES_PER_REGION = 20

PREFERRED_MODELS = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "ai21.jamba-1-5-mini-v1:0",
    "ai21.jamba-1-5-large-v1:0",
    "cohere.command-r-v1:0",
    "cohere.command-r-plus-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-2-lite-v1:0",
]


def error_message(error):
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        return f"{details.get('Code')}: {details.get('Message')}"
    return f"{type(error).__name__}: {error}"


def caller_identity():
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
    except Exception as error:
        print(f"Could not read caller identity: {error_message(error)}")
        return

    print("AWS identity")
    print("=" * 80)
    print(f"Account: {identity.get('Account')}")
    print(f"Arn:     {identity.get('Arn')}")
    print()


def unique(items):
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            yield item


def active_text_models(region):
    bedrock = boto3.client("bedrock", region_name=region)

    try:
        response = bedrock.list_foundation_models(
            byOutputModality="TEXT",
            byInferenceType="ON_DEMAND",
        )
    except Exception as error:
        print(f"{region}: could not list foundation models: {error_message(error)}")
        return []

    model_ids = []
    for model in response.get("modelSummaries", []):
        lifecycle = model.get("modelLifecycle", {}).get("status")
        if lifecycle and lifecycle != "ACTIVE":
            continue
        model_ids.append(model.get("modelId"))
    return model_ids


def active_inference_profiles(region):
    bedrock = boto3.client("bedrock", region_name=region)

    try:
        response = bedrock.list_inference_profiles()
    except Exception as error:
        print(f"{region}: could not list inference profiles: {error_message(error)}")
        return []

    profile_ids = []
    for profile in response.get("inferenceProfileSummaries", []):
        status = profile.get("status")
        if status and status != "ACTIVE":
            continue
        profile_ids.append(profile.get("inferenceProfileId"))
    return profile_ids


def candidate_models(region):
    return list(
        unique(
            [
                *PREFERRED_MODELS,
                *active_inference_profiles(region),
                *active_text_models(region),
            ]
        )
    )


def text_from_converse_response(response):
    parts = []
    message = response.get("output", {}).get("message", {})
    for block in message.get("content", []):
        text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def ask_model(region, model_id):
    runtime = boto3.client("bedrock-runtime", region_name=region)
    response = runtime.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": PROMPT}],
            }
        ],
        inferenceConfig={
            "temperature": 0.3,
            "maxTokens": 512,
        },
    )
    return text_from_converse_response(response)


last_error = None
operation_not_allowed_count = 0

caller_identity()

for region in REGIONS:
    models = candidate_models(region)
    print(f"\nTrying {len(models)} models/profiles in {region}...")

    failures_printed = 0

    for model_id in models:
        try:
            answer = ask_model(region, model_id)
        except Exception as error:
            last_error = error
            message = error_message(error)
            if "Operation not allowed" in message:
                operation_not_allowed_count += 1
            if failures_printed < MAX_FAILURES_PER_REGION:
                print(f"{region} | {model_id}: {message}")
                failures_printed += 1
            elif failures_printed == MAX_FAILURES_PER_REGION:
                print(f"{region}: suppressing additional failures...")
                failures_printed += 1
            continue

        print(f"\nRegion: {region}")
        print(f"Model: {model_id}")
        print(answer)
        raise SystemExit(0)

if operation_not_allowed_count:
    print()
    print("Every tested invocation was rejected by Bedrock.")
    print("This usually means the AWS identity above can list Bedrock models,")
    print("but is not allowed to invoke them, or model access is not enabled.")
    print()
    print("Check these in AWS:")
    print("1. Bedrock console -> Model access: enable at least one text model.")
    print("2. IAM policy for this identity: allow bedrock:InvokeModel.")
    print("3. If using inference profiles, allow the target US regions too:")
    print("   us-east-1, us-east-2, and us-west-2.")
    print()

raise RuntimeError("No candidate Bedrock model worked.") from last_error

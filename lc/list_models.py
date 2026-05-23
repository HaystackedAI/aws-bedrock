import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"


def print_error(label, error):
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        code = details.get("Code", type(error).__name__)
        message = details.get("Message", str(error))
        print(f"{label}: {code}: {message}")
    else:
        print(f"{label}: {type(error).__name__}: {error}")


def lifecycle_status(item):
    return item.get("modelLifecycle", {}).get("status", "")


def supported_inference_types(item):
    return ", ".join(item.get("inferenceTypesSupported", [])) or "-"


def supported_response_modalities(item):
    return ", ".join(item.get("outputModalities", [])) or "-"


def list_foundation_models(client):
    print(f"Foundation models in {REGION}")
    print("=" * 80)

    try:
        response = client.list_foundation_models()
    except Exception as error:
        print_error("Could not list foundation models", error)
        return

    models = sorted(
        response.get("modelSummaries", []),
        key=lambda item: (item.get("providerName", ""), item.get("modelId", "")),
    )

    if not models:
        print("No foundation models returned.")
        return

    for model in models:
        print(f"modelId: {model.get('modelId')}")
        print(f"  name: {model.get('modelName', '-')}")
        print(f"  provider: {model.get('providerName', '-')}")
        print(f"  lifecycle: {lifecycle_status(model) or '-'}")
        print(f"  outputs: {supported_response_modalities(model)}")
        print(f"  inference: {supported_inference_types(model)}")
        print(f"  streaming: {model.get('responseStreamingSupported', False)}")
        print()


def list_inference_profiles(client):
    print(f"Inference profiles in {REGION}")
    print("=" * 80)

    try:
        response = client.list_inference_profiles()
    except Exception as error:
        print_error("Could not list inference profiles", error)
        return

    profiles = sorted(
        response.get("inferenceProfileSummaries", []),
        key=lambda item: item.get("inferenceProfileId", ""),
    )

    if not profiles:
        print("No inference profiles returned.")
        return

    for profile in profiles:
        print(f"profileId: {profile.get('inferenceProfileId')}")
        print(f"  name: {profile.get('inferenceProfileName', '-')}")
        print(f"  status: {profile.get('status', '-')}")
        print(f"  type: {profile.get('type', '-')}")
        for model in profile.get("models", []):
            print(f"  modelArn: {model.get('modelArn')}")
        print()


def main():
    client = boto3.client("bedrock", region_name=REGION)
    list_foundation_models(client)
    print()
    list_inference_profiles(client)


if __name__ == "__main__":
    main()

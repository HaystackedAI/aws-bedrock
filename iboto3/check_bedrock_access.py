import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-east-1"

MODEL_IDS = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-2-lite-v1:0",
]


def aws_error(error):
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        return f"{details.get('Code')}: {details.get('Message')}"
    return f"{type(error).__name__}: {error}"


def print_identity():
    try:
        identity = boto3.client("sts").get_caller_identity()
    except NoCredentialsError:
        print("No AWS credentials found.")
        return
    except Exception as error:
        print(f"Could not get caller identity: {aws_error(error)}")
        return

    print("AWS identity")
    print("=" * 80)
    print(f"Account: {identity.get('Account')}")
    print(f"Arn:     {identity.get('Arn')}")
    print()


def check_model_availability(client, model_id):
    try:
        response = client.get_foundation_model_availability(modelId=model_id)
    except Exception as error:
        print(f"{model_id}")
        print(f"  check failed: {aws_error(error)}")
        print()
        return

    agreement = response.get("agreementAvailability", {})
    print(model_id)
    print(f"  agreement:     {agreement.get('status', '-')}")
    if agreement.get("errorMessage"):
        print(f"  agreement err: {agreement.get('errorMessage')}")
    print(f"  authorization: {response.get('authorizationStatus', '-')}")
    print(f"  entitlement:   {response.get('entitlementAvailability', '-')}")
    print(f"  region:        {response.get('regionAvailability', '-')}")
    print()


def main():
    print_identity()

    client = boto3.client("bedrock", region_name=REGION)
    print(f"Foundation model availability in {REGION}")
    print("=" * 80)

    for model_id in MODEL_IDS:
        check_model_availability(client, model_id)


if __name__ == "__main__":
    main()

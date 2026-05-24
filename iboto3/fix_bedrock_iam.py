import re
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"
BEDROCK_FULL_ACCESS_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"


def aws_error(error):
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        return f"{details.get('Code')}: {details.get('Message')}"
    return f"{type(error).__name__}: {error}"


def user_name_from_arn(arn):
    match = re.fullmatch(r"arn:aws:iam::\d+:user/(.+)", arn)
    if not match:
        return None
    return match.group(1)


def print_availability():
    bedrock = boto3.client("bedrock", region_name=REGION)
    response = bedrock.get_foundation_model_availability(modelId=MODEL_ID)
    agreement = response.get("agreementAvailability", {})

    print(f"{MODEL_ID} availability in {REGION}")
    print(f"  agreement:     {agreement.get('status', '-')}")
    print(f"  authorization: {response.get('authorizationStatus', '-')}")
    print(f"  entitlement:   {response.get('entitlementAvailability', '-')}")
    print(f"  region:        {response.get('regionAvailability', '-')}")
    return response.get("authorizationStatus") == "AUTHORIZED"


def main():
    try:
        identity = boto3.client("sts").get_caller_identity()
    except NoCredentialsError:
        print("No AWS credentials found.")
        return 1
    except Exception as error:
        print(f"Could not get caller identity: {aws_error(error)}")
        return 1

    arn = identity["Arn"]
    user_name = user_name_from_arn(arn)

    print(f"Caller: {arn}")
    if not user_name:
        print("This script only auto-fixes direct IAM users, not assumed roles.")
        print("Attach AmazonBedrockFullAccess to the role shown above instead.")
        return 1

    print(f"IAM user: {user_name}")
    print()

    try:
        if print_availability():
            print("Already authorized.")
            return 0
    except Exception as error:
        print(f"Could not check current Bedrock availability: {aws_error(error)}")

    print()
    print(f"Attaching {BEDROCK_FULL_ACCESS_POLICY_ARN}...")

    iam = boto3.client("iam")
    try:
        iam.attach_user_policy(
            UserName=user_name,
            PolicyArn=BEDROCK_FULL_ACCESS_POLICY_ARN,
        )
    except Exception as error:
        print(f"Could not attach policy: {aws_error(error)}")
        print()
        print("Attach this managed policy manually in IAM:")
        print(f"  {BEDROCK_FULL_ACCESS_POLICY_ARN}")
        return 1

    print("Policy attached. Waiting for IAM propagation...")

    for seconds in [5, 10, 20, 30]:
        time.sleep(seconds)
        print()
        try:
            if print_availability():
                print("Authorized now. Try: python unittest.py")
                return 0
        except Exception as error:
            print(f"Availability check failed: {aws_error(error)}")

    print()
    print("The policy is attached, but Bedrock still says NOT_AUTHORIZED.")
    print("At that point, check for an Organizations SCP, permissions boundary,")
    print("or account-level Bedrock restriction.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

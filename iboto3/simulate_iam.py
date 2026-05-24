import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"

ACTIONS = [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream",
]

RESOURCES = [
    "*",
    f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}",
]


def aws_error(error):
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        return f"{details.get('Code')}: {details.get('Message')}"
    return f"{type(error).__name__}: {error}"


def caller_arn():
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Arn"]


def main():
    try:
        arn = caller_arn()
    except NoCredentialsError:
        print("No AWS credentials found.")
        return
    except Exception as error:
        print(f"Could not get caller identity: {aws_error(error)}")
        return

    print(f"Simulating permissions for: {arn}")
    print()

    iam = boto3.client("iam")
    try:
        response = iam.simulate_principal_policy(
            PolicySourceArn=arn,
            ActionNames=ACTIONS,
            ResourceArns=RESOURCES,
        )
    except Exception as error:
        print(f"Could not simulate policy: {aws_error(error)}")
        print()
        print("If this is AccessDenied, check in the AWS console instead:")
        print("IAM user -> Permissions -> Policy simulator")
        return

    for result in response.get("EvaluationResults", []):
        action = result.get("EvalActionName")
        resource = result.get("EvalResourceName")
        decision = result.get("EvalDecision")
        print(f"{action}")
        print(f"  resource: {resource}")
        print(f"  decision: {decision}")

        matched = result.get("MatchedStatements", [])
        if matched:
            print("  matched statements:")
            for statement in matched:
                source = statement.get("SourcePolicyId", "-")
                effect = statement.get("Effect", "-")
                print(f"    {effect} from {source}")

        missing = result.get("MissingContextValues", [])
        if missing:
            print(f"  missing context: {', '.join(missing)}")
        print()


if __name__ == "__main__":
    main()

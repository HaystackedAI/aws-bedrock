import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"
MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}"
ACTIONS = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]


def aws_error(error):
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        return f"{details.get('Code')}: {details.get('Message')}"
    return f"{type(error).__name__}: {error}"


def user_name_from_arn(arn):
    marker = ":user/"
    if marker not in arn:
        return None
    return arn.split(marker, 1)[1]


def print_identity():
    identity = boto3.client("sts").get_caller_identity()
    print("AWS identity")
    print("=" * 80)
    print(f"Account: {identity.get('Account')}")
    print(f"Arn:     {identity.get('Arn')}")
    print()
    return identity


def print_model_availability():
    print(f"Bedrock availability for {MODEL_ID} in {REGION}")
    print("=" * 80)
    try:
        response = boto3.client("bedrock", region_name=REGION).get_foundation_model_availability(
            modelId=MODEL_ID
        )
    except Exception as error:
        print(f"Could not check availability: {aws_error(error)}")
        print()
        return

    agreement = response.get("agreementAvailability", {})
    print(f"agreement:     {agreement.get('status', '-')}")
    print(f"authorization: {response.get('authorizationStatus', '-')}")
    print(f"entitlement:   {response.get('entitlementAvailability', '-')}")
    print(f"region:        {response.get('regionAvailability', '-')}")
    print()


def print_user_permissions(iam, user_name):
    print(f"IAM user permissions for {user_name}")
    print("=" * 80)

    try:
        user = iam.get_user(UserName=user_name)["User"]
    except Exception as error:
        print(f"Could not get user: {aws_error(error)}")
        print()
        return

    boundary = user.get("PermissionsBoundary", {})
    if boundary:
        print(f"permissions boundary: {boundary.get('PermissionsBoundaryArn')}")
    else:
        print("permissions boundary: none")

    try:
        attached = iam.list_attached_user_policies(UserName=user_name).get("AttachedPolicies", [])
        print("attached user policies:")
        if attached:
            for policy in attached:
                print(f"  {policy.get('PolicyName')}: {policy.get('PolicyArn')}")
        else:
            print("  none")
    except Exception as error:
        print(f"could not list attached user policies: {aws_error(error)}")

    try:
        inline = iam.list_user_policies(UserName=user_name).get("PolicyNames", [])
        print("inline user policies:")
        if inline:
            for name in inline:
                print(f"  {name}")
        else:
            print("  none")
    except Exception as error:
        print(f"could not list inline user policies: {aws_error(error)}")

    try:
        groups = iam.list_groups_for_user(UserName=user_name).get("Groups", [])
        print("groups:")
        if groups:
            for group in groups:
                print(f"  {group.get('GroupName')}")
        else:
            print("  none")
    except Exception as error:
        print(f"could not list groups: {aws_error(error)}")

    print()


def print_simulation(iam, arn):
    print("IAM policy simulation")
    print("=" * 80)
    for resource in ["*", MODEL_ARN]:
        try:
            response = iam.simulate_principal_policy(
                PolicySourceArn=arn,
                ActionNames=ACTIONS,
                ResourceArns=[resource],
            )
        except Exception as error:
            print(f"Could not run IAM simulation for {resource}: {aws_error(error)}")
            print()
            continue

        for result in response.get("EvaluationResults", []):
            print(f"{result.get('EvalActionName')}")
            print(f"  resource: {result.get('EvalResourceName')}")
            print(f"  decision: {result.get('EvalDecision')}")

            statements = result.get("MatchedStatements", [])
            if statements:
                print("  matched statements:")
                for statement in statements:
                    print(
                        f"    {statement.get('Effect')} from "
                        f"{statement.get('SourcePolicyId', '-')}"
                    )

            missing = result.get("MissingContextValues", [])
            if missing:
                print(f"  missing context: {', '.join(missing)}")
            print()


def print_organization_info():
    print("AWS Organizations check")
    print("=" * 80)
    org = boto3.client("organizations")
    try:
        response = org.describe_organization()
    except Exception as error:
        print(f"Could not describe organization: {aws_error(error)}")
        print("If this is AccessDenied, check SCPs from the management account.")
        print()
        return

    organization = response.get("Organization", {})
    print(f"organization id: {organization.get('Id', '-')}")
    print(f"management account: {organization.get('MasterAccountId') or organization.get('ManagementAccountId', '-')}")
    print()


def main():
    try:
        identity = print_identity()
    except NoCredentialsError:
        print("No AWS credentials found.")
        return 1
    except Exception as error:
        print(f"Could not get caller identity: {aws_error(error)}")
        return 1

    print_model_availability()

    arn = identity.get("Arn", "")
    user_name = user_name_from_arn(arn)
    if not user_name:
        print("The current identity is not a direct IAM user.")
        print("Check policies on the role/session shown in the ARN above.")
        return 1

    iam = boto3.client("iam")
    print_user_permissions(iam, user_name)
    print_simulation(iam, arn)
    print_organization_info()

    print("What matters:")
    print("- IAM simulation must be allowed for bedrock:InvokeModel.")
    print("- Bedrock availability must show authorization: AUTHORIZED.")
    print("- If IAM simulation is allowed but Bedrock is NOT_AUTHORIZED, check SCPs,")
    print("  permissions boundaries, Bedrock account restrictions, or wait for propagation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

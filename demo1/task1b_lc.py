import os, sys, warnings, json
import boto3

from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")
module_path = ".."
sys.path.append(os.path.abspath(module_path))

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")


model_id = "meta.llama3-8b-instruct-v1:0"
model_kwargs = {
    "temperature": 0.1,
    "max_gen_len": 512,
    "top_p": 0.9,
}

chat_model = ChatBedrock(
    client=bedrock_client,
    model=model_id,
    model_kwargs=model_kwargs,)



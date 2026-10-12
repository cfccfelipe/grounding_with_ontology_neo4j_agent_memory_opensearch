import sys
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from src.config import AWS_REGION, BEDROCK_EMBEDDING_MODEL_ID, BEDROCK_LLM_MODEL_ID
from src.graph_client import GraphLabDriver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preflight")


def check_identity() -> bool:
    try:
        ident = boto3.client("sts", region_name=AWS_REGION).get_caller_identity()
        print(f"AWS account : {ident['Account']}")
        print(f"AWS arn     : {ident['Arn']}")
        print(f"AWS region  : {AWS_REGION}")
        return True
    except (NoCredentialsError, ClientError) as exc:
        print(f"[FAIL] AWS credentials not usable: {exc}")
        print("       Fix: configure your default profile (e.g. `ada credentials update` / isengardcli).")
        return False


def check_bedrock_embeddings() -> bool:
    try:
        lab = GraphLabDriver()
        vec = lab.get_embedding("preflight embedding check")
        lab.close()
        print(f"[OK]  Bedrock embeddings ({BEDROCK_EMBEDDING_MODEL_ID}) -> dim {len(vec)}")
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        print(f"[FAIL] Bedrock embedding call failed ({code}): {exc}")
        if code in ("AccessDeniedException", "ValidationException"):
            print("       Enable model access in the Bedrock console (Model access) for Titan Text Embeddings V2.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Embedding preflight error: {exc}")
        return False


def check_bedrock_llm() -> bool:
    try:
        rt = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        rt.converse(
            modelId=BEDROCK_LLM_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}],
            inferenceConfig={"maxTokens": 8, "temperature": 0},
        )
        print(f"[OK]  Bedrock LLM ({BEDROCK_LLM_MODEL_ID}) reachable via Converse")
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        print(f"[FAIL] Bedrock LLM call failed ({code}): {exc}")
        if code in ("AccessDeniedException", "ValidationException"):
            print("       Enable model access for the LLM, or set BEDROCK_LLM_MODEL_ID to a model you have")
            print("       (inference-profile IDs like us.anthropic... or us.amazon.nova-pro-v1:0).")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] LLM preflight error: {exc}")
        return False


if __name__ == "__main__":
    print("\n=== Preflight: AWS + Bedrock ===")
    ok = check_identity()
    ok = check_bedrock_embeddings() and ok
    ok = check_bedrock_llm() and ok
    print("=== Preflight PASSED ===\n" if ok else "=== Preflight FAILED ===\n")
    sys.exit(0 if ok else 1)

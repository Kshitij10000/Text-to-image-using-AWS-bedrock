import boto3
import json
import base64
import os

# Initialize the Bedrock client with the correct service name
bedrock = boto3.client(service_name="bedrock-runtime")

# Define the payload according to the Titan image generator API specification
payload = {
    "textToImageParams": {
        "text": "blue backpack on a table"
    },
    "taskType": "TEXT_IMAGE",
    "imageGenerationConfig": {
        "cfgScale": 8,
        "seed": 0,
        "quality": "standard",
        "width": 1024,
        "height": 1024,
        "numberOfImages": 1
    }
}

# Convert the payload to JSON
body = json.dumps(payload)

# Set the model ID as specified
model_id = "amazon.titan-image-generator-v1"

# Invoke the model with the prepared parameters
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
print(response_body)
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save image to a file in the output directory.
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)

from unsloth import FastLanguageModel
import torch
import os

# Print current working directory to help locate model files
print(f"Current working directory: {os.getcwd()}")

# Model configuration
max_seq_length = 2048
load_in_4bit = True

# Path to your saved fine-tuned model - Updated to use your specific path
model_path = "E:/DS_8b/DeepSeek-R1-Medical-COT_910"  # Using the path you provided

print(f"Loading model from: {model_path}")

# Load the fine-tuned model and tokenizer
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Set the model to inference mode
FastLanguageModel.for_inference(model)
print("Model prepared for inference")

# Setup prompt template similar to what was used during training
test_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
"""

# Test question (dermatology description)
test_question = "根据检查所见：外周黄褐色色素沉着，中央白色瘢痕样斑片，伴点状及线状血管。会得到怎么样的一个结果"

print("\nTesting model with question:")
print(test_question)
print("\nGenerating response...")

# Format the question
formatted_input = test_prompt_template.format(test_question)

# Tokenize input
inputs = tokenizer([formatted_input], return_tensors="pt")

# Move inputs to the same device as the model
if torch.cuda.is_available():
    device = "cuda"
    inputs = inputs.to(device)
    model = model.to(device)
    print("Using CUDA for inference")
else:
    device = "cpu"
    print("Using CPU for inference (this might be slow)")

# Generate response
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    temperature=0.7,
    top_p=0.9,
    use_cache=True,
)

# Decode the response
full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Extract only the generated part (after the prompt)
response_only = full_response.split("### Response:")[1].strip()

print("\n============= MODEL RESPONSE =============")
print(response_only)
print("==========================================")
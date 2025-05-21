import json
from openai import OpenAI

# Initialize the DeepSeek API client
client = OpenAI(api_key="sk-47d52774a5b9404da946f51e5c27d58a", base_url="https://api.deepseek.com")


def generate_complex_cot(question, response):
    # Construct the prompt for the complex reasoning
    prompt = f"""
    你是一名临床医学专家，擅长根据病史和症状进行医学推理。请阅读以下问题并回答：

    问题：{question}
    回答：{response}

    根据上述问题和回答，生成详细的推理过程，不要直接给出答案。
    生成推理链：
    """

    # Calling DeepSeek's API to generate the reasoning process
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    # Return the generated complex reasoning chain
    return response.choices[0].message.content.strip()


def process_and_generate_cot(input_json_file, output_json_file):
    # Load the data from the existing JSON file
    with open(input_json_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # Create a list to store the results
    results = []

    # Iterate over each entry in the all_data dictionary
    for key, value in all_data.items():
        question = value["question"]
        response = value["response"]

        # Generate the complex chain of thought
        complex_cot = generate_complex_cot(question, response)

        # Prepare the result for this entry
        result = {
            "Question": question,
            "Complex_CoT": complex_cot,
            "Response": response
        }

        # Append the result to the results list
        results.append(result)

    # Save all results to a new JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_json_file}")


# Example usage
input_json_file = "output.json"  # The input JSON file containing questions and responses
output_json_file = "medical_data_with_cot_reasoner.json"  # The output JSON file with the complex chain of thought

# Process the input file and generate the COT for each question-response pair
process_and_generate_cot(input_json_file, output_json_file)


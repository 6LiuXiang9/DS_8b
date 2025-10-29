import json
from openai import OpenAI

# 初始化 DeepSeek API 客户端
client = OpenAI(
    api_key="sk-ae7eb1639bfb4a2999bc83becfb65",  # 
    base_url="https://api.deepseek.com"
)

def generate_complex_cot(question: str, response: str) -> str:
    """
    调用教师模型生成医学推理链（Chain of Thought）
    """
    prompt = f"""
你是一名临床医学专家，擅长根据医学检查报告中的“观察所见”和“提示”信息进行推理分析。
请根据以下输入生成详细的医学推理链，仅做学术分析，不要直接给出最终诊断或治疗建议。

【输入信息】
- 观察/提示：{question}
- 最终参考回答（医生结论）：{response}

【任务要求】
1. Step 1. 分析报告中描述的影像或皮肤镜所见（逐项解读）。
2. Step 2. 解释这些表现所提示的组织或病理学改变及其医学意义。
3. Step 3. 结合这些改变，推断可能涉及的常见疾病类型或病理过程（仅作为病理推断，不给出最终诊断）。
4. Step 4. 说明本次推理的局限性（例如：仅基于影像描述，缺乏临床病史或病理学信息，因而可靠性有限），
       但不要向外部提出补充信息请求，也不要与用户进行对话。

注意：输出必须用“Step 1 … Step 2 … Step 3 … Step 4 …”分条列出，逻辑清晰、专业，避免任何问答或对话式表述。
    """

    try:
        response_obj = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in medical reasoning."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,   # 输出稳定
            stream=False
        )
        return response_obj.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] Failed to generate CoT: {e}")
        return ""

def process_and_generate_cot(input_json_file: str, output_json_file: str):
    """
    读取输入文件，调用教师模型生成推理链，保存为新 JSON
    """
    with open(input_json_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    results = []
    success_count = 0
    fail_count = 0

    for key, value in all_data.items():
        question = value.get("question", "")
        response = value.get("response", "")

        if not question or not response:
            print(f"[Warning] Entry {key} missing question or response, skipped.")
            fail_count += 1
            continue

        print(f"[Processing] {key} ...")
        complex_cot = generate_complex_cot(question, response)

        if complex_cot:
            results.append({
                "Question": question,
                "Complex_CoT": complex_cot,
                "Response": response
            })
            success_count += 1
        else:
            fail_count += 1

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Data saved to {output_json_file}")
    print(f"成功生成: {success_count} 条 | 失败: {fail_count} 条")

# ===== 示例运行 =====
if __name__ == "__main__":
    input_json_file = "output_4010_newprompt.json"  # 输入数据
    output_json_file = "medical_data_with_cot_reasoner_4010_newprompt-forstep.json"  # 输出结果
    process_and_generate_cot(input_json_file, output_json_file)

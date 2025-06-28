from unsloth import FastLanguageModel
import torch
import os
import json
import random
from bert_score import BERTScorer
import pandas as pd
from tqdm import tqdm
import numpy as np
import time

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Model configuration
max_seq_length = 2048
load_in_4bit = True

# Paths - 根据你的实际路径修正
model_path = "E:/DS_8b/DeepSeek-R1-Medical-COT_710"  # 修正为实际模型路径
test_data_path = "Test_medical_data_with_cot_reasoner_100.json"  # 当前目录下的测试数据

print(f"Loading model from: {model_path}")
print(f"Loading test data from: {test_data_path}")

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"❌ Model path does not exist: {model_path}")
    exit(1)

if not os.path.exists(test_data_path):
    print(f"❌ Test data path does not exist: {test_data_path}")
    exit(1)

# Load the fine-tuned model and tokenizer
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Set the model to inference mode
FastLanguageModel.for_inference(model)
print("✅ Model prepared for inference")

# Load test dataset
try:
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Loaded test dataset: {len(test_data)} samples")
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    exit(1)

# Setup prompt template (consistent with training)
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

# Initialize BERTScore for Chinese - 修复版本兼容性问题
print("Initializing BERTScorer...")
scorer = None

# 尝试多种初始化方式
init_methods = [
    # 方法1: 使用lang参数
    lambda: BERTScorer(lang='zh'),
    # 方法2: 使用model_type参数
    lambda: BERTScorer(model_type='bert-base-chinese'),
    # 方法3: 使用默认中文模型
    lambda: BERTScorer(model_type='hfl/chinese-bert-wwm-ext'),
    # 方法4: 最基本的初始化
    lambda: BERTScorer()
]

for i, init_method in enumerate(init_methods):
    try:
        print(f"尝试初始化方法 {i + 1}...")
        scorer = init_method()
        print(f"✅ BERTScorer initialized successfully with method {i + 1}")
        break
    except Exception as e:
        print(f"❌ Method {i + 1} failed: {e}")
        continue

if scorer is None:
    print("❌ All BERTScorer initialization methods failed")
    print("请检查bert-score包的安装: pip install bert-score")
    exit(1)

# Setup device
if torch.cuda.is_available():
    device = "cuda"
    model = model.to(device)
    print("✅ Using CUDA for inference")
else:
    device = "cpu"
    print("⚠️ Using CPU for inference (this might be slow)")


def generate_model_response(question, max_retries=3):
    """生成模型回答，带重试机制"""
    for attempt in range(max_retries):
        try:
            formatted_input = test_prompt_template.format(question)
            inputs = tokenizer([formatted_input], return_tensors="pt")

            if device == "cuda":
                inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=1200,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response_only = full_response.split("### Response:")[1].strip()

            # 处理CoT思维链格式
            if "</think>" in response_only:
                # 提取思维链和最终回答
                parts = response_only.split("</think>")
                if len(parts) > 1:
                    think_part = parts[0].replace("<think>", "").strip()
                    final_answer = parts[1].strip()
                    return {
                        "full_response": response_only,
                        "thinking": think_part,
                        "final_answer": final_answer
                    }

            # 如果没有思维链格式，返回整个回答
            return {
                "full_response": response_only,
                "thinking": "",
                "final_answer": response_only
            }

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return {
                    "full_response": f"Error generating response: {e}",
                    "thinking": "",
                    "final_answer": f"Error: {e}"
                }
            time.sleep(1)


def safe_bert_score(candidates, references, max_retries=3):
    """安全的BERTScore计算，带重试机制"""
    for attempt in range(max_retries):
        try:
            # 清理输入文本
            clean_candidates = [str(c).strip() if c else "" for c in candidates]
            clean_references = [str(r).strip() if r else "" for r in references]

            # 如果任何一个为空，返回默认分数
            if not any(clean_candidates) or not any(clean_references):
                return [0.0] * len(candidates), [0.0] * len(candidates), [0.0] * len(candidates)

            P, R, F1 = scorer.score(clean_candidates, clean_references)
            return P, R, F1

        except Exception as e:
            print(f"⚠️ BERTScore calculation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("使用默认分数...")
                return [0.0] * len(candidates), [0.0] * len(candidates), [0.0] * len(candidates)
            time.sleep(1)


def calculate_comprehensive_bertscore(model_output, reference_data):
    """
    计算全面的BERTScore评估：
    1. Response BERTScore (模型最终回答 vs 参考回答)
    2. CoT BERTScore (模型思维链 vs 参考思维链)
    3. Combined BERTScore-L (模型完整回答 vs 参考完整内容)
    """

    # 准备数据
    model_response = model_output["final_answer"] or ""
    model_cot = model_output["thinking"] or ""
    model_combined = f"{model_cot} {model_response}".strip()

    reference_response = reference_data["Response"] or ""
    reference_cot = reference_data.get("Complex_CoT", "") or ""
    reference_combined = f"{reference_cot} {reference_response}".strip()

    scores = {}

    try:
        # 1. Response BERTScore
        if model_response and reference_response:
            P_resp, R_resp, F1_resp = safe_bert_score([model_response], [reference_response])
            scores['response'] = {
                'precision': float(P_resp[0]) if len(P_resp) > 0 else 0.0,
                'recall': float(R_resp[0]) if len(R_resp) > 0 else 0.0,
                'f1': float(F1_resp[0]) if len(F1_resp) > 0 else 0.0
            }
        else:
            scores['response'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # 2. CoT BERTScore
        if model_cot and reference_cot:
            P_cot, R_cot, F1_cot = safe_bert_score([model_cot], [reference_cot])
            scores['cot'] = {
                'precision': float(P_cot[0]) if len(P_cot) > 0 else 0.0,
                'recall': float(R_cot[0]) if len(R_cot) > 0 else 0.0,
                'f1': float(F1_cot[0]) if len(F1_cot) > 0 else 0.0
            }
        else:
            # 如果CoT为空，给予默认分数
            scores['cot'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # 3. Combined BERTScore-L (整体评估)
        if model_combined and reference_combined:
            P_comb, R_comb, F1_comb = safe_bert_score([model_combined], [reference_combined])
            scores['combined'] = {
                'precision': float(P_comb[0]) if len(P_comb) > 0 else 0.0,
                'recall': float(R_comb[0]) if len(R_comb) > 0 else 0.0,
                'f1': float(F1_comb[0]) if len(F1_comb) > 0 else 0.0
            }
        else:
            scores['combined'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    except Exception as e:
        print(f"❌ Error calculating comprehensive BERTScore: {e}")
        # 返回默认分数
        scores = {
            'response': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'cot': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'combined': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        }

    return scores


def evaluate_model_on_test_set(num_samples=None, random_sample=False):
    """在测试集上评估模型性能"""

    # 确定测试样本
    if num_samples is None:
        test_samples = test_data
        print(f"📊 使用完整测试集进行评估: {len(test_samples)} 个样本")
    else:
        if random_sample:
            test_samples = random.sample(test_data, min(num_samples, len(test_data)))
            print(f"📊 随机选择 {len(test_samples)} 个样本进行评估")
        else:
            test_samples = test_data[:num_samples]
            print(f"📊 使用前 {len(test_samples)} 个样本进行评估")

    results = []

    # 用于收集所有分数以计算平均值
    all_response_scores = []
    all_cot_scores = []
    all_combined_scores = []

    print("\n🚀 开始生成模型回答并计算BERTScore...")

    for i, sample in enumerate(tqdm(test_samples, desc="评估进行中", ncols=100)):
        try:
            question = sample["Question"]
            reference_response = sample["Response"]
            reference_cot = sample.get("Complex_CoT", "")

            # 生成模型回答
            model_output = generate_model_response(question)

            # 计算三种BERTScore
            bert_scores = calculate_comprehensive_bertscore(model_output, sample)

            # 收集分数用于统计
            all_response_scores.append(bert_scores['response']['f1'])
            all_cot_scores.append(bert_scores['cot']['f1'])
            all_combined_scores.append(bert_scores['combined']['f1'])

            # 保存详细结果
            result = {
                'sample_id': i,
                'question': question,
                'reference_response': reference_response,
                'reference_cot': reference_cot,
                'model_full_response': model_output["full_response"],
                'model_thinking': model_output["thinking"],
                'model_final_answer': model_output["final_answer"],

                # Response BERTScore
                'response_bert_precision': bert_scores['response']['precision'],
                'response_bert_recall': bert_scores['response']['recall'],
                'response_bert_f1': bert_scores['response']['f1'],

                # CoT BERTScore
                'cot_bert_precision': bert_scores['cot']['precision'],
                'cot_bert_recall': bert_scores['cot']['recall'],
                'cot_bert_f1': bert_scores['cot']['f1'],

                # Combined BERTScore-L
                'combined_bert_precision': bert_scores['combined']['precision'],
                'combined_bert_recall': bert_scores['combined']['recall'],
                'combined_bert_f1': bert_scores['combined']['f1'],
            }

            results.append(result)

            # 每10个样本显示一次进度
            if (i + 1) % 10 == 0:
                avg_resp = np.mean([r['response_bert_f1'] for r in results])
                avg_cot = np.mean([r['cot_bert_f1'] for r in results])
                avg_comb = np.mean([r['combined_bert_f1'] for r in results])
                print(
                    f"\n📊 已处理 {i + 1} 个样本 - 当前平均F1: Response={avg_resp:.3f}, CoT={avg_cot:.3f}, Combined={avg_comb:.3f}")

        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            continue

    print(f"✅ 成功评估 {len(results)} 个样本")

    if len(results) == 0:
        print("❌ 没有成功评估任何样本，评估终止")
        return None, None

    # 计算三个维度的统计信息
    stats = calculate_comprehensive_statistics(all_response_scores, all_cot_scores, all_combined_scores)

    return results, stats


def calculate_comprehensive_statistics(response_scores, cot_scores, combined_scores):
    """计算三个维度的详细统计信息"""

    def calc_stats(scores):
        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'q75': float(np.percentile(scores_array, 75))
        }

    stats = {
        'total_samples': len(response_scores),
        'response_f1_stats': calc_stats(response_scores),
        'cot_f1_stats': calc_stats(cot_scores),
        'combined_f1_stats': calc_stats(combined_scores),

        # 平均分数（关键指标）
        'average_scores': {
            'response_f1_mean': float(np.mean(response_scores)),
            'cot_f1_mean': float(np.mean(cot_scores)),
            'combined_f1_mean': float(np.mean(combined_scores))
        }
    }

    return stats


def analyze_performance_by_categories(results):
    """按不同类别分析性能"""
    if not results:
        return

    def categorize_score(score):
        if score >= 0.8:
            return 'Excellent (≥0.8)'
        elif score >= 0.6:
            return 'Good (0.6-0.8)'
        elif score >= 0.4:
            return 'Fair (0.4-0.6)'
        else:
            return 'Poor (<0.4)'

    categories = {
        'Response': {'Excellent (≥0.8)': 0, 'Good (0.6-0.8)': 0, 'Fair (0.4-0.6)': 0, 'Poor (<0.4)': 0},
        'CoT': {'Excellent (≥0.8)': 0, 'Good (0.6-0.8)': 0, 'Fair (0.4-0.6)': 0, 'Poor (<0.4)': 0},
        'Combined': {'Excellent (≥0.8)': 0, 'Good (0.6-0.8)': 0, 'Fair (0.4-0.6)': 0, 'Poor (<0.4)': 0}
    }

    for result in results:
        # Response分类
        resp_category = categorize_score(result['response_bert_f1'])
        categories['Response'][resp_category] += 1

        # CoT分类
        cot_category = categorize_score(result['cot_bert_f1'])
        categories['CoT'][cot_category] += 1

        # Combined分类
        comb_category = categorize_score(result['combined_bert_f1'])
        categories['Combined'][comb_category] += 1

    print("\n📊 三维度性能分布分析:")
    print("=" * 80)

    for dimension, cats in categories.items():
        print(f"\n🎯 {dimension} BERTScore F1 分布:")
        print("-" * 50)
        total = sum(cats.values())
        for category, count in cats.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {category}: {count:3d} 样本 ({percentage:5.1f}%)")


def display_comprehensive_results(results, stats):
    """显示全面的评估结果"""
    print("\n" + "=" * 80)
    print("🎯 DeepSeek-R1-Medical-COT_710 全面评估结果")
    print("=" * 80)

    if stats:
        print(f"📊 总样本数: {stats['total_samples']}")

        # 显示三个维度的平均F1分数
        avg_scores = stats['average_scores']
        print(f"\n🏆 平均F1分数总览:")
        print(f"{'维度':<15} {'平均F1分数':<12} {'表现等级'}")
        print("-" * 45)

        def get_performance_level(score):
            if score >= 0.8:
                return "优秀 ⭐⭐⭐"
            elif score >= 0.6:
                return "良好 ⭐⭐"
            elif score >= 0.4:
                return "一般 ⭐"
            else:
                return "需改进"

        print(
            f"{'Response':<15} {avg_scores['response_f1_mean']:<12.4f} {get_performance_level(avg_scores['response_f1_mean'])}")
        print(f"{'CoT':<15} {avg_scores['cot_f1_mean']:<12.4f} {get_performance_level(avg_scores['cot_f1_mean'])}")
        print(
            f"{'Combined':<15} {avg_scores['combined_f1_mean']:<12.4f} {get_performance_level(avg_scores['combined_f1_mean'])}")

        # 详细统计信息
        print(f"\n📈 详细统计信息:")
        print(f"{'维度':<12} {'均值':<8} {'标准差':<8} {'中位数':<8} {'最小值':<8} {'最大值':<8}")
        print("-" * 65)

        dimensions = [
            ('Response', stats['response_f1_stats']),
            ('CoT', stats['cot_f1_stats']),
            ('Combined', stats['combined_f1_stats'])
        ]

        for dim_name, dim_stats in dimensions:
            print(f"{dim_name:<12} "
                  f"{dim_stats['mean']:.4f}   "
                  f"{dim_stats['std']:.4f}   "
                  f"{dim_stats['median']:.4f}   "
                  f"{dim_stats['min']:.4f}   "
                  f"{dim_stats['max']:.4f}")

    # 性能分布分析
    if results:
        analyze_performance_by_categories(results)


def save_comprehensive_results(results, stats, filename="comprehensive_evaluation_results_01.json"):
    """保存全面的评估结果"""
    output_data = {
        "evaluation_info": {
            "model_path": model_path,
            "test_data_path": test_data_path,
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_samples": len(test_data),
            "evaluated_samples": len(results) if results else 0,
            "evaluation_dimensions": ["Response", "CoT", "Combined"]
        },
        "summary_statistics": stats,
        "detailed_results": results
    }

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 全面评估结果已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存结果时出错: {e}")


def create_comprehensive_summary_report(stats):
    """创建全面的评估摘要报告"""
    if not stats:
        return

    avg_scores = stats['average_scores']

    summary = f"""
📋 DeepSeek-R1-Medical-COT_710 全面评估摘要报告
{'=' * 60}

📊 评估概况:
   - 测试样本数: {stats['total_samples']}
   - 评估维度: Response, CoT, Combined
   - 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

🎯 三维度平均F1分数:
   - Response F1:  {avg_scores['response_f1_mean']:.4f}
   - CoT F1:       {avg_scores['cot_f1_mean']:.4f}  
   - Combined F1:  {avg_scores['combined_f1_mean']:.4f}

📈 整体性能评估:
   - Response: {'优秀' if avg_scores['response_f1_mean'] >= 0.8 else '良好' if avg_scores['response_f1_mean'] >= 0.6 else '需改进'}
   - CoT: {'优秀' if avg_scores['cot_f1_mean'] >= 0.8 else '良好' if avg_scores['cot_f1_mean'] >= 0.6 else '需改进'}
   - Combined: {'优秀' if avg_scores['combined_f1_mean'] >= 0.8 else '良好' if avg_scores['combined_f1_mean'] >= 0.6 else '需改进'}

💡 改进建议:
   - Response表现: {'继续保持' if avg_scores['response_f1_mean'] >= 0.7 else '增强回答质量训练'}
   - CoT表现: {'继续保持' if avg_scores['cot_f1_mean'] >= 0.7 else '加强思维链训练'}
   - 整体表现: {'模型表现优异，可投入使用' if avg_scores['combined_f1_mean'] >= 0.75 else '建议继续优化训练'}

🔍 关键发现:
   - 最强项: {'Response' if avg_scores['response_f1_mean'] == max(avg_scores.values()) else 'CoT' if avg_scores['cot_f1_mean'] == max(avg_scores.values()) else 'Combined'}
   - 最需要改进: {'Response' if avg_scores['response_f1_mean'] == min(avg_scores.values()) else 'CoT' if avg_scores['cot_f1_mean'] == min(avg_scores.values()) else 'Combined'}
"""

    print(summary)

    # 保存摘要报告
    try:
        with open("comprehensive_evaluation_summary_01.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        print("📄 全面摘要报告已保存到: comprehensive_evaluation_summary_01.txt")
    except Exception as e:
        print(f"❌ 保存摘要报告时出错: {e}")


# 主评估流程
if __name__ == "__main__":
    print("\n🎯 开始全面评估 DeepSeek-R1-Medical-COT_710 模型")
    print("📝 三维度BERTScore评估: Response + CoT + Combined")
    print("=" * 80)

    # 用户选择评估规模
    print("\n📊 评估选项:")
    print("1. 完整测试集评估 (所有100个样本)")
    print("2. 快速评估 (随机30个样本)")
    print("3. 小规模测试 (前10个样本)")
    print("4. 自定义样本数量")

    choice = input("\n请选择评估方式 (1/2/3/4): ").strip()

    if choice == "1":
        print("🚀 开始完整测试集评估...")
        results, stats = evaluate_model_on_test_set()

    elif choice == "2":
        print("🚀 开始快速评估...")
        results, stats = evaluate_model_on_test_set(num_samples=30, random_sample=True)

    elif choice == "3":
        print("🚀 开始小规模测试...")
        results, stats = evaluate_model_on_test_set(num_samples=10, random_sample=False)

    elif choice == "4":
        try:
            num_samples = int(input("请输入样本数量: "))
            random_choice = input("是否随机选择样本? (y/n): ").strip().lower() == 'y'
            print(f"🚀 开始自定义评估 ({num_samples} 个样本)...")
            results, stats = evaluate_model_on_test_set(num_samples=num_samples, random_sample=random_choice)
        except ValueError:
            print("❌ 输入无效，使用默认快速评估...")
            results, stats = evaluate_model_on_test_set(num_samples=30, random_sample=True)

    else:
        print("❌ 选择无效，使用默认快速评估...")
        results, stats = evaluate_model_on_test_set(num_samples=30, random_sample=True)

    # 显示和保存结果
    if results and stats:
        display_comprehensive_results(results, stats)
        save_comprehensive_results(results, stats)
        create_comprehensive_summary_report(stats)

        # 显示最终的三维度平均F1分数
        avg_scores = stats['average_scores']
        print(f"\n🎯 最终三维度平均F1分数:")
        print(f"Response F1 平均分: {avg_scores['response_f1_mean']:.4f}")
        print(f"CoT F1 平均分:      {avg_scores['cot_f1_mean']:.4f}")
        print(f"Combined F1 平均分: {avg_scores['combined_f1_mean']:.4f}")

    else:
        print("❌ 评估失败，请检查模型和数据路径")

    print("\n✅ 全面评估完成！")
    print("\n📋 评估文件输出:")
    print("   - comprehensive_evaluation_results_01.json (详细结果)")
    print("   - comprehensive_evaluation_summary_01.txt (摘要报告)")
    print("\n🎯 三维度F1平均分数已计算并展示")

    # 等待用户按键结束
    input("\n按任意键结束程序...")
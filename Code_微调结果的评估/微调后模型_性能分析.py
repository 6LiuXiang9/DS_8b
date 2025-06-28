from unsloth import FastLanguageModel
import torch
import os
import time
import psutil
import threading
from transformers import TextIteratorStreamer
from threading import Thread
from collections import deque
import math

# 尝试导入GPU监控库
try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    HAS_PYNVML = False
    gpu_handle = None


class AdvancedPerformanceMonitor:
    """高级性能监控器 - 包含推理速度、回复速度和内存峰值的精确计算"""

    def __init__(self):
        self.start_time = None
        self.metrics = []
        self.monitoring = False

        # 速度计算相关
        self.inference_start_time = None
        self.first_token_time = None
        self.generation_end_time = None

        # Token计数
        self.input_tokens = 0
        self.output_tokens = 0
        self.real_time_tokens = 0

        # 实时速度跟踪
        self.token_timestamps = []
        self.response_chunks = []

        # 内存峰值跟踪
        self.memory_baseline = None
        self.system_memory_peak = 0
        self.gpu_memory_peak = 0
        self.pytorch_memory_peak = 0
        self.memory_snapshots = []

    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.monitoring = True
        self.metrics = []
        self.token_timestamps = []
        self.response_chunks = []
        self.memory_snapshots = []

        # 记录内存基线
        self.record_memory_baseline()

    def record_memory_baseline(self):
        """记录内存使用基线"""
        # 系统内存
        memory = psutil.virtual_memory()
        self.memory_baseline = {
            'system_memory_used': memory.used / 1024 ** 3,  # GB
            'system_memory_percent': memory.percent,
        }

        # GPU内存
        if torch.cuda.is_available():
            self.memory_baseline['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024 ** 3
            self.memory_baseline['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024 ** 3

            if HAS_PYNVML and gpu_handle:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    self.memory_baseline['gpu_memory_used'] = mem_info.used / 1024 ** 3
                    self.memory_baseline['gpu_memory_total'] = mem_info.total / 1024 ** 3
                except:
                    pass

        print(f"[内存基线] 系统内存: {self.memory_baseline['system_memory_used']:.2f}GB "
              f"({self.memory_baseline['system_memory_percent']:.1f}%)")
        if torch.cuda.is_available():
            print(f"[内存基线] GPU内存: {self.memory_baseline.get('gpu_memory_allocated', 0):.2f}GB")

    def update_memory_peaks(self):
        """更新内存峰值记录"""
        # 系统内存
        memory = psutil.virtual_memory()
        current_system_memory = memory.used / 1024 ** 3
        self.system_memory_peak = max(self.system_memory_peak, current_system_memory)

        # GPU内存
        if torch.cuda.is_available():
            current_pytorch_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            current_pytorch_reserved = torch.cuda.memory_reserved() / 1024 ** 3

            self.pytorch_memory_peak = max(self.pytorch_memory_peak, current_pytorch_allocated)

            if HAS_PYNVML and gpu_handle:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    current_gpu_memory = mem_info.used / 1024 ** 3
                    self.gpu_memory_peak = max(self.gpu_memory_peak, current_gpu_memory)
                except:
                    pass

        # 记录详细快照 (每秒记录一次以避免过多数据)
        current_time = time.time()
        if not self.memory_snapshots or (current_time - self.memory_snapshots[-1]['timestamp']) >= 1.0:
            snapshot = {
                'timestamp': current_time - self.start_time,
                'system_memory': current_system_memory,
                'system_memory_percent': memory.percent,
            }

            if torch.cuda.is_available():
                snapshot['pytorch_allocated'] = current_pytorch_allocated
                snapshot['pytorch_reserved'] = current_pytorch_reserved

                if HAS_PYNVML and gpu_handle:
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                        snapshot['gpu_memory_used'] = mem_info.used / 1024 ** 3
                        snapshot['gpu_memory_utilization'] = (mem_info.used / mem_info.total) * 100
                    except:
                        pass

            self.memory_snapshots.append(snapshot)

    def start_inference_timing(self, input_token_count):
        """开始推理计时"""
        self.inference_start_time = time.time()
        self.input_tokens = input_token_count
        self.output_tokens = 0
        self.real_time_tokens = 0
        print(f"[DEBUG] 推理开始，输入Token数: {input_token_count}")

    def record_first_token(self):
        """记录首个Token生成时间"""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            first_token_latency = self.first_token_time - self.inference_start_time
            print(f"[DEBUG] 首Token延迟: {first_token_latency:.3f}秒")
            return first_token_latency
        return None

    def record_token_generation(self, new_text):
        """记录每个Token生成的时间"""
        current_time = time.time()
        self.token_timestamps.append(current_time)
        self.response_chunks.append(new_text)
        self.real_time_tokens += 1

    def end_inference_timing(self, total_generated_text):
        """结束推理计时并计算最终统计"""
        self.generation_end_time = time.time()

        # 使用tokenizer精确计算输出Token数
        if hasattr(self, 'tokenizer') and self.tokenizer:
            self.output_tokens = len(self.tokenizer.encode(total_generated_text))
        else:
            # 备用方案：使用实时计数
            self.output_tokens = self.real_time_tokens

        print(f"[DEBUG] 推理结束，输出Token数: {self.output_tokens}")

    def collect_snapshot(self):
        """收集当前资源使用快照"""
        if not self.monitoring:
            return

        snapshot = {
            'timestamp': time.time() - self.start_time,
            'cpu': psutil.cpu_percent(),
        }

        # 更新内存峰值
        self.update_memory_peaks()

        if torch.cuda.is_available():
            snapshot['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024 ** 3
            snapshot['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024 ** 3

            if HAS_PYNVML and gpu_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    snapshot['gpu_usage'] = util.gpu

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    snapshot['gpu_memory_percent'] = (mem_info.used / mem_info.total) * 100

                    # 获取GPU温度和功耗
                    try:
                        snapshot['gpu_temp'] = pynvml.nvmlDeviceGetTemperature(
                            gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        pass
                    try:
                        snapshot['gpu_power'] = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0  # W
                    except:
                        pass
                except:
                    pass

        self.metrics.append(snapshot)

    def stop(self):
        """停止监控"""
        self.monitoring = False

    def calculate_speeds(self):
        """计算各种速度指标"""
        if not self.inference_start_time or not self.generation_end_time:
            return {}

        # 基础时间计算
        total_inference_time = self.generation_end_time - self.inference_start_time
        first_token_latency = (self.first_token_time - self.inference_start_time) if self.first_token_time else 0
        generation_time = self.generation_end_time - self.first_token_time if self.first_token_time else total_inference_time

        speeds = {
            # 时间指标
            'total_inference_time': total_inference_time,
            'first_token_latency': first_token_latency,
            'pure_generation_time': generation_time,

            # Token计数
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.input_tokens + self.output_tokens,
        }

        # 计算各种速度
        if total_inference_time > 0:
            # 1. 总体推理速度 (包括首Token延迟)
            speeds['total_inference_speed'] = self.output_tokens / total_inference_time

            # 2. 整体处理速度 (输入+输出)
            speeds['overall_processing_speed'] = (self.input_tokens + self.output_tokens) / total_inference_time

        if generation_time > 0 and self.output_tokens > 1:
            # 3. 纯生成速度 (排除首Token延迟)
            speeds['pure_generation_speed'] = (self.output_tokens - 1) / generation_time

            # 4. 回复速度 (从首Token开始的连续生成速度)
            speeds['response_speed'] = (self.output_tokens - 1) / generation_time

        # 5. 实时平均速度计算
        if len(self.token_timestamps) > 1:
            time_intervals = []
            for i in range(1, len(self.token_timestamps)):
                interval = self.token_timestamps[i] - self.token_timestamps[i - 1]
                if interval > 0:
                    time_intervals.append(interval)

            if time_intervals:
                avg_interval = sum(time_intervals) / len(time_intervals)
                speeds['real_time_avg_speed'] = 1.0 / avg_interval if avg_interval > 0 else 0
                speeds['real_time_intervals'] = time_intervals

        return speeds

    def get_summary(self):
        """获取完整的监控摘要"""
        if not self.metrics:
            return {}

        cpu_values = [m['cpu'] for m in self.metrics]
        summary = {
            'duration': self.metrics[-1]['timestamp'],
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'cpu_min': min(cpu_values),
        }

        # 内存峰值信息
        summary['memory_peaks'] = {
            'system_memory_peak': self.system_memory_peak,
            'pytorch_memory_peak': self.pytorch_memory_peak,
            'gpu_memory_peak': self.gpu_memory_peak,
        }

        # 内存增长分析
        if self.memory_baseline:
            baseline_system = self.memory_baseline.get('system_memory_used', 0)
            baseline_pytorch = self.memory_baseline.get('gpu_memory_allocated', 0)

            summary['memory_growth'] = {
                'system_memory_growth': self.system_memory_peak - baseline_system,
                'pytorch_memory_growth': self.pytorch_memory_peak - baseline_pytorch,
            }

        if torch.cuda.is_available():
            gpu_mem_values = [m.get('gpu_memory_allocated', 0) for m in self.metrics]
            summary['gpu_memory_avg'] = sum(gpu_mem_values) / len(gpu_mem_values)

            if any('gpu_usage' in m for m in self.metrics):
                gpu_usage_values = [m['gpu_usage'] for m in self.metrics if 'gpu_usage' in m]
                summary['gpu_usage_avg'] = sum(gpu_usage_values) / len(gpu_usage_values)
                summary['gpu_usage_max'] = max(gpu_usage_values)
                summary['gpu_usage_min'] = min(gpu_usage_values)

            if any('gpu_temp' in m for m in self.metrics):
                gpu_temp_values = [m['gpu_temp'] for m in self.metrics if 'gpu_temp' in m]
                summary['gpu_temp_avg'] = sum(gpu_temp_values) / len(gpu_temp_values)
                summary['gpu_temp_max'] = max(gpu_temp_values)

            if any('gpu_power' in m for m in self.metrics):
                gpu_power_values = [m['gpu_power'] for m in self.metrics if 'gpu_power' in m]
                summary['gpu_power_avg'] = sum(gpu_power_values) / len(gpu_power_values)
                summary['gpu_power_max'] = max(gpu_power_values)

        return summary

    def get_memory_analysis(self):
        """获取详细的内存分析"""
        if not self.memory_snapshots:
            return {}

        analysis = {
            'baseline': self.memory_baseline,
            'peaks': {
                'system_memory_peak': self.system_memory_peak,
                'pytorch_memory_peak': self.pytorch_memory_peak,
                'gpu_memory_peak': self.gpu_memory_peak,
            }
        }

        # 计算内存使用趋势
        if len(self.memory_snapshots) > 1:
            system_memories = [s['system_memory'] for s in self.memory_snapshots]
            pytorch_memories = [s.get('pytorch_allocated', 0) for s in self.memory_snapshots]

            analysis['trends'] = {
                'system_memory_trend': 'increasing' if system_memories[-1] > system_memories[0] else 'stable',
                'pytorch_memory_trend': 'increasing' if pytorch_memories[-1] > pytorch_memories[0] else 'stable',
                'system_memory_variance': max(system_memories) - min(system_memories),
                'pytorch_memory_variance': max(pytorch_memories) - min(pytorch_memories),
            }

        # 内存效率分析
        if self.memory_baseline and torch.cuda.is_available():
            baseline_pytorch = self.memory_baseline.get('gpu_memory_allocated', 0)
            model_memory_footprint = self.pytorch_memory_peak - baseline_pytorch

            analysis['efficiency'] = {
                'model_memory_footprint': model_memory_footprint,
                'memory_efficiency': 'good' if model_memory_footprint < 6 else 'high' if model_memory_footprint < 8 else 'very_high'
            }

        return analysis


def print_detailed_performance_report(speeds, summary, memory_analysis):
    """打印详细的性能报告"""

    print(f"\n{'=' * 80}")
    print("🔥 详细性能分析报告")
    print(f"{'=' * 80}")

    # 1. 速度指标分析
    print(f"\n📊 速度指标分析:")
    print(f"{'-' * 50}")

    if speeds:
        print(f"⏱️  首Token延迟: {speeds.get('first_token_latency', 0):.3f} 秒")
        print(f"⏱️  总推理时间: {speeds.get('total_inference_time', 0):.3f} 秒")
        print(f"⏱️  纯生成时间: {speeds.get('pure_generation_time', 0):.3f} 秒")

        print(f"\n🔢 Token 统计:")
        print(f"   输入 Tokens: {speeds.get('input_tokens', 0)}")
        print(f"   输出 Tokens: {speeds.get('output_tokens', 0)}")
        print(f"   总计 Tokens: {speeds.get('total_tokens', 0)}")

        print(f"\n🚀 推理速度 (关键指标):")
        total_speed = speeds.get('total_inference_speed', 0)
        pure_speed = speeds.get('pure_generation_speed', 0)
        response_speed = speeds.get('response_speed', 0)
        realtime_speed = speeds.get('real_time_avg_speed', 0)

        print(f"   📈 总体推理速度: {total_speed:.2f} tokens/秒 (包含首Token延迟)")
        print(f"   ⚡ 纯生成速度: {pure_speed:.2f} tokens/秒 (排除首Token延迟)")
        print(f"   💬 回复速度: {response_speed:.2f} tokens/秒 (连续生成速度)")
        print(f"   📊 实时平均速度: {realtime_speed:.2f} tokens/秒 (基于实际间隔)")

        # 速度评级
        print(f"\n📝 速度评级:")
        speed_to_evaluate = pure_speed  # 使用纯生成速度作为主要评估指标

        if speed_to_evaluate > 30:
            rating = "🏆 优秀 (Excellent)"
            color = "✅"
        elif speed_to_evaluate > 20:
            rating = "🥇 良好 (Good)"
            color = "🟢"
        elif speed_to_evaluate > 15:
            rating = "🥈 中等 (Average)"
            color = "🟡"
        elif speed_to_evaluate > 10:
            rating = "🥉 偏低 (Below Average)"
            color = "🟠"
        else:
            rating = "❌ 需要优化 (Needs Optimization)"
            color = "🔴"

        print(f"   {color} 纯生成速度评级: {rating}")

        # 首Token延迟评级
        latency = speeds.get('first_token_latency', 0)
        if latency < 0.5:
            latency_rating = "🏆 优秀 (<0.5s)"
        elif latency < 1.0:
            latency_rating = "🟢 良好 (<1.0s)"
        elif latency < 2.0:
            latency_rating = "🟡 中等 (<2.0s)"
        else:
            latency_rating = "🔴 需要优化 (>2.0s)"

        print(f"   {latency_rating[:2]} 首Token延迟评级: {latency_rating[2:]}")

    # 2. 内存使用分析
    print(f"\n💾 内存使用分析:")
    print(f"{'-' * 50}")

    if memory_analysis:
        baseline = memory_analysis.get('baseline', {})
        peaks = memory_analysis.get('peaks', {})
        trends = memory_analysis.get('trends', {})
        efficiency = memory_analysis.get('efficiency', {})

        print(f"🔧 内存基线:")
        if baseline:
            print(f"   系统内存基线: {baseline.get('system_memory_used', 0):.2f} GB "
                  f"({baseline.get('system_memory_percent', 0):.1f}%)")
            print(f"   GPU内存基线: {baseline.get('gpu_memory_allocated', 0):.2f} GB")

        print(f"\n📈 内存峰值:")
        system_peak = peaks.get('system_memory_peak', 0)
        pytorch_peak = peaks.get('pytorch_memory_peak', 0)
        gpu_peak = peaks.get('gpu_memory_peak', 0)

        print(f"   系统内存峰值: {system_peak:.2f} GB")
        print(f"   PyTorch GPU内存峰值: {pytorch_peak:.2f} GB")
        if gpu_peak > 0:
            print(f"   GPU显存峰值: {gpu_peak:.2f} GB")

        # 内存增长分析
        if baseline and summary and 'memory_growth' in summary:
            growth = summary['memory_growth']
            system_growth = growth.get('system_memory_growth', 0)
            pytorch_growth = growth.get('pytorch_memory_growth', 0)

            print(f"\n📊 内存增长:")
            print(f"   系统内存增长: {system_growth:.2f} GB")
            print(f"   GPU内存增长: {pytorch_growth:.2f} GB")

            # 内存增长评估
            if pytorch_growth < 1.0:
                print(f"   ✅ GPU内存增长合理")
            elif pytorch_growth < 2.0:
                print(f"   🟡 GPU内存增长中等")
            else:
                print(f"   ⚠️  GPU内存增长较大")

        # 内存效率分析
        if efficiency:
            model_footprint = efficiency.get('model_memory_footprint', 0)
            memory_efficiency = efficiency.get('memory_efficiency', 'unknown')

            print(f"\n🎯 内存效率:")
            print(f"   模型内存占用: {model_footprint:.2f} GB")

            if memory_efficiency == 'good':
                print(f"   ✅ 内存使用效率: 优秀 (<6GB)")
            elif memory_efficiency == 'high':
                print(f"   🟡 内存使用效率: 中等 (6-8GB)")
            else:
                print(f"   🔴 内存使用效率: 需优化 (>8GB)")

        # 内存趋势分析
        if trends:
            print(f"\n📉 内存趋势:")
            sys_trend = trends.get('system_memory_trend', 'unknown')
            pytorch_trend = trends.get('pytorch_memory_trend', 'unknown')

            print(f"   系统内存趋势: {sys_trend}")
            print(f"   GPU内存趋势: {pytorch_trend}")

            sys_variance = trends.get('system_memory_variance', 0)
            pytorch_variance = trends.get('pytorch_memory_variance', 0)

            print(f"   系统内存波动: {sys_variance:.2f} GB")
            print(f"   GPU内存波动: {pytorch_variance:.2f} GB")

    # 3. 资源使用分析
    print(f"\n💻 资源使用分析:")
    print(f"{'-' * 50}")

    if summary:
        print(f"🔧 CPU 使用情况:")
        cpu_avg = summary.get('cpu_avg', 0)
        cpu_max = summary.get('cpu_max', 0)
        print(f"   平均使用率: {cpu_avg:.1f}%")
        print(f"   峰值使用率: {cpu_max:.1f}%")

        if cpu_avg > 80:
            print(f"   ⚠️  CPU使用率较高，可能成为瓶颈")
        elif cpu_avg < 20:
            print(f"   ✅ CPU使用率正常，GPU为主要计算单元")

        if torch.cuda.is_available():
            print(f"\n🎮 GPU 使用情况:")
            gpu_usage_avg = summary.get('gpu_usage_avg', 0)
            gpu_usage_max = summary.get('gpu_usage_max', 0)
            gpu_memory_avg = summary.get('gpu_memory_avg', 0)

            print(f"   计算使用率 - 平均: {gpu_usage_avg:.1f}%, 峰值: {gpu_usage_max:.1f}%")
            print(f"   显存使用 - 平均: {gpu_memory_avg:.2f}GB")

            if 'gpu_temp_avg' in summary:
                print(f"   温度 - 平均: {summary['gpu_temp_avg']:.1f}°C, "
                      f"峰值: {summary['gpu_temp_max']:.1f}°C")
            if 'gpu_power_avg' in summary:
                print(f"   功耗 - 平均: {summary['gpu_power_avg']:.1f}W, "
                      f"峰值: {summary['gpu_power_max']:.1f}W")

            # GPU使用率评估
            if gpu_usage_avg > 85:
                print(f"   ✅ GPU使用率优秀，计算资源充分利用")
            elif gpu_usage_avg > 70:
                print(f"   🟡 GPU使用率良好")
            else:
                print(f"   ⚠️  GPU使用率偏低，可能存在瓶颈")

    # 4. 优化建议
    print(f"\n🛠️  优化建议:")
    print(f"{'-' * 50}")

    suggestions = []

    if speeds:
        if speeds.get('first_token_latency', 0) > 1.5:
            suggestions.append("🔥 添加模型预热来减少首Token延迟")

        if speeds.get('pure_generation_speed', 0) < 20:
            suggestions.append("⚡ 考虑使用FP16精度替代4bit量化 (如果显存充足)")
            suggestions.append("🔧 检查CPU-GPU数据传输是否存在瓶颈")

    if summary:
        if summary.get('gpu_usage_avg', 0) < 70:
            suggestions.append("📈 GPU利用率偏低，检查模型配置和输入批次大小")

        if summary.get('cpu_avg', 0) > 70:
            suggestions.append("🚀 优化数据预处理流程，减少CPU负载")

    # 内存相关建议
    if memory_analysis:
        efficiency = memory_analysis.get('efficiency', {})
        if efficiency.get('memory_efficiency') == 'very_high':
            suggestions.append("💾 内存使用量大，考虑更激进的量化策略或模型分片")

        peaks = memory_analysis.get('peaks', {})
        if peaks.get('pytorch_memory_peak', 0) > 7:  # 接近8GB显存限制
            suggestions.append("⚠️  GPU内存接近限制，建议降低max_new_tokens或使用更小batch size")

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    else:
        print(f"   ✅ 当前性能表现优秀，无需特殊优化!")

    # 5. 硬件配置评估
    if torch.cuda.is_available():
        print(f"\n🔌 硬件配置评估:")
        print(f"{'-' * 50}")
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"   GPU型号: {gpu_name}")
        print(f"   显存容量: {gpu_memory:.1f} GB")

        if "RTX 4060" in gpu_name:
            print(f"   💡 RTX 4060 Laptop GPU适合运行8B模型，性能表现符合预期")

        if gpu_memory < 8:
            print(f"   ⚠️  显存容量偏小，建议升级到8GB+的GPU")
        else:
            print(f"   ✅ 显存容量充足")

        # 内存利用率评估
        if memory_analysis:
            peaks = memory_analysis.get('peaks', {})
            pytorch_peak = peaks.get('pytorch_memory_peak', 0)
            utilization = (pytorch_peak / gpu_memory) * 100 if gpu_memory > 0 else 0

            print(f"   📊 显存利用率: {utilization:.1f}% ({pytorch_peak:.2f}GB / {gpu_memory:.1f}GB)")

            if utilization > 90:
                print(f"   🔴 显存利用率很高，接近满载")
            elif utilization > 70:
                print(f"   🟡 显存利用率较高，运行良好")
            elif utilization > 50:
                print(f"   ✅ 显存利用率适中，还有优化空间")
            else:
                print(f"   💡 显存利用率较低，可考虑运行更大模型")


def main():
    """主函数"""
    # Print current working directory to help locate model files
    print(f"Current working directory: {os.getcwd()}")

    # Model configuration
    max_seq_length = 2048
    load_in_4bit = True

    # Path to your saved fine-tuned model
    model_path = "E:/DS_8b/DeepSeek-R1-Medical-COT_910"
    print(f"Loading model from: {model_path}")

    # 初始化高级监控器
    monitor = AdvancedPerformanceMonitor()

    print("开始模型加载监控...")
    monitor.start()

    # Load the fine-tuned model and tokenizer
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        # 将tokenizer添加到监控器中用于精确计算
        monitor.tokenizer = tokenizer
        print("Model and tokenizer loaded successfully!")
        monitor.collect_snapshot()  # 记录加载后状态
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set the model to inference mode
    FastLanguageModel.for_inference(model)
    print("Model prepared for inference")

    # Setup prompt template
    test_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Please answer the following medical question.

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

    # Tokenize input and count tokens
    inputs = tokenizer([formatted_input], return_tensors="pt")
    input_token_count = inputs.input_ids.shape[1]

    # Move inputs to the same device as the model
    if torch.cuda.is_available():
        device = "cuda"
        inputs = inputs.to(device)
        model = model.to(device)
        print("Using CUDA for inference")
    else:
        device = "cpu"
        print("Using CPU for inference (this might be slow)")

    # 记录推理前状态
    monitor.collect_snapshot()

    # 高级流式生成 + 精确速度监控
    print(f"\n{'=' * 80}")
    print("🚀 开始高精度推理速度监控")
    print(f"{'=' * 80}")

    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generation parameters
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 1200,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_cache": True,
        "streamer": streamer,
        "do_sample": True,
    }

    # 监控推理过程
    def monitor_during_generation():
        """推理过程中的监控函数"""
        while monitor.monitoring:
            monitor.collect_snapshot()
            time.sleep(0.1)  # 每100ms采样一次

    # 开始推理计时
    monitor.start_inference_timing(input_token_count)

    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    # Start monitoring thread
    monitor_thread = Thread(target=monitor_during_generation)
    monitor_thread.daemon = True
    monitor_thread.start()

    thread.start()

    # 实时监控和记录
    first_token_received = False
    generated_text = ""
    token_count = 0

    print("📝 生成响应:")
    print(f"{'-' * 60}")
    print("Response: ", end="", flush=True)

    for new_text in streamer:
        if not first_token_received:
            # 记录首Token
            latency = monitor.record_first_token()
            first_token_received = True
            print(f"\n[⚡ 首Token延迟: {latency:.3f}秒]")
            print("Response: ", end="", flush=True)

        # 记录每个token的生成时间
        monitor.record_token_generation(new_text)

        print(new_text, end="", flush=True)
        generated_text += new_text
        token_count += 1

    thread.join()

    # 结束计时
    monitor.end_inference_timing(generated_text)

    # 停止监控
    monitor.stop()

    print(f"\n{'-' * 60}")

    # 计算详细速度指标
    speeds = monitor.calculate_speeds()
    summary = monitor.get_summary()
    memory_analysis = monitor.get_memory_analysis()

    # 打印详细性能报告
    print_detailed_performance_report(speeds, summary, memory_analysis)

    # PyTorch GPU内存详情
    if torch.cuda.is_available():
        print(f"\n🔧 PyTorch GPU内存详情:")
        print(f"{'-' * 50}")
        current_allocated = torch.cuda.memory_allocated() / 1024 ** 3
        current_reserved = torch.cuda.memory_reserved() / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3

        print(f"   当前已分配: {current_allocated:.2f} GB")
        print(f"   当前已保留: {current_reserved:.2f} GB")
        print(f"   峰值已分配: {max_allocated:.2f} GB")

        # 内存碎片分析
        memory_fragmentation = current_reserved - current_allocated
        print(f"   内存碎片: {memory_fragmentation:.2f} GB")

        if memory_fragmentation > 1.0:
            print(f"   ⚠️  内存碎片较多，建议定期清理GPU缓存")
        else:
            print(f"   ✅ 内存碎片控制良好")

    # 内存使用历史分析
    if monitor.memory_snapshots and len(monitor.memory_snapshots) > 2:
        print(f"\n📈 内存使用历史分析:")
        print(f"{'-' * 50}")

        # 分析内存使用模式
        pytorch_memories = [s.get('pytorch_allocated', 0) for s in monitor.memory_snapshots]
        system_memories = [s['system_memory'] for s in monitor.memory_snapshots]
        timestamps = [s['timestamp'] for s in monitor.memory_snapshots]

        # 找到内存增长最快的阶段
        max_pytorch_growth = 0
        max_growth_period = None

        for i in range(1, len(pytorch_memories)):
            growth = pytorch_memories[i] - pytorch_memories[i - 1]
            if growth > max_pytorch_growth:
                max_pytorch_growth = growth
                max_growth_period = (timestamps[i - 1], timestamps[i])

        if max_growth_period:
            print(f"   最大内存增长: {max_pytorch_growth:.2f} GB")
            print(f"   增长时段: {max_growth_period[0]:.1f}s - {max_growth_period[1]:.1f}s")

        # 内存稳定性分析
        pytorch_variance = max(pytorch_memories) - min(pytorch_memories)
        if pytorch_variance < 0.5:
            print(f"   ✅ GPU内存使用稳定 (波动: {pytorch_variance:.2f}GB)")
        else:
            print(f"   🟡 GPU内存使用有波动 (波动: {pytorch_variance:.2f}GB)")

    # 性能瓶颈分析
    print(f"\n🔍 性能瓶颈分析:")
    print(f"{'-' * 50}")

    bottlenecks = []

    if speeds:
        # 首Token延迟瓶颈
        if speeds.get('first_token_latency', 0) > 2.0:
            bottlenecks.append("首Token延迟过高 - 可能需要模型预热或优化加载")

        # 生成速度瓶颈
        if speeds.get('pure_generation_speed', 0) < 15:
            bottlenecks.append("生成速度偏低 - 可能是量化精度或硬件限制")

    if summary:
        # GPU利用率瓶颈
        if summary.get('gpu_usage_avg', 0) < 60:
            bottlenecks.append("GPU利用率低 - 可能存在CPU-GPU数据传输瓶颈")

        # CPU瓶颈
        if summary.get('cpu_avg', 0) > 80:
            bottlenecks.append("CPU使用率高 - 数据预处理可能成为瓶颈")

    # 内存瓶颈
    if memory_analysis:
        efficiency = memory_analysis.get('efficiency', {})
        if efficiency.get('memory_efficiency') == 'very_high':
            bottlenecks.append("内存使用量大 - 可能限制更大的批处理或序列长度")

    if bottlenecks:
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"   {i}. ⚠️  {bottleneck}")
    else:
        print(f"   ✅ 未发现明显性能瓶颈")

    # 总体评估
    print(f"\n🏆 总体性能评估:")
    print(f"{'-' * 50}")

    overall_score = 0
    max_score = 5

    # 评分标准
    if speeds:
        # 首Token延迟评分 (20%)
        latency = speeds.get('first_token_latency', 0)
        if latency < 0.5:
            overall_score += 1
        elif latency < 1.0:
            overall_score += 0.8
        elif latency < 2.0:
            overall_score += 0.6
        else:
            overall_score += 0.3

        # 生成速度评分 (40%)
        speed = speeds.get('pure_generation_speed', 0)
        if speed > 30:
            overall_score += 2
        elif speed > 20:
            overall_score += 1.6
        elif speed > 15:
            overall_score += 1.2
        elif speed > 10:
            overall_score += 0.8
        else:
            overall_score += 0.4

    # GPU利用率评分 (20%)
    if summary and summary.get('gpu_usage_avg', 0) > 0:
        gpu_usage = summary.get('gpu_usage_avg', 0)
        if gpu_usage > 85:
            overall_score += 1
        elif gpu_usage > 70:
            overall_score += 0.8
        elif gpu_usage > 50:
            overall_score += 0.6
        else:
            overall_score += 0.3

    # 内存效率评分 (20%)
    if memory_analysis:
        efficiency = memory_analysis.get('efficiency', {})
        memory_eff = efficiency.get('memory_efficiency', 'unknown')
        if memory_eff == 'good':
            overall_score += 1
        elif memory_eff == 'high':
            overall_score += 0.7
        else:
            overall_score += 0.4

    # 评级
    percentage = (overall_score / max_score) * 100

    if percentage >= 90:
        grade = "🏆 A+ (优秀)"
        color = "✅"
    elif percentage >= 80:
        grade = "🥇 A (良好)"
        color = "🟢"
    elif percentage >= 70:
        grade = "🥈 B (中等)"
        color = "🟡"
    elif percentage >= 60:
        grade = "🥉 C (及格)"
        color = "🟠"
    else:
        grade = "❌ D (需要优化)"
        color = "🔴"

    print(f"   {color} 综合评分: {overall_score:.1f}/{max_score} ({percentage:.1f}%)")
    print(f"   {color} 性能等级: {grade}")

    # 推荐的下一步行动
    print(f"\n🎯 推荐的下一步行动:")
    print(f"{'-' * 50}")

    if percentage >= 90:
        print(f"   🎉 性能表现优秀！可以考虑：")
        print(f"   • 尝试运行更大的模型")
        print(f"   • 增加批处理大小")
        print(f"   • 测试更复杂的推理任务")
    elif percentage >= 70:
        print(f"   🔧 性能良好，可以考虑小幅优化：")
        print(f"   • 添加模型预热")
        print(f"   • 调整量化参数")
        print(f"   • 优化输入数据流水线")
    else:
        print(f"   ⚡ 需要重点优化：")
        print(f"   • 检查硬件配置")
        print(f"   • 尝试不同的量化策略")
        print(f"   • 考虑模型或硬件升级")

    # 实时速度变化分析 (如果有足够数据)
    if speeds and 'real_time_intervals' in speeds and len(speeds['real_time_intervals']) > 5:
        print(f"\n📈 实时速度变化分析:")
        print(f"{'-' * 50}")
        intervals = speeds['real_time_intervals']
        speeds_list = [1.0 / interval for interval in intervals if interval > 0]

        if speeds_list:
            min_speed = min(speeds_list)
            max_speed = max(speeds_list)
            avg_speed = sum(speeds_list) / len(speeds_list)

            print(f"   实时速度范围: {min_speed:.1f} - {max_speed:.1f} tokens/秒")
            print(f"   实时平均速度: {avg_speed:.1f} tokens/秒")
            print(f"   速度稳定性: {'稳定' if (max_speed - min_speed) < 5 else '有波动'}")

    print(f"\n{'=' * 80}")
    print("✅ 监控完成! 性能分析已生成")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
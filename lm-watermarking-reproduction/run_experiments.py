#!/usr/bin/env python3

import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList
)

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']#绘图设置
plt.rcParams['axes.unicode_minus'] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"#设备
MODEL_NAME = "facebook/opt-125m"#模型名称，可改
MAX_NEW_TOKENS = 200#最多生成200个token
NUM_PROMPTS = 50  #测试prompts数，可改
SEED = 42#随机种子

CONFIGS = [#实验配置，有四种，可改
    {"gamma": 0.25, "delta": 2.0, "name": "gamma0.25_delta2.0"},
    {"gamma": 0.25, "delta": 5.0, "name": "gamma0.25_delta5.0"},
    {"gamma": 0.25, "delta": 0.5, "name": "gamma0.25_delta0.5"},
    {"gamma": 0.5, "delta": 2.0, "name": "gamma0.5_delta2.0"},
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_prompts_from_json(json_path="/home/wutong/proj/XYT/newKGW/c4_prompts.json", n=NUM_PROMPTS):#加载prompts

    print(f"\nLoading prompts from local file: {json_path}")

    with open(json_path, 'r') as f:
        all_prompts = json.load(f)#打开文件，读取prompts
    selected = random.sample(all_prompts, min(n, len(all_prompts)))#随机采样

    selected = [p[:300] if len(p) > 300 else p for p in selected]#过长prompts截断
    return selected

def load_model():#加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)#加载分词器
    if tokenizer.pad_token is None:#如果没有填充token，就用结束token代替
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(#加载模型
        MODEL_NAME,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer

def calculate_perplexity(text, model, tokenizer):#PPL计算
    if not text or len(text.strip()) < 10:
        return float('inf')#文本太短，返回无穷大

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(DEVICE)

    with torch.no_grad():#让模型预测下一个token，计算损失，然后exp(loss)得到困惑度
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity

def generate_watermarked(prompt, gamma, delta, model, tokenizer):#生成带水印文本
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=gamma,
        delta=delta,
        seeding_scheme="selfhash"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([watermark_processor]),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            min_new_tokens=10,
        )
    #提取新生成的token（去掉输入的prompt），解码成文字
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[:, input_len:]
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return generated_text


def generate_non_watermarked(prompt, model, tokenizer):#生成无水印文本
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            min_new_tokens=10,
        )

    #定义无水印生成函数，和上面几乎一样，只是不传水印处理器。
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[:, input_len:]
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return generated_text

def detect_watermark(text, gamma, tokenizer):#定义水印检测函数
    detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=gamma,
        seeding_scheme="selfhash",
        device=DEVICE,
        tokenizer=tokenizer,
        z_threshold=4.0,#z-score超过4判定为有水印
        normalizers=[],
        ignore_repeated_ngrams=True#忽略重复的n-gram
    )
    return detector.detect(text)#调用检测函数，返回结果

def run_single_experiment(prompt, config, model, tokenizer):#单次实验
    gamma, delta, name = config["gamma"], config["delta"], config["name"]

    #生成无水印文本，计算困惑度，检测水印
    text_no_wm = generate_non_watermarked(prompt, model, tokenizer)
    ppl_no_wm = calculate_perplexity(prompt + text_no_wm, model, tokenizer)
    detect_no_wm = detect_watermark(text_no_wm, gamma, tokenizer)

    #生成带水印文本，计算困惑度，检测水印
    text_wm = generate_watermarked(prompt, gamma, delta, model, tokenizer)
    ppl_wm = calculate_perplexity(prompt + text_wm, model, tokenizer)
    detect_wm = detect_watermark(text_wm, gamma, tokenizer)

    return {#返回一个字典，包含所有结果
        "prompt": prompt[:100],
        "config_name": name,
        "gamma": gamma,
        "delta": delta,
        "non_watermarked": {
            "text": text_no_wm[:200],
            "z_score": detect_no_wm["z_score"],
            "p_value": detect_no_wm["p_value"],
            "green_fraction": detect_no_wm["green_fraction"],
            "perplexity": ppl_no_wm,
            "is_watermarked": detect_no_wm["z_score"] > 4.0
        },
        "watermarked": {
            "text": text_wm[:200],
            "z_score": detect_wm["z_score"],
            "p_value": detect_wm["p_value"],
            "green_fraction": detect_wm["green_fraction"],
            "perplexity": ppl_wm,
            "is_watermarked": detect_wm["z_score"] > 4.0
        }
    }

def create_plots(results, config_name, output_dir="plots"):#画图
    os.makedirs(output_dir, exist_ok=True)#plots文件夹

    n_runs = len(results)
    run_ids = list(range(1, n_runs + 1))
    z_scores_wm = [r["watermarked"]["z_score"] for r in results]
    z_scores_no = [r["non_watermarked"]["z_score"] for r in results]
    ppl_wm = [r["watermarked"]["perplexity"] for r in results]
    ppl_no = [r["non_watermarked"]["perplexity"] for r in results]

    # Plot 1: z-score
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(n_runs)
    width = 0.35

    ax1.bar(x - width / 2, z_scores_no, width, label='No Watermark', color='skyblue')
    ax1.bar(x + width / 2, z_scores_wm, width, label='Watermarked', color='coral')

    ax1.axhline(y=4.0, color='r', linestyle='--', label='Detection Threshold (z=4.0)')
    ax1.set_xlabel('Experiment Run', fontsize=12)
    ax1.set_ylabel('z-score', fontsize=12)
    ax1.set_title(f'{config_name} - z-score Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Run {i + 1}' for i in range(n_runs)], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{config_name}_zscore_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Perplexity
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(run_ids, ppl_no, 'o-', label='No Watermark', color='skyblue', markersize=8)
    ax2.plot(run_ids, ppl_wm, 's-', label='Watermarked', color='coral', markersize=8)
    ax2.set_xlabel('Experiment Run', fontsize=12)
    ax2.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax2.set_title(f'{config_name} - Perplexity Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{config_name}_perplexity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_dir}/")


def create_summary_plots(all_results, output_dir="plots"):#汇总图
    os.makedirs(output_dir, exist_ok=True)

    config_names = list(all_results.keys())
    n_configs = len(config_names)
    n_runs = len(all_results[config_names[0]])

    run_ids = list(range(1, n_runs + 1))

    config_colors = {
        "gamma0.25_delta2.0": "blue",
        "gamma0.25_delta5.0": "red",
        "gamma0.25_delta0.5": "green",
        "gamma0.5_delta2.0": "purple",
    }

    line_style_wm = "solid"
    line_style_no = "dashed"

    # Figure 1: z-score
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    for config_name in config_names:
        results = all_results[config_name]
        z_scores_wm = [r["watermarked"]["z_score"] for r in results]
        z_scores_no = [r["non_watermarked"]["z_score"] for r in results]

        display_name = config_name.replace("_", ", ")

        ax1.plot(run_ids, z_scores_wm, 'o-',
                 label=f'{display_name} (Watermarked)',
                 color=config_colors[config_name],
                 linestyle=line_style_wm,
                 linewidth=2, markersize=8)

        ax1.plot(run_ids, z_scores_no, 's--',
                 label=f'{display_name} (No Watermark)',
                 color=config_colors[config_name],
                 linestyle=line_style_no,
                 linewidth=2, markersize=8)

    ax1.axhline(y=4.0, color='r', linestyle='-', linewidth=2, label='Detection Threshold (z=4.0)')
    ax1.set_xlabel('Experiment Run (Prompt Index)', fontsize=12)
    ax1.set_ylabel('z-score', fontsize=12)
    ax1.set_title('z-score Trends Across Configurations (8 Lines)', fontsize=14)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_zscore_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Perplexity
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    for config_name in config_names:
        results = all_results[config_name]
        ppl_wm = [r["watermarked"]["perplexity"] for r in results]
        ppl_no = [r["non_watermarked"]["perplexity"] for r in results]

        display_name = config_name.replace("_", ", ")

        ax2.plot(run_ids, ppl_wm, 'o-',
                 label=f'{display_name} (Watermarked)',
                 color=config_colors[config_name],
                 linestyle=line_style_wm,
                 linewidth=2, markersize=8)

        ax2.plot(run_ids, ppl_no, 's--',
                 label=f'{display_name} (No Watermark)',
                 color=config_colors[config_name],
                 linestyle=line_style_no,
                 linewidth=2, markersize=8)

    ax2.set_xlabel('Experiment Run (Prompt Index)', fontsize=12)
    ax2.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax2.set_title('Perplexity Trends Across Configurations (8 Lines)', fontsize=14)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_perplexity_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary trend plots saved to {output_dir}/")

    #计算平均值
    mean_z_wm = [np.mean([r["watermarked"]["z_score"] for r in all_results[name]]) for name in config_names]
    mean_z_no = [np.mean([r["non_watermarked"]["z_score"] for r in all_results[name]]) for name in config_names]
    mean_ppl_wm = [np.mean([r["watermarked"]["perplexity"] for r in all_results[name]]) for name in config_names]
    mean_ppl_no = [np.mean([r["non_watermarked"]["perplexity"] for r in all_results[name]]) for name in config_names]

    return mean_z_wm, mean_z_no, mean_ppl_wm, mean_ppl_no

def main():
    model, tokenizer = load_model()#加载模型
    prompts = load_prompts_from_json(n=NUM_PROMPTS)#加载50个prom
    all_results = {}#创建空字典存储所有结果

    # Run experiments
    for config in CONFIGS:
        print(f"\n{'=' * 80}")#遍历4种配置
        print(f"Running configuration: {config['name']} (γ={config['gamma']}, δ={config['delta']})")
        print(f"{'=' * 80}")
        config_results = []

        for i, prompt in enumerate(tqdm(prompts, desc="Progress", unit="prompt")):#遍历50个prompts
            result = run_single_experiment(prompt, config, model, tokenizer)
            config_results.append(result)

        all_results[config['name']] = config_results
        create_plots(config_results, config['name'])#生成图片

        wm_z_mean = np.mean([r['watermarked']['z_score'] for r in config_results])#计算当前配置的平均值和标准差
        wm_z_std = np.std([r['watermarked']['z_score'] for r in config_results])
        wm_ppl_mean = np.mean([r['watermarked']['perplexity'] for r in config_results])
        no_z_mean = np.mean([r['non_watermarked']['z_score'] for r in config_results])
        no_ppl_mean = np.mean([r['non_watermarked']['perplexity'] for r in config_results])

        print(f"\n  Configuration Summary:")#打印汇总
        print(f"    Watermarked:   z = {wm_z_mean:.3f} ± {wm_z_std:.3f}, PPL = {wm_ppl_mean:.2f}")
        print(f"    No Watermark:  z = {no_z_mean:.3f}, PPL = {no_ppl_mean:.2f}")

    mean_z_wm, mean_z_no, mean_ppl_wm, mean_ppl_no = create_summary_plots(all_results)#汇总

    output_file = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"#保存所有结果到JSON文件
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_file}")

    print("\n" + "=" * 100)#打印表格
    print("FINAL RESULTS SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Configuration':<25} {'WM z-score':<15} {'NoWM z-score':<15} {'WM PPL':<12} {'NoWM PPL':<12} {'Detectable':<12}")
    print("-" * 100)

    config_names = list(all_results.keys())
    for i, name in enumerate(config_names):
        wm_z = mean_z_wm[i]
        no_z = mean_z_no[i]
        wm_ppl = mean_ppl_wm[i]
        no_ppl = mean_ppl_no[i]
        detectable = "✓ Yes" if wm_z > 4.0 else "✗ No"
        display_name = name.replace("_", ", ")
        print(f"{display_name:<25} {wm_z:<15.3f} {no_z:<15.3f} {wm_ppl:<12.2f} {no_ppl:<12.2f} {detectable:<12}")

    print("=" * 100)
    print(f"\nExperiment completed!")
    print(f"\nOutput files:")
    print(f"  - Individual plots: plots/*.png")
    print(f"  - Summary trend plots (8 lines each):")
    print(f"      plots/summary_zscore_trends.png")
    print(f"      plots/summary_perplexity_trends.png")
    print(f"      plots/summary_greenfraction_trends.png")
    print(f"  - Raw data: {output_file}")


if __name__ == "__main__":
    main()
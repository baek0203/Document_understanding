"""
훈련 결과 시각화 스크립트
여러 모델의 훈련 결과를 비교하고 시각화합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from PIL import Image

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 결과 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent
RUNS_DIR = BASE_DIR / 'runs' / 'doclaynet'
OUTPUT_DIR = BASE_DIR / 'visualization_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# 모델 실행 디렉토리
MODEL_RUNS = {
    'YOLOv8n': 'yolov8n_doclaynet4',
    'YOLO11n': 'yolo11n',
    'YOLO11n_v2': 'yolo11n2'
}

def load_results(model_name, run_dir):
    """각 모델의 results.csv 로드"""
    csv_path = RUNS_DIR / run_dir / 'results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['model'] = model_name
        # 공백 제거
        df.columns = df.columns.str.strip()
        return df
    return None

def plot_training_metrics():
    """훈련 메트릭 비교 그래프"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics Comparison Across Models', fontsize=16, fontweight='bold')

    all_data = []
    for model_name, run_dir in MODEL_RUNS.items():
        df = load_results(model_name, run_dir)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("No results data found!")
        return

    # mAP50 비교
    ax = axes[0, 0]
    for df in all_data:
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], marker='o', label=df['model'].iloc[0], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mAP50', fontsize=12)
    ax.set_title('mAP@0.5 During Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mAP50-95 비교
    ax = axes[0, 1]
    for df in all_data:
        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], marker='s', label=df['model'].iloc[0], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mAP50-95', fontsize=12)
    ax.set_title('mAP@0.5:0.95 During Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1, 0]
    for df in all_data:
        ax.plot(df['epoch'], df['metrics/precision(B)'], marker='^', label=f"{df['model'].iloc[0]} - Precision", linewidth=2)
        ax.plot(df['epoch'], df['metrics/recall(B)'], marker='v', label=f"{df['model'].iloc[0]} - Recall", linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision and Recall', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss 비교
    ax = axes[1, 1]
    for df in all_data:
        total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
        ax.plot(df['epoch'], total_loss, marker='D', label=df['model'].iloc[0], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Training Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_final_metrics_table():
    """최종 메트릭 비교 표"""
    final_metrics = []

    for model_name, run_dir in MODEL_RUNS.items():
        df = load_results(model_name, run_dir)
        if df is not None:
            last_epoch = df.iloc[-1]
            final_metrics.append({
                'Model': model_name,
                'Epochs': int(last_epoch['epoch']),
                'Precision': f"{last_epoch['metrics/precision(B)']:.4f}",
                'Recall': f"{last_epoch['metrics/recall(B)']:.4f}",
                'mAP50': f"{last_epoch['metrics/mAP50(B)']:.4f}",
                'mAP50-95': f"{last_epoch['metrics/mAP50-95(B)']:.4f}",
                'Box Loss': f"{last_epoch['val/box_loss']:.4f}",
                'Class Loss': f"{last_epoch['val/cls_loss']:.4f}"
            })

    metrics_df = pd.DataFrame(final_metrics)

    # 표 시각화
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=metrics_df.values,
                    colLabels=metrics_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 헤더 스타일
    for i in range(len(metrics_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 행 색상
    for i in range(1, len(metrics_df) + 1):
        for j in range(len(metrics_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Final Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)

    output_path = OUTPUT_DIR / 'final_metrics_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # CSV로도 저장
    csv_path = OUTPUT_DIR / 'final_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return metrics_df

def display_confusion_matrices():
    """혼동 행렬 비교"""
    fig, axes = plt.subplots(1, len(MODEL_RUNS), figsize=(18, 5))

    if len(MODEL_RUNS) == 1:
        axes = [axes]

    for idx, (model_name, run_dir) in enumerate(MODEL_RUNS.items()):
        cm_path = RUNS_DIR / run_dir / 'confusion_matrix_normalized.png'
        if cm_path.exists():
            img = Image.open(cm_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{model_name}\nNormalized Confusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].axis('off')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'confusion_matrices_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def display_pr_curves():
    """PR 곡선 비교"""
    fig, axes = plt.subplots(1, len(MODEL_RUNS), figsize=(18, 5))

    if len(MODEL_RUNS) == 1:
        axes = [axes]

    for idx, (model_name, run_dir) in enumerate(MODEL_RUNS.items()):
        pr_path = RUNS_DIR / run_dir / 'BoxPR_curve.png'
        if pr_path.exists():
            img = Image.open(pr_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{model_name}\nPrecision-Recall Curve', fontsize=12, fontweight='bold')
            axes[idx].axis('off')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'pr_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_summary_report():
    """종합 리포트 생성"""
    report_path = OUTPUT_DIR / 'TRAINING_SUMMARY.md'

    # 최종 메트릭 수집
    all_data = []
    for model_name, run_dir in MODEL_RUNS.items():
        df = load_results(model_name, run_dir)
        if df is not None:
            last_epoch = df.iloc[-1]
            all_data.append({
                'model': model_name,
                'epochs': int(last_epoch['epoch']),
                'mAP50': last_epoch['metrics/mAP50(B)'],
                'mAP50_95': last_epoch['metrics/mAP50-95(B)'],
                'precision': last_epoch['metrics/precision(B)'],
                'recall': last_epoch['metrics/recall(B)']
            })

    # 최고 성능 모델 찾기
    best_by_map50 = max(all_data, key=lambda x: x['mAP50'])
    best_by_map50_95 = max(all_data, key=lambda x: x['mAP50_95'])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DocLayNet YOLO Training Results Summary\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"Total models trained: {len(MODEL_RUNS)}\n\n")

        f.write("## Model Performance\n\n")
        f.write("| Model | Epochs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |\n")
        f.write("|-------|--------|-----------|--------|---------|-------------|\n")

        for data in all_data:
            f.write(f"| {data['model']} | {data['epochs']} | "
                   f"{data['precision']:.4f} | {data['recall']:.4f} | "
                   f"{data['mAP50']:.4f} | {data['mAP50_95']:.4f} |\n")

        f.write("\n## Best Models\n\n")
        f.write(f"- **Best mAP@0.5**: {best_by_map50['model']} ({best_by_map50['mAP50']:.4f})\n")
        f.write(f"- **Best mAP@0.5:0.95**: {best_by_map50_95['model']} ({best_by_map50_95['mAP50_95']:.4f})\n\n")

        f.write("## Visualizations\n\n")
        f.write("1. [Training Metrics Comparison](training_metrics_comparison.png)\n")
        f.write("2. [Final Metrics Table](final_metrics_table.png)\n")
        f.write("3. [Confusion Matrices](confusion_matrices_comparison.png)\n")
        f.write("4. [PR Curves](pr_curves_comparison.png)\n\n")

        f.write("## Model Weights\n\n")
        for model_name, run_dir in MODEL_RUNS.items():
            weights_dir = RUNS_DIR / run_dir / 'weights'
            if weights_dir.exists():
                f.write(f"### {model_name}\n")
                f.write(f"- Best weights: `{weights_dir / 'best.pt'}`\n")
                f.write(f"- Last weights: `{weights_dir / 'last.pt'}`\n\n")

    print(f"Saved: {report_path}")

def main():
    print("=" * 60)
    print("DocLayNet YOLO Training Results Visualization")
    print("=" * 60)
    print()

    print("1. Creating training metrics comparison...")
    plot_training_metrics()

    print("\n2. Creating final metrics table...")
    metrics_df = create_final_metrics_table()
    print("\nFinal Metrics:")
    print(metrics_df.to_string(index=False))

    print("\n3. Displaying confusion matrices...")
    display_confusion_matrices()

    print("\n4. Displaying PR curves...")
    display_pr_curves()

    print("\n5. Creating summary report...")
    create_summary_report()

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {file.name}")

if __name__ == '__main__':
    main()

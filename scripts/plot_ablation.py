import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_ablation_from_csv(csv_path, title, output_filename):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Extract the settings and AUROC values
    # Ensure they are in a logical order
    setting_map = {
        'lexical_only': 'Lexical Only\n(Text Semantics)',
        'uncertainty_only': 'Uncertainty Only\n(LLM Log-Probs)',
        'hybrid_no_context': 'Hybrid\n(No Context)',
        'hybrid_with_context': 'Hybrid\n(With Context)'
    }
    
    # Reorder dataframe to match the logical progression
    df['order'] = df['setting'].map({'lexical_only': 1, 'uncertainty_only': 2, 'hybrid_no_context': 3, 'hybrid_with_context': 4})
    df = df.sort_values('order')
    
    labels = [setting_map.get(s, s) for s in df['setting']]
    auroc = df['auroc'].values
    
    # Custom colors highlighting the progression
    colors = ['#aec7e8', '#ffbb78', '#98df8a', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, auroc, color=colors[:len(labels)], width=0.5)

    ax.set_ylabel('AUROC Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Dynamically set y-limits to zoom in on the differences
    min_auroc = min(auroc)
    max_auroc = max(auroc)
    padding = (max_auroc - min_auroc) * 0.5 if max_auroc > min_auroc else 0.05
    ax.set_ylim(max(0.0, min_auroc - padding), min(1.0, max_auroc + padding))

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (padding * 0.05),
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Minimalist grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join('docs/images', output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == '__main__':
    out_dir = 'docs/images'
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot LR
    plot_ablation_from_csv(
        'results/ablation/val_lr/summary.csv', 
        'Ablation Study: Hybrid LR Performance', 
        'ablation_lr.png'
    )
    
    # Plot SVM
    plot_ablation_from_csv(
        'results/ablation/val_svm/summary.csv', 
        'Ablation Study: Hybrid SVM Performance', 
        'ablation_svm.png'
    )

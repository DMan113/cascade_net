import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_temporal_trends():
    csv_path = "outputs/all_quarters_results.csv"
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не знайдено!")
        return

    # Завантажуємо дані
    df = pd.read_csv(csv_path)

    # Вибираємо тільки 3 головні моделі для чистоти графіка
    target_models = ["B1: XGBoost (nodes)", "B3: VanillaGNN", "B5: CascadeNet"]
    df_filtered = df[df['Model'].isin(target_models)].copy()

    # Сортуємо квартали (щоб вісь Х була по порядку)
    df_filtered = df_filtered.sort_values(by='Quarter')

    # Налаштовуємо стиль
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

    # Створюємо сітку з 3 графіків (один під одним)
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    colors = {"B1: XGBoost (nodes)": "#e74c3c", "B3: VanillaGNN": "#3498db", "B5: CascadeNet": "#2ecc71"}
    markers = {"B1: XGBoost (nodes)": "s", "B3: VanillaGNN": "^", "B5: CascadeNet": "o"}

    # 1. Графік F1-Score (Якість розпізнавання)
    sns.lineplot(data=df_filtered, x='Quarter', y='F1', hue='Model',
                 palette=colors, style='Model', markers=markers, dashes=False,
                 ax=axes[0], linewidth=2.5, markersize=8)
    axes[0].set_title("Temporal Stability: F1-Score (Higher is better)", fontweight='bold', pad=15)
    axes[0].set_ylabel("F1 Score")
    axes[0].legend(title="")

    # 2. Графік Brier Score (Якість калібрування)
    sns.lineplot(data=df_filtered, x='Quarter', y='Brier', hue='Model',
                 palette=colors, style='Model', markers=markers, dashes=False,
                 ax=axes[1], linewidth=2.5, markersize=8)
    axes[1].set_title("Calibration Quality: Brier Score (Lower is better)", fontweight='bold', pad=15)
    axes[1].set_ylabel("Brier Score")
    axes[1].legend(title="")

    # 3. Графік Spearman (Тільки для CascadeNet, бо XGBoost цього не вміє)
    df_cascadenet = df[df['Model'] == "B5: CascadeNet"].copy()
    sns.lineplot(data=df_cascadenet, x='Quarter', y='Cascade_Spearman',
                 color="#2ecc71", marker="o", ax=axes[2], linewidth=2.5, markersize=8)
    axes[2].set_title("Cascade Ranking: Spearman Correlation (Near 1.0 is perfect)", fontweight='bold', pad=15)
    axes[2].set_ylabel("Spearman ρ")
    axes[2].set_xlabel("Financial Quarter")
    axes[2].set_ylim(0.95, 1.0)  # Масштабуємо, щоб показати стабільність

    # Красиво повертаємо підписи на осі Х
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Зберігаємо
    out_img = "outputs/temporal_analysis_trends.png"
    plt.savefig(out_img, bbox_inches='tight')
    print(f"✅ Графіки успішно збережено в {out_img}")


if __name__ == "__main__":
    plot_temporal_trends()
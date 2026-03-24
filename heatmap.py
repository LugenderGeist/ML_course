import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_full_correlation_heatmap(df, figsize=(16, 14)):

    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include=[np.number])

    # Рассчитываем корреляционную матрицу
    correlation_matrix = numeric_df.corr()

    # Создаем график
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix,
                mask=mask,
                annot=False,
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"})
    plt.title('Тепловая карта корреляций всех числовых признаков', fontsize=16, pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"\n📊 Всего числовых признаков: {len(numeric_df.columns)}")


def plot_high_correlation_heatmap(df, threshold=0.35, figsize=None):

    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include=[np.number])

    # Рассчитываем корреляционную матрицу
    correlation_matrix = numeric_df.corr()

    # Получаем корреляции с целевой переменной
    target_correlations = correlation_matrix['TARGET_deathRate'].sort_values(ascending=False)

    # Выбираем признаки с корреляцией больше threshold по модулю
    high_corr_features = target_correlations[abs(target_correlations) > threshold].index.tolist()
    if 'TARGET_deathRate' in high_corr_features:
        high_corr_features.remove('TARGET_deathRate')

    # Разделяем на положительные и отрицательные
    positive_features = [f for f in high_corr_features if target_correlations[f] > threshold]
    negative_features = [f for f in high_corr_features if target_correlations[f] < -threshold]

    # Объединяем признаки для тепловой карты (включая целевую переменную)
    all_features = ['TARGET_deathRate'] + high_corr_features

    if len(high_corr_features) > 0:
        # Создаем подмножество корреляционной матрицы
        subset_corr = correlation_matrix.loc[all_features, all_features]

        # Автоматически определяем размер графика, если не указан
        if figsize is None:
            figsize = (max(10, min(14, len(all_features) * 0.6)),
                       max(10, min(14, len(all_features) * 0.6)))

        # Строим тепловую карту
        plt.figure(figsize=figsize)
        sns.heatmap(subset_corr,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
                    annot_kws={"size": 9})

        plt.title(
            f'Тепловая карта корреляций признаков\nс корреляцией > {threshold * 100:.0f}',
            fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

        # Выводим информацию об отобранных признаках
        print(f"\n ПРИЗНАКИ С КОРРЕЛЯЦИЕЙ > {threshold * 100:.0f}% (|r| > {threshold}):")
        print("=" * 70)
        print(f"Всего отобрано {len(high_corr_features)} признаков для построения модели")
        print()

        if positive_features:
            print(" ПОЛОЖИТЕЛЬНАЯ КОРРЕЛЯЦИЯ (увеличивают смертность):")
            for feature in positive_features:
                print(f"  {feature:35} | {target_correlations[feature]:.4f}")
            print()

        if negative_features:
            print(" ОТРИЦАТЕЛЬНАЯ КОРРЕЛЯЦИЯ (снижают смертность):")
            for feature in negative_features:
                print(f"  {feature:35} | {target_correlations[feature]:.4f}")

    else:
        print(f"\n Нет признаков с корреляцией > {threshold * 100:.0f}%")
        # Показываем топ-5 признаков
        top_features = target_correlations[1:6].index.tolist() + target_correlations.tail(5).index.tolist()
        all_features = ['TARGET_deathRate'] + top_features
        subset_corr = correlation_matrix.loc[all_features, all_features]

        if figsize is None:
            figsize = (max(10, min(14, len(all_features) * 0.6)),
                       max(10, min(14, len(all_features) * 0.6)))

        plt.figure(figsize=figsize)
        sns.heatmap(subset_corr,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
                    annot_kws={"size": 9})

        plt.title(f'Тепловая карта корреляций \nнет признаков с корреляцией > {threshold * 100:.0f}%',
                  fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()


def plot_all_heatmaps(df, threshold=0.35):

    print("\n" + "=" * 80)
    print(" ПОСТРОЕНИЕ ТЕПЛОВЫХ КАРТ КОРРЕЛЯЦИЙ")
    print("=" * 80)

    print("\n1. Полная тепловая карта всех числовых признаков:")
    plot_full_correlation_heatmap(df)

    print("\n2. Детальная тепловая карта признаков с высокой корреляцией:")
    plot_high_correlation_heatmap(df, threshold)
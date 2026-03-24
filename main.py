import pandas as pd
from heatmap import plot_all_heatmaps

# Путь к файлу
file_path = r'C:\Users\Александра\PycharmProjects\ML_course\5\data\raw\cancer_reg.csv'

# Загрузка данных
try:
    df = pd.read_csv(file_path)
    print(" Файл загружен")
    print(f" Размер данных: {df.shape}")

    # Построение тепловых карт
    plot_all_heatmaps(df, threshold=0.35)

except Exception as e:
    print(f" Ошибка при загрузке: {e}")
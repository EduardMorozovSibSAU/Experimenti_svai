import pandas as pd
import sys
sys.path.insert(0, '.')
from Experiment_1 import *

cfg.DATA_TYPE = 'increment'
dm_df = load_dm_movings('new_data.xlsx', 'ДМ_перемещения приростом')
ts_df = load_closest_ts('Все параметры_12 циклов_new_26.12.xlsx', 'Ближайшие ТС_new', cfg.MAX_DISTANCE)
soil_df = load_soil_types('Все параметры_12 циклов_new_26.12.xlsx', 'Колонки_new')
temp_df = load_temperature('Все параметры_12 циклов_new_26.12.xlsx', 'Температура_new')
climate_df = load_climate('Все параметры_12 циклов_new_26.12.xlsx', 'Климат')
building_df = load_building_type('Все параметры_12 циклов_new_26.12.xlsx', 'Тип сооружения')
swamp_df = load_swamp_status('Заболоченность.csv', exist=True)

print("=== dm_df ===")
print("shape:", dm_df.shape)
print("Участок sample:", dm_df['Участок'].head(3).tolist())
print("Цикл sample:", dm_df['Цикл'].head(3).tolist(), "| dtype:", dm_df['Цикл'].dtype)

print("\n=== swamp_df ===")
print("cols:", swamp_df.columns.tolist())
print("Участок sample:", swamp_df['Участок'].head(3).tolist())

dm_with_swamp = dm_df.merge(swamp_df, on='Участок', how='left')
print("\n=== dm_with_swamp ===")
print("shape:", dm_with_swamp.shape)
print("Swamp nulls:", dm_with_swamp['Swamp'].isna().sum(), "/", len(dm_with_swamp))

temp_merged = temp_df.merge(climate_df, on='Цикл', how='left').drop(columns=['tai'], errors='ignore')
temp_features = temp_merged.columns.drop(['TS_num', 'Цикл'], errors='ignore')
temp_lagged = create_lagged_features(temp_merged, 'TS_num', 'Цикл', temp_features.tolist(), 3)
ts_with_soil = temp_lagged.merge(soil_df, on='TS_num', how='left')

print("\n=== Цикл dtype comparison ===")
print("dm_df   Цикл dtype:", dm_df['Цикл'].dtype,   "| sample:", dm_df['Цикл'].unique()[:3])
print("ts_soil Цикл dtype:", ts_with_soil['Цикл'].dtype, "| sample:", ts_with_soil['Цикл'].unique()[:3])

ts_unique = ts_df.loc[ts_df.groupby('Mark_num')['Distance'].idxmin()]
result = dm_with_swamp.merge(ts_unique[['Mark_num', 'TS_num', 'Distance']], on='Mark_num', how='left')
result = result.merge(building_df, on='Mark_num', how='left')
result = result.merge(ts_with_soil, on=['TS_num', 'Цикл'], how='left')

print("\n=== Final merge ===")
print("shape before dropna:", result.shape)
nulls = result.isna().sum().sort_values(ascending=False)
print("Top-10 null cols:")
print(nulls.head(10))
full_df = result.dropna()
print("shape after dropna:", full_df.shape)

print("\n=== Columns in full_df ===")
print(sorted(full_df.columns.tolist()))

# Проверяем что ожидает scale_all_data
temp_cols = [
    [f'temp_{d}_{lag}' for d in ['6','7','8','9','10']]
    for lag in [1, 2, 3]
]
snow_presc_temp = []
for num in range(3):
    snow_presc_temp.append(f'snowi_{num+1}')
    snow_presc_temp.append(f'preci_{num+1}')
    snow_presc_temp.append(f'tempi_{num+1}')
names_di_columns = ['di-1', 'di-2', 'di-3']
scale_cols = [*names_di_columns, *snow_presc_temp, *[c for group in temp_cols for c in group], 'Distance']
print("\n=== Expected scale_cols ===")
print(scale_cols)
missing = [c for c in scale_cols if c not in full_df.columns]
print("\n=== MISSING cols ===")
print(missing)

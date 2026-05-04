# 1. IMPORTS

import os
from pathlib import Path
from dataclasses import dataclass, field

from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import random
import matplotlib.pyplot as plt
import tqdm
import joblib


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class Config:
    """Глобальные настройки эксперимента."""
    # Пути
    ROOT: Path = Path(".") # корневая папка для всех данных и результатов (можно изменить на "Experiment_1" или другой)
    DATA_FILE: str = "data_experiment_1_scaled.csv" # итоговый файл после загрузки и предобработки, который будет использоваться для обучения
    RESULTS_FOLDER: str = "Results_experiment" # название папки для сохранения моделей, графиков и итогов (создаётся автоматически)
    
    # Данные
    WINDOW_SIZE: int = 4 # размер окна для создания лагов (di, di-1, di-2, di-3)
    WINDOW_SIZE_TEMP: int = 3 # размер окна для создания лагов (tempi-1, tempi-2, tempi-3) 
    OUTLIER_THRESHOLD: float = 12.0
    MAX_DISTANCE: float = 50.0  # метров для фильтрации ТС
    QUANTILE_LOW: float = 0.02 # нижняя граница для фильтрации выбросов по квантилям (можно менять для экспериментов)
    QUANTILE_HIGH: float = 0.98 # верхняя граница для фильтрации выбросов по квантилям (можно менять для экспериментов)
    
    TEMPERATURE_IS_ABSOLUTE: bool = False # флаг для абсолютных значений температур (для 12, 14 экспериментов)
    DI_IS_ABSOLUTE: bool = False # флаг для абсолютных значений di

    # Модель
    INPUT_DIM: int = 31 # ← для 11-14 может быть больше из-за дополнительных temp-фич
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [32, 128, 64, 64, 32]) # можно менять для экспериментов
    DROPOUT_RATE: float = 0.2 # для всех слоёв, кроме последних 2, где может быть больше
    DROPOUT_RATE_LAST: float = 0.5 # для последних 2 слоёв (можно увеличить для регуляризации)
    BATCH_SIZE: int = 256 
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    N_EPOCHS: int = 2000
    EARLY_STOP_PATIENCE: Optional[int] = None  # если нужно — раскомментировать
    
    # Тренировка
    TEST_SIZE: float = 0.2 # доля тестовой выборки при сплите
    RANDOM_STATE: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu" # устройство для обучения (GPU или CPU)
    
    # Флаги эксперимента
    NORMALIZE_DI_LAGS: bool = True       # False → не нормализовать di-1, di-2, di-3
    TARGET_ABSOLUTE: bool = False        # True  → предсказывать abs(di)
    SWAMP_AS_PERCENT: bool = False       # True  → заболоченность как процент (0–100), иначе категория 0/1/2
    DATA_TYPE: str = "increment"         # "increment" — приростом, "sequential" — последовательно
    # Почвенные кластеры для анализа
    SOIL_STACK: set = frozenset([
        '1111112222222222', '1111113333322222', '1111114442222222',
        '1111122222222222', '1111122222333222', '1111144422222222',
        '1114422222322222', '3364444555550000'
    ])
    
    # Циклы
    MONTHS: List[str] = field(default_factory=lambda: [
    "II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"
    ])
    ROMAN_TO_INT: Dict[str, int] = field(default_factory=lambda: {
    'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,
    'VII':7,'VIII':8,'IX':9,'X':10,'XI':11,'XII':12, 'XIII':13
    })


cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Полезные функции
# ─────────────────────────────────────────────────────────────────────────────

def truncate_string(x: str, length: int) -> str:
    """Обрезает строку до заданной длины."""
    return x[:length] if isinstance(x, str) else x

def convert_cycle_format(cycle: str, mapping: Dict[str, int], reverse: bool = False) -> str:
    """Конвертирует номер цикла между римскими и арабскими цифрами."""

def create_lagged_features(
    df: pd.DataFrame, 
    group_col: str, 
    time_col: str, 
    feature_cols: List[str], 
    num_lags: int
) -> pd.DataFrame:
    
    lags = list(range(1, num_lags))
    """Создаёт лаговые признаки для временного ряда."""
    df = df.copy()
    df[time_col] = pd.Categorical(
        df[time_col], 
        categories=cfg.MONTHS, 
        ordered=True
    )
    df = df.sort_values([group_col, time_col])
    
    for lag in lags:
        shifted = (
            df.groupby(group_col)[feature_cols]
            .shift(lag)
            .add_suffix(f'_{lag+1}')
        )
        df = pd.concat([df, shifted], axis=1)
    
    # Переименовываем исходные фичи с суффиксом _1
    df = df.rename(columns={col: f"{col}_1" for col in feature_cols})
    return df.reset_index(drop=True)

def filter_outliers_by_diff(df: pd.DataFrame, cols: List[str], threshold: float) -> pd.DataFrame:
    """Удаляет выбросы по максимальной разнице между соседними значениями."""
    diff = df[cols].diff(axis=1).abs()
    return df[diff.max(axis=1) < threshold].copy()


def filter_outliers_by_quantile(df: pd.DataFrame, cols: List[str],  low: float = 0.02,  high: float = 0.98) -> pd.DataFrame:
    """Удаляет выбросы по квантильному размаху."""
    lower = df[cols].quantile(low, numeric_only=False)
    upper = df[cols].quantile(high, numeric_only=False)
    sub = df[cols]
    sub, lower = sub.align(lower, axis=1, copy=False)
    sub, upper = sub.align(upper, axis=1, copy=False)
    mask = (sub >= lower).all(axis=1) & (sub <= upper).all(axis=1)
    return df[mask].copy()



def cluster_soil_types(df: pd.DataFrame, soil_cols: List[str], min_cluster_size: int = 0) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """Группирует строки по уникальным комбинациям типов почв."""

    soil_keys = df[soil_cols].apply(lambda row: tuple(row.values), axis=1)

    labels, uniques = pd.factorize(soil_keys)
    
    counts = np.bincount(labels)
    valid_mask = counts >= min_cluster_size
    valid_ids = np.where(valid_mask)[0]
    
    return pd.Series(labels, index=df.index), valid_ids, uniques[valid_mask]



# ─────────────────────────────────────────────────────────────────────────────
# 4. Загрузка данных и предобработка
# ─────────────────────────────────────────────────────────────────────────────

def load_dm_movings(filepath: str, sheet_name: str, filter_func: str = None, num_cols = 16) -> pd.DataFrame:
    """Загружает и преобразует данные о смещениях ДМ"""
    df_new = pd.read_excel(filepath, sheet_name=sheet_name)
    if num_cols < 16:
        df_new.columns = ['Объект',
                        'Участок',
                        'Номер ДМ', 
                        'Имя ДМ', 
                        'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
    else:
        df_new.columns = ['Объект',
                        'Участок',
                        'Номер ДМ',
                        'Имя ДМ',
                        'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII','XIII']
    
    
    df_new = df_new.drop(0)
    
    base_cols = ["Объект", "Участок", "Номер ДМ", "Имя ДМ"]
    months_rev = cfg.MONTHS[::-1]
    results = []
    
    names_di_columns = ['di']
    for num in range(cfg.WINDOW_SIZE):
        if num > 0:
            names_di_columns.append(f'di-{num}')


    for i in range(len(months_rev) - cfg.WINDOW_SIZE + 1):
        cols = months_rev[i:i+cfg.WINDOW_SIZE]
        temp = df_new[base_cols + cols].copy()
        temp['Цикл'] = months_rev[i]
        temp.rename(columns=dict(zip(cols, names_di_columns)), inplace=True)
        results.append(temp)
    
    final = pd.concat(results, ignore_index=True)
    
    # Нормализация названий и очистка
    rename_map = {'Объект': 'Site', 'Номер ДМ': 'Mark_num', 'Имя ДМ': 'Mark_name'}
    final = final.rename(columns=rename_map).dropna()
    final['Site'] = final['Site'].apply(lambda x: truncate_string(x, 7))
    final['Участок'] = final['Участок'].apply(lambda x: truncate_string(x, 10))
    
    # Фильтрация выбросов
    if filter_func == 'diff':
        final = filter_outliers_by_diff(final, names_di_columns, cfg.OUTLIER_THRESHOLD)
    elif filter_func == 'quantile' or filter_func is None:
        final = filter_outliers_by_quantile(final, names_di_columns, cfg.QUANTILE_LOW, cfg.QUANTILE_HIGH)

    if cfg.DI_IS_ABSOLUTE:
        final[['di', 'di-1', 'di-2', 'di-3']] = final[['di', 'di-1', 'di-2', 'di-3']].apply(abs)

    return final


def load_closest_ts(filepath: str, sheet_name: str = None, max_distance: float = None) -> pd.DataFrame:
    """Загружает данные о ближайших температурных станциях."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.drop(columns=['Ранг'], errors='ignore').dropna()
    
    if max_distance:
        df = df[df['Расстояние (метры)'] <= max_distance]
    
    rename_map = {
        'Номер ТС': 'TS_num', 'Объект': 'Site', 'Имя ТС': 'TS_name',
        'Расстояние (метры)': 'Distance', 'Номер ДМ': 'Mark_num', 'Имя ДМ': 'Mark_name'
    }
    return df.rename(columns=rename_map)


def load_soil_types(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """Загружает типы почв на разных глубинах."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.drop(columns=['Колонка16', 'Колонка10'], errors='ignore').dropna()
    
    new_columns = {
    'Номер ТС': 'TS_num',
    'Вид грунта 0': 'SoilType_0',
    'Вид грунта 0.5': 'SoilType_05',
    'Вид грунта 1': 'SoilType_1',
    'Вид грунта 1.5': 'SoilType_15',
    'Вид грунта 2': 'SoilType_2',
    'Вид грунта 2.5': 'SoilType_25',
    'Вид грунта 3': 'SoilType_3',
    'Вид грунта 3.5': 'SoilType_35',
    'Вид грунта 4': 'SoilType_4',
    'Вид грунта 4.5': 'SoilType_45',
    'Вид грунта 5': 'SoilType_5',
    'Вид грунта 6': 'SoilType_6',
    'Вид грунта 7': 'SoilType_7',
    'Вид грунта 8': 'SoilType_8',
    'Вид грунта 9': 'SoilType_9',
    'Вид грунта 10': 'SoilType_10'
}
    return df.rename(columns=new_columns)

def load_temperature_growth(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    df_temp_new = pd.read_excel(filepath, sheet_name=sheet_name)
    df_temp_new.drop(0, inplace=True)
    df_temp_new.columns = ['Объект', 'Участок', 'TS_num', 'Имя ТС', 'Depth', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII']

    id_cols = ['Объект', 'Участок', 'TS_num', 'Имя ТС', 'Depth']

    df_long = df_temp_new.melt(
        id_vars=id_cols,
        var_name='Цикл',
        value_name='value'
    )

    df_final = df_long.pivot_table(
        index=['TS_num', 'Цикл'],
        columns='Depth',
        values='value',
        aggfunc='first'
    ).reset_index()

    df_final.columns.name = None
    df_temp_new = df_final.rename(columns=lambda x: f'temp_{int(x)}' if isinstance(x, (int, float)) else x)

    cols = ['temp_6', 'temp_7', 'temp_8', 'temp_9', 'temp_10']
    low = df_temp_new[cols].quantile(0.02)
    high = df_temp_new[cols].quantile(0.98)
    df_temp_new = df_temp_new[
        (df_temp_new[cols] >= low).all(axis=1) &
        (df_temp_new[cols] <= high).all(axis=1)
    ].copy()

    if cfg.TEMPERATURE_IS_ABSOLUTE:
        df_temp_new[cols] = df_temp_new[cols].apply(abs)

    return df_temp_new


def load_temperature(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """Загружает температурные данные с конвертацией циклов."""
    df_temp_new = pd.read_excel(filepath, sheet_name=sheet_name)
    new_columns = {
        'Номер ТС': 'TS_num',
        '0': 'temp_0',
        '0.5': 'temp_05',
        '1': 'temp_1',
        '1.5': 'temp_15',
        '2': 'temp_2',
        '2.5': 'temp_25',
        '3': 'temp_3',
        '3.5': 'temp_35',
        '4': 'temp_4',
        '4.5': 'temp_45',
        '5': 'temp_5',
        '6': 'temp_6',
        '7': 'temp_7',
        '8': 'temp_8',
        '9': 'temp_9',
        '10': 'temp_10',
        'Тср,°C (3.0-10.0 м)': 'tai'
    }

    roman_to_int = {
        'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,
        'VII':7,'VIII':8,'IX':9,'X':10,'XI':11,'XII':12, 'XIII':13
    }

    int_to_roman = {v:k for k,v in roman_to_int.items()}

    df_temp_new["Цикл"] = df_temp_new["Цикл"].map(roman_to_int) + 1
    df_temp_new["Цикл"] = df_temp_new["Цикл"].map(int_to_roman)

    # df_temp_new = df_temp_new[df_temp_new['Цикл'].isin(['X', 'XI', 'XII'])].dropna() # берем только 3 последних цикла
    df_temp_new = df_temp_new.rename(columns=new_columns) # Применяем переименование
    df_temp_new = df_temp_new.drop(['temp_0', 'temp_05', 'temp_1', 'temp_15', 'temp_2', 'temp_25', 'temp_3', 'temp_35', 'temp_4', 'temp_45', 'temp_5'], 
                                    axis =1)

    df_temp_new = df_temp_new[df_temp_new['Цикл'] != 'XIII']
    return df_temp_new


def load_climate(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """Загружает климатические данные (снег, осадки, температура)."""
    df = pd.read_excel(filepath, sheet_name=sheet_name).T
    df.reset_index(inplace=True)
    df.columns = ["Цикл", "snowi", "preci", "tempi"]
    return df.iloc[1:].reset_index(drop=True)


def load_swamp_status(filepath: str, exist: bool = True) -> pd.DataFrame:
    """Загружает и кодирует статус заболоченности."""
    if exist:
        return pd.read_csv(filepath).drop("Unnamed: 0", axis = 1)
    else:
        df = pd.read_csv(filepath).drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
        df = df.iloc[5:].copy()
        df.columns = ['Site', 'Участок', 'Swamp']
        
        # Нормализация названий участков
        df['Site'] = df['Site'].apply(lambda x: truncate_string(x, 7))
        df.loc[25, 'Site'] = "ГТМ.018"  # фикс известного пропуска
        
        # Кодирование: 0=нет, 1=частично, 2=да
        df['Swamp'] = df['Swamp'].replace({'Нет':0, 'нет':0, 'Скорее нет':0, 'да':2})
        df['Swamp'] = df['Swamp'].apply(lambda x: 1 if isinstance(x, str) else x)
        
        # Генерация уникальных идентификаторов участков
        df['Участок'] = df['Участок'].str.split().str[0]
        df.loc[:34, 'Участок'] = (
            df.loc[:34, 'Site'] + '.' + 
            (df.loc[:34].groupby('Site').cumcount() + 1).astype(str).str.zfill(2)
        )
        return df.drop(columns=['Site']).reset_index(drop=True)
    
def load_swamp_percent(filepath: str) -> pd.DataFrame:
    """Загружает процент заболоченности и возвращает DataFrame с колонками Участок, Swamp."""
    df = pd.read_excel(filepath)
    df = df[['Номер участка', '% заболоченности территории']].copy()
    df.columns = ['Участок', 'Swamp']
    return df.dropna().reset_index(drop=True)


def load_building_type(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """Загружает тип сооружения (линейное/нелинейное)."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df['Тип сооружения'] = df['Тип сооружения'].apply(lambda x: 0 if x == 'Линейное' else 1)
    return df.rename(columns={'Тип сооружения': 'Geometry', 'Номер ДМ': 'Mark_num'})

def merge_all_data(
    dm_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    soil_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    swamp_df: pd.DataFrame,
    building_df: pd.DataFrame
) -> pd.DataFrame:
    """Объединяет все источники данных в единый датафрейм."""

    # 1. Температуры + климат + лаги
    temp_merged = temp_df.merge(climate_df, on='Цикл', how='left').drop(columns=['tai'], errors='ignore')
    temp_features = temp_merged.columns.drop(['TS_num', 'Цикл'], errors='ignore')
    temp_lagged = create_lagged_features(temp_merged, 'TS_num', 'Цикл', temp_features.tolist(), 3)
    
    # 2. Температуры + типы почв
    ts_with_soil = temp_lagged.merge(soil_df, on='TS_num', how='left')
    
    # 3. Смещения + заболоченность
    dm_with_swamp = dm_df.merge(swamp_df, on='Участок', how='left')
    
    # 4. Выбор ближайшей ТС для каждого ДМ
    ts_unique = ts_df.loc[ts_df.groupby("Mark_num")["Distance"].idxmin()]
    
    # 5. Финальный мердж
    result = dm_with_swamp.merge(
        ts_unique[['Mark_num', 'TS_num', 'Distance']], 
        on='Mark_num', how='left'
    )
    result = result.merge(building_df, on='Mark_num', how='left')
    result = result.merge(ts_with_soil, on=['TS_num', 'Цикл'], how='left')
    
    # Финальная очистка и переименование
    result = result.rename(columns={'Цикл': 'Cycle'}).dropna()

    return result
    

def scale_all_data(dataframe):
    
    temp_cols = [
        [f'temp_{d}_{lag}' for d in ['6','7','8','9','10']] 
        for lag in list(range(cfg.WINDOW_SIZE_TEMP + 1))[1:]
    ]
    snow_presc_temp = [] # создается столько прееменных сколько окно
    for num in range(cfg.WINDOW_SIZE_TEMP):
        snow_presc_temp.append(f'snowi_{num+1}')
        snow_presc_temp.append(f'preci_{num+1}')
        snow_presc_temp.append(f'tempi_{num+1}')

    names_di_columns = [] # создается столько прееменных сколько окно
    for num in range(cfg.WINDOW_SIZE):
        if num > 0:
            names_di_columns.append(f'di-{num}')

    soil_cols = [f'SoilType_{d}' for d in ['0','05','1','15','2','25','3','35','4','45','5','6','7','8','9','10']]

    scale_cols = [
        *(names_di_columns if cfg.NORMALIZE_DI_LAGS else []),
        *snow_presc_temp, *[c for group in temp_cols for c in group], 'Distance']

    scaler = StandardScaler()
    dataframe[scale_cols] = scaler.fit_transform(dataframe[scale_cols])  # ← для Exp.2: применить только к части колонок
    dataframe[soil_cols] = dataframe[soil_cols].astype(int)  # кодируем типы почв как int (можно оставить как есть для one-hot)
    joblib.dump(scaler, cfg.ROOT / "scaler.pkl")
    
    dataframe.to_csv(cfg.ROOT / cfg.DATA_FILE, index=False)
    return dataframe


# ─────────────────────────────────────────────────────────────────────────────
# 5. Создение модели
# ─────────────────────────────────────────────────────────────────────────────

class NeuralNet(nn.Module):
    """Многослойный перцептрон для регрессии смещений"""
    
    def __init__(self, input_dim: int = 33, hidden_dims: List[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = cfg.HIDDEN_DIMS
        
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims)-2 else cfg.DROPOUT_RATE_LAST)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GTMDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()

        target_is_absolute = cfg.TARGET_ABSOLUTE
        swamp_percent = cfg.SWAMP_AS_PERCENT
        
        # Группы колонок
        temp_cols = [
            [f'temp_{d}_{lag}' for d in ['6','7','8','9','10']] 
            for lag in list(range(cfg.WINDOW_SIZE_TEMP + 1))[1:]
        ]

        snow_presc_temp = [] # создается столько прееменных сколько окно
        for num in range(cfg.WINDOW_SIZE_TEMP):
            snow_presc_temp.append(f'snowi_{num+1}')
            snow_presc_temp.append(f'preci_{num+1}')
            snow_presc_temp.append(f'tempi_{num+1}')

        names_di_columns = [] # создается столько прееменных сколько окно
        for num in range(cfg.WINDOW_SIZE):
            if num > 0:
                names_di_columns.append(f'di-{num}')

        other_cols = [
            *names_di_columns, *snow_presc_temp,
            'Distance', 'Geometry', 
            'Swamp'
        ]
        
        if swamp_percent:
            data = pd.concat([data, pd.DataFrame([data.iloc[-1]]*2)], ignore_index=True)
            data.iloc[-2, data.columns.get_loc('Geometry')] = 0
            data.iloc[-1, data.columns.get_loc('Geometry')] = 1
            
            # Формирование признаков и таргета
            X = data[[*other_cols, *[c for group in temp_cols for c in group]]]
            X = pd.get_dummies(X, columns=['Geometry'], dtype=float)
            y = data['di'].abs() if target_is_absolute else data['di']
            meta = data[['Mark_num', 'Cycle']].values.tolist()
            
            # Убираем фиктивные строки
            X = X.iloc[:-2].reset_index(drop=True)
            y = y.iloc[:-2].reset_index(drop=True)
            meta = meta[:-2]
            
            # Тензоры
            self.X = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
            self.y = torch.tensor(y.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)
            self.meta = meta
            self.input_dim = self.X.shape[1]

        else:   
            data = pd.concat([data, pd.DataFrame([data.iloc[-1]]*3)], ignore_index=True)
            data.iloc[-3, data.columns.get_loc('Swamp')] = 2
            data.iloc[-2, data.columns.get_loc('Swamp')] = 1
            data.iloc[-1, data.columns.get_loc('Swamp')] = 0
            data.iloc[-2, data.columns.get_loc('Geometry')] = 0
            data.iloc[-1, data.columns.get_loc('Geometry')] = 1
            
            # Формирование признаков и таргета
            X = data[[*other_cols, *[c for group in temp_cols for c in group]]]
            X = pd.get_dummies(X, columns=['Geometry', 'Swamp'], dtype=float)
            y = data['di'].abs() if target_is_absolute else data['di']
            meta = data[['Mark_num', 'Cycle']].values.tolist()
            
            # Убираем фиктивные строки
            X = X.iloc[:-3].reset_index(drop=True)
            y = y.iloc[:-3].reset_index(drop=True)
            meta = meta[:-3]
            
            # Тензоры
            self.X = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
            self.y = torch.tensor(y.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)
            self.meta = meta
            self.input_dim = self.X.shape[1]
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List]:
        return self.X[idx], self.y[idx], self.meta[idx]
    

# ─────────────────────────────────────────────────────────────────────────────
# 5. Подготовка данных 
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    filepath: str,
    cycles_to_include: Optional[List[str]] = None,
    max_distance: Optional[float] = None,
    min_cluster_size: int = 0
) -> List[Tuple[int, pd.DataFrame, str]]:
    """
    Подготавливает данные для обучения/тестирования, группируя по типам почв.
    
    Args:
        filepath: путь к CSV с данными
        cycles_to_include: список циклов для фильтрации (None = все)
        max_distance: максимальное расстояние ДМ-ТС для фильтрации
        min_cluster_size: фильтр кластера если количество элементов < min_cluster_size
    
    Returns:
        Список кортежей (cluster_id, dataframe, soil_code_string)
    """
    df = pd.read_csv(filepath)


    if cycles_to_include:
        df = df[df['Cycle'].isin(cycles_to_include)]
    if max_distance:
        df = df[df['Distance'] <= max_distance]
    
    soil_cols = [f'SoilType_{d}' for d in ['0','05','1','15','2','25','3','35','4','45','5','6','7','8','9','10']]
    labels, valid_ids, uniques = cluster_soil_types(df, soil_cols, min_cluster_size)
    
    result = []
    for cid, soil_tuple in zip(valid_ids, uniques):
        soil_code = ''.join(map(str, soil_tuple))
        subset = df[labels == cid].copy()
        result.append((int(cid), subset, soil_code))
    
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 7. Обучение
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    data_groups: List[Tuple[int, pd.DataFrame, str]],
    output_folder: Path,
    soil_filter: set = None
) -> pd.DataFrame:
    """Обучает отдельные модели для каждого почвенного кластера."""
    output_folder.mkdir(parents=True, exist_ok=True)
    summary = []
    

    for cluster_id, cluster_df, soil_code in data_groups:
        if soil_filter and soil_code not in soil_filter:
            continue
        
        # Train/test split
        train_df, test_df = train_test_split(
            cluster_df, test_size=cfg.TEST_SIZE, 
            random_state=cfg.RANDOM_STATE, shuffle=True
        )
        
        # Datasets & loaders
        train_ds = GTMDataset(train_df)
        test_ds = GTMDataset(test_df)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False)
        
        # Model setup
        model = NeuralNet(input_dim=cfg.INPUT_DIM).to(cfg.DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        
        # Training
        train_losses, test_losses = [], []
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in tqdm.tqdm(range(cfg.N_EPOCHS), desc=f"Train {soil_code}", leave=False):
            # Train
            model.train()
            epoch_train_loss = 0
            for x, y, _ in train_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                optimizer.zero_grad()
                loss = torch.sqrt(criterion(model(x), y))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            train_losses.append(epoch_train_loss / len(train_loader))
            
            # Eval
            model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for x, y, _ in test_loader:
                    x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                    loss = torch.sqrt(criterion(model(x), y))
                    epoch_test_loss += loss.item()
            test_losses.append(epoch_test_loss / len(test_loader))
            
            # Save best
            if epoch_test_loss < best_loss:
                best_loss = epoch_test_loss
                best_model_state = model.state_dict()
            
            # Early stopping (опционально)
            if cfg.EARLY_STOP_PATIENCE and epoch - train_losses.index(min(train_losses)) > cfg.EARLY_STOP_PATIENCE:
                break
        
        # Plot losses
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(train_losses); ax1.set(title='Train RMSE', xlabel='Epoch', ylabel='Loss')
        ax2.plot(test_losses); ax2.set(title='Test RMSE', xlabel='Epoch', ylabel='Loss')
        fig.suptitle(f"Soil: {soil_code}")
        fig.savefig(output_folder / f"loss_{cluster_id}_{soil_code}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Final evaluation
        model.load_state_dict(best_model_state)
        model.eval()
        all_preds, all_actual, all_meta = [], [], []
        
        with torch.no_grad():
            for x, y, meta in DataLoader(test_ds, batch_size=1024, shuffle=False):
                pred = model(x.to(cfg.DEVICE)).cpu()
                all_preds.extend(pred.squeeze().tolist())
                all_actual.extend(y.squeeze().tolist())
                all_meta.extend(meta)
        
        # Metrics
        r2 = r2_score(all_actual, all_preds)
        rmse = np.sqrt(np.mean((np.array(all_actual) - np.array(all_preds))**2))
        mae = mean_absolute_error(all_actual, all_preds)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'DM': [*all_meta[0]],
            'Cycle': [*all_meta[1]],
            'Actual': all_actual,
            'Predicted': all_preds
        })

        pred_df.to_excel(output_folder / f"preds_{soil_code}.xlsx", index=False)
        torch.save(model.state_dict(), output_folder / f"model_{soil_code}.pt")
        
        # Log
        print(f"\n[{soil_code}] DM:{len(cluster_df['Mark_num'].unique())} | "
              f"R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
        
        summary.append({
            'Soil_types': soil_code,
            'DM': cluster_df['Mark_num'].nunique(),
            'Rows': len(cluster_df),
            'R2': r2, 'RMSE': rmse, 'MAE': mae
        })
    
        
    # print(summary)
    summary_df = pd.DataFrame(summary).sort_values('Rows', ascending=False)
    summary_df.to_excel(output_folder / 'summary.xlsx', index=False)
    return summary_df


def evaluate_on_cycles(
    data_groups: List[Tuple[int, pd.DataFrame, str]],
    model_folder: Path,
    soil_filter: set = None,
    output_prefix: str = ""
) -> pd.DataFrame:
    """Оценивает обученные модели на указанных циклах."""
    summary = []
    
    for cluster_id, cluster_df, soil_code in data_groups:
        if soil_filter and soil_code not in soil_filter:
            continue
        
        model_path = model_folder / f"model_{soil_code}.pt"
        if not model_path.exists():
            continue
        
        # Load model
        model = NeuralNet(input_dim=cfg.INPUT_DIM).to(cfg.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE, weights_only=True))
        model.eval()
        
        # Evaluate
        test_ds = GTMDataset(cluster_df)
        test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
        
        preds, actuals, metas = [], [], []
        with torch.no_grad():
            for x, y, meta in test_loader:
                pred = model(x.to(cfg.DEVICE)).cpu()
                preds.extend(pred.squeeze().tolist())
                actuals.extend(y.squeeze().tolist())
                metas.extend(meta)
        
        r2 = r2_score(actuals, preds)
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds))**2))
        mae = mean_absolute_error(actuals, preds)
        
        print(f"[{output_prefix}{soil_code}] R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
        
        summary.append({
            'Soil_types': soil_code,
            'DM': cluster_df['Mark_num'].nunique(),
            'Rows': len(cluster_df),
            'R2': r2, 'RMSE': rmse, 'MAE': mae
        })
    
    return pd.DataFrame(summary).sort_values('Rows', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Пресеты экспериментов
# ─────────────────────────────────────────────────────────────────────────────

# Каждый пресет — словарь с полями, которые перезаписывают дефолты Config.
# DATA_TYPE:         "increment"  — лист "ДМ_перемещения приростом"
#                    "sequential" — лист "ДМ_перемещения последовательно"
# NORMALIZE_DI_LAGS: True  — нормализовать di-1, di-2, di-3
#                    False — оставить как есть
# TARGET_ABSOLUTE:   True  — предсказывать abs(di) как таргет
# DI_IS_ABSOLUTE:    True  — применить abs() к di-лагам как признакам
# SWAMP_AS_PERCENT:  True  — заболоченность как процент (файл .xlsx)
#                    False — категория 0/1/2 (файл .csv)

EXPERIMENT_CONFIGS: Dict[int, dict] = {
    1: dict(DATA_TYPE="increment",  NORMALIZE_DI_LAGS=True,  TARGET_ABSOLUTE=False, SWAMP_AS_PERCENT=False),
    2: dict(DATA_TYPE="increment",  NORMALIZE_DI_LAGS=False, TARGET_ABSOLUTE=False, SWAMP_AS_PERCENT=False),
    3: dict(DATA_TYPE="sequential", NORMALIZE_DI_LAGS=False, TARGET_ABSOLUTE=False, SWAMP_AS_PERCENT=False),
    4: dict(DATA_TYPE="increment",  NORMALIZE_DI_LAGS=True,  TARGET_ABSOLUTE=True,  SWAMP_AS_PERCENT=False, DI_IS_ABSOLUTE=True),   # abs(di) и в признаках, и в таргете
    5: dict(DATA_TYPE="increment",  NORMALIZE_DI_LAGS=True,  TARGET_ABSOLUTE=True,  SWAMP_AS_PERCENT=False, DI_IS_ABSOLUTE=False),  # abs(di) только в таргете
    6: dict(DATA_TYPE="sequential", NORMALIZE_DI_LAGS=True,  TARGET_ABSOLUTE=True,  SWAMP_AS_PERCENT=False),
    7: dict(DATA_TYPE="sequential", NORMALIZE_DI_LAGS=True,  TARGET_ABSOLUTE=False, SWAMP_AS_PERCENT=True),
    8: dict(DATA_TYPE="sequential", NORMALIZE_DI_LAGS=True,  TARGET_ABSOLUTE=False, SWAMP_AS_PERCENT=False),
    9: dict(DATA_TYPE="increment",  NORMALIZE_DI_LAGS=False, TARGET_ABSOLUTE=False, SWAMP_AS_PERCENT=True),
}

DM_SHEET: Dict[str, str] = {
    "increment":  "ДМ_перемещения приростом",
    "sequential": "ДМ_перемещения последовательно",
}


def main(exp_id: int = 1):

    # Применяем пресет эксперимента к глобальному cfg
    if exp_id in EXPERIMENT_CONFIGS:
        for key, val in EXPERIMENT_CONFIGS[exp_id].items():
            object.__setattr__(cfg, key, val)

    seed_everything(cfg.RANDOM_STATE)

    print(f"Using device: {cfg.DEVICE} | Experiment #{exp_id}")
    print(f"  DATA_TYPE={cfg.DATA_TYPE} | NORMALIZE_DI_LAGS={cfg.NORMALIZE_DI_LAGS} | "
          f"TARGET_ABSOLUTE={cfg.TARGET_ABSOLUTE} | SWAMP_AS_PERCENT={cfg.SWAMP_AS_PERCENT}")

    # 1. ЗАГРУЗКА ДАННЫХ
    print("Загрузка данных...")
    dm_sheet = DM_SHEET[cfg.DATA_TYPE]
    if dm_sheet == "ДМ_перемещения последовательно":
        dm_df = load_dm_movings("Все параметры_12 циклов_new_26.12.xlsx", "Перемещения", num_cols = 15)
    else:
        dm_df = load_dm_movings("new_data.xlsx", dm_sheet)

    ts_df = load_closest_ts("Все параметры_12 циклов_new_26.12.xlsx","Ближайшие ТС_new", cfg.MAX_DISTANCE)
    soil_df = load_soil_types("Все параметры_12 циклов_new_26.12.xlsx", "Колонки_new")
    temp_df = load_temperature("Все параметры_12 циклов_new_26.12.xlsx", "Температура_new")
    climate_df = load_climate("Все параметры_12 циклов_new_26.12.xlsx", "Климат")
    building_df = load_building_type("Все параметры_12 циклов_new_26.12.xlsx","Тип сооружения")

    if cfg.SWAMP_AS_PERCENT:
        swamp_df = load_swamp_percent("Заболоченность_процент.xlsx")
    else:
        swamp_df = load_swamp_status("Заболоченность.csv", exist=True)

    # 2. ОБЪЕДИНЕНИЕ

    print("Объединение данных...")
    full_df = merge_all_data(dm_df, ts_df, soil_df, temp_df, climate_df, swamp_df, building_df)

    # Сохранение (опционально)
    # full_df.to_csv(cfg.ROOT / "full_data.csv", index=False)

    # 3. Нормализация
    print("Нормализация данных...")
    full_df = scale_all_data(full_df)

    print("Обучение модели...")
    
    train_groups = prepare_data(cfg.ROOT / cfg.DATA_FILE, cycles_to_include=["II","III","IV","V","VI","VII","VIII","IX","X"])

    train_summary = train_model(
        train_groups, 
        output_folder=cfg.ROOT / cfg.RESULTS_FOLDER / f"exp_{exp_id}",
        soil_filter=cfg.SOIL_STACK
    )

    print(train_summary)

    print("Тестированиие на 11 цикле...")
    test_11 = prepare_data(cfg.ROOT / cfg.DATA_FILE, cycles_to_include=['XI'])
    eval_11 = evaluate_on_cycles(
        test_11, 
        cfg.ROOT / cfg.RESULTS_FOLDER / f"exp_{exp_id}",  # ← папка текущего эксперимента
        cfg.SOIL_STACK, 
        output_prefix=f"Exp{exp_id}_Cycle11_"
    )
    eval_11.to_excel(cfg.ROOT / cfg.RESULTS_FOLDER / f"exp_{exp_id}" / "eval_cycle_XI.xlsx", index=False)
    eval_11.to_excel(cfg.ROOT / cfg.RESULTS_FOLDER / f"exp_{exp_id}" / "reg_summary_11.xlsx", index=False)
    
    print("Тестированиие на 12 цикле...")
    test_12 = prepare_data(cfg.ROOT / cfg.DATA_FILE, cycles_to_include=['XII'])
    eval_12 = evaluate_on_cycles(
        test_12,
        cfg.ROOT / cfg.RESULTS_FOLDER / f"exp_{exp_id}",
        cfg.SOIL_STACK,
        output_prefix=f"Exp{exp_id}_Cycle12_"
    )
    eval_12.to_excel(cfg.ROOT / cfg.RESULTS_FOLDER / f"exp_{exp_id}" / "eval_cycle_XII.xlsx", index=False)
    eval_12.to_excel(cfg.ROOT / cfg.RESULTS_FOLDER / f"exp_{exp_id}" / "reg_summary_12.xlsx", index=False)



if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("experiments.log", encoding="utf-8"),
        ]
    )

    # Запустить конкретный эксперимент: python Experiment_1.py 3
    # Запустить все:                    python Experiment_1.py
    if len(sys.argv) > 1:
        ids_to_run = [int(x) for x in sys.argv[1:]]
    else:
        ids_to_run = sorted(EXPERIMENT_CONFIGS.keys())

    for exp_id in ids_to_run:
        logging.info(f"{'='*60}")
        logging.info(f"НАЧАЛО эксперимента #{exp_id}")
        logging.info(f"{'='*60}")
        try:
            main(exp_id)
            logging.info(f"ЗАВЕРШЁН эксперимент #{exp_id}")
        except Exception as e:
            logging.exception(f"ОШИБКА в эксперименте #{exp_id}: {e}")
            logging.info("Продолжаем со следующим экспериментом...")
        finally:
            globals()['cfg'] = Config()

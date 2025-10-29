import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os


class ApplicationDataPreprocessor:
    """
    Класс предобработки таблиц application_train.csv и application_test.csv.
    Упрощенная версия с простой импутацией.
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=True):
        self.file_directory = file_directory if file_directory is not None else ''
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.start = None

    def load_dataframes(self):
        """Загрузить application_train.csv и application_test.csv."""
        if self.verbose:
            self.start = datetime.now()
            print('#' * 60)
            print('#   Загрузка application_train.csv и application_test.csv   #')
            print('#' * 60)
            print("Загрузка в память...")

        self.application_train = pd.read_csv(self.file_directory + 'application_train.csv')
        self.application_test = pd.read_csv(self.file_directory + 'application_test.csv')
        self.initial_shape = self.application_train.shape

        if self.verbose:
            print("Загружено.")
            print(f"Время загрузки: {datetime.now() - self.start}")

    def data_cleaning(self):
        """Очистка данных."""
        if self.verbose:
            print("\nВыполняется очистка данных...")

        # 1) удалить FLAG_DOCUMENT с низкой дисперсией
        flag_cols_to_drop = [
            'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_20'
        ]
        self.application_train = self.application_train.drop(columns=flag_cols_to_drop, errors='ignore')
        self.application_test = self.application_test.drop(columns=flag_cols_to_drop, errors='ignore')

        # 2) DAYS_BIRTH -> годы
        if 'DAYS_BIRTH' in self.application_train.columns:
            self.application_train['DAYS_BIRTH'] = pd.to_numeric(self.application_train['DAYS_BIRTH'], errors='coerce')
            self.application_train['DAYS_BIRTH'] = self.application_train['DAYS_BIRTH'].apply(
                lambda x: (-x) / 365 if pd.notna(x) else np.nan)
        if 'DAYS_BIRTH' in self.application_test.columns:
            self.application_test['DAYS_BIRTH'] = pd.to_numeric(self.application_test['DAYS_BIRTH'], errors='coerce')
            self.application_test['DAYS_BIRTH'] = self.application_test['DAYS_BIRTH'].apply(
                lambda x: (-x) / 365 if pd.notna(x) else np.nan)

        # 3) DAYS_EMPLOYED: заменить аномалию 365243 на NaN
        for df in (self.application_train, self.application_test):
            if 'DAYS_EMPLOYED' in df.columns:
                df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
                df['DAYS_EMPLOYED'] = pd.to_numeric(df['DAYS_EMPLOYED'], errors='coerce')
                df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: (-x) / 365 if pd.notna(x) else np.nan)

        # 4) OBS_*: значения > 30 -> NaN
        for col in ['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']:
            if col in self.application_train.columns:
                self.application_train.loc[self.application_train[col] > 30, col] = np.nan
            if col in self.application_test.columns:
                self.application_test.loc[self.application_test[col] > 30, col] = np.nan

        # 5) удалить строки с CODE_GENDER == 'XNA' в train и соответствующие строки из target
        if 'CODE_GENDER' in self.application_train.columns:
            # Сохраняем индексы для удаления из target
            xna_mask = self.application_train['CODE_GENDER'] == 'XNA'
            xna_indices = self.application_train[xna_mask].index

            # Удаляем из X_train
            self.application_train = self.application_train[~xna_mask]

            # Удаляем соответствующие строки из target
            if hasattr(self, 'application_train_target'):
                self.application_train_target = self.application_train_target.drop(xna_indices)

            if self.verbose:
                print(f"Удалено строк с CODE_GENDER == 'XNA': {len(xna_indices)}")

        # 6) заполнить категориальные пропуски 'XNA'
        categorical_columns = list(set(self.application_train.select_dtypes(include=['object']).columns) |
                                   set(self.application_test.select_dtypes(include=['object']).columns))
        for col in categorical_columns:
            if col not in self.application_train.columns:
                self.application_train[col] = 'XNA'
            if col not in self.application_test.columns:
                self.application_test[col] = 'XNA'
            self.application_train[col] = self.application_train[col].fillna('XNA').astype(str)
            self.application_test[col] = self.application_test[col].fillna('XNA').astype(str)

        # 7) REGION_RATING_*: привести к числу
        for col in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']:
            if col in self.application_train.columns:
                self.application_train[col] = pd.to_numeric(self.application_train[col], errors='coerce')
            if col in self.application_test.columns:
                self.application_test[col] = pd.to_numeric(self.application_test[col], errors='coerce')

        # 8) подсчёт пропусков
        self.application_train['MISSING_VALS_TOTAL_APP'] = self.application_train.isna().sum(axis=1)
        self.application_test['MISSING_VALS_TOTAL_APP'] = self.application_test.isna().sum(axis=1)

        if self.verbose:
            print("Очистка завершена.")

    def simple_imputation(self):
        """Простая импутация пропусков медианой/модой."""
        if self.verbose:
            print("\nПростая импутация пропусков...")

        # Импутация EXT_SOURCE медианой
        for ext_col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
            if ext_col in self.application_train.columns:
                median_val = self.application_train[ext_col].median()
                self.application_train[ext_col] = self.application_train[ext_col].fillna(median_val)
                if ext_col in self.application_test.columns:
                    self.application_test[ext_col] = self.application_test[ext_col].fillna(median_val)

        # Импутация числовых колонок медианой
        numeric_cols = self.application_train.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'TARGET' and col != 'SK_ID_CURR':
                median_val = self.application_train[col].median()
                self.application_train[col] = self.application_train[col].fillna(median_val)
                if col in self.application_test.columns:
                    self.application_test[col] = self.application_test[col].fillna(median_val)

        if self.verbose:
            print("Импутация завершена.")

    def numeric_feature_engineering(self, data):
        """Создание числовых признаков."""
        df = data.copy()
        eps = 1e-8

        # Приведение базовых колонок к числам
        for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Финансовые соотношения
        df['CREDIT_INCOME_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('AMT_INCOME_TOTAL', 0) + eps)
        df['CREDIT_ANNUITY_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('AMT_ANNUITY', 0) + eps)
        df['ANNUITY_INCOME_RATIO'] = df.get('AMT_ANNUITY', 0) / (df.get('AMT_INCOME_TOTAL', 0) + eps)
        df['INCOME_ANNUITY_DIFF'] = df.get('AMT_INCOME_TOTAL', 0) - df.get('AMT_ANNUITY', 0)
        df['CREDIT_GOODS_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('AMT_GOODS_PRICE', 0) + eps)
        df['CREDIT_GOODS_DIFF'] = df.get('AMT_CREDIT', 0) - df.get('AMT_GOODS_PRICE', 0)
        df['GOODS_INCOME_RATIO'] = df.get('AMT_GOODS_PRICE', 0) / (df.get('AMT_INCOME_TOTAL', 0) + eps)

        # Возраст и работа
        df['DAYS_BIRTH'] = pd.to_numeric(df.get('DAYS_BIRTH'), errors='coerce')
        df['DAYS_EMPLOYED'] = pd.to_numeric(df.get('DAYS_EMPLOYED'), errors='coerce')
        df['AGE_EMPLOYED_DIFF'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
        df['EMPLOYED_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + eps)

        if 'AMT_CREDIT' in df.columns and 'HOUR_APPR_PROCESS_START' in df.columns:
            df['HOUR_PROCESS_CREDIT_MUL'] = df['AMT_CREDIT'] * df['HOUR_APPR_PROCESS_START']
        else:
            df['HOUR_PROCESS_CREDIT_MUL'] = 0

        # Автомобиль
        df['OWN_CAR_AGE'] = pd.to_numeric(df.get('OWN_CAR_AGE'), errors='coerce')
        df['CAR_EMPLOYED_DIFF'] = df['OWN_CAR_AGE'] - df.get('DAYS_EMPLOYED', 0)
        df['CAR_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / (df.get('DAYS_EMPLOYED', 0) + eps)

        # Контакты
        contact_cols = [c for c in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL'] if c in df.columns]
        df['FLAG_CONTACTS_SUM'] = df[contact_cols].sum(axis=1) if contact_cols else 0

        # Семья и доход
        df['CNT_FAM_MEMBERS'] = pd.to_numeric(df.get('CNT_FAM_MEMBERS', 0), errors='coerce').fillna(0)
        df['CNT_CHILDREN'] = pd.to_numeric(df.get('CNT_CHILDREN', 0), errors='coerce').fillna(0)
        df['AMT_INCOME_TOTAL'] = pd.to_numeric(df.get('AMT_INCOME_TOTAL', 0), errors='coerce').fillna(0)
        df['CNT_NON_CHILDREN'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
        df['PER_CAPITA_INCOME'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

        # Рейтинг региона
        df['REGION_RATING_MAX'] = df[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].max(axis=1)
        df['REGION_RATING_MIN'] = df[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].min(axis=1)
        df['REGION_RATING_MEAN'] = (df.get('REGION_RATING_CLIENT', 0) + df.get('REGION_RATING_CLIENT_W_CITY', 0)) / 2

        # EXT_SOURCE агрегаты
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
        for c in ext_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if ext_cols:
            df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
            df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
            df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)

        # Апартаменты
        avg_columns = [col for col in df.columns if col.endswith('_AVG')]
        mode_columns = [col for col in df.columns if col.endswith('_MODE')]
        medi_columns = [col for col in df.columns if col.endswith('_MEDI')]

        def safe_sum(cols):
            if not cols:
                return 0
            return df[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)

        df['APARTMENTS_SUM_AVG'] = safe_sum(avg_columns)
        df['APARTMENTS_SUM_MODE'] = safe_sum(mode_columns)
        df['APARTMENTS_SUM_MEDI'] = safe_sum(medi_columns)

        # OBS/DEF фичи
        for c in ['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df['OBS_30_60_SUM'] = df.get('OBS_30_CNT_SOCIAL_CIRCLE', 0) + df.get('OBS_60_CNT_SOCIAL_CIRCLE', 0)
        df['DEF_30_60_SUM'] = df.get('DEF_30_CNT_SOCIAL_CIRCLE', 0) + df.get('DEF_60_CNT_SOCIAL_CIRCLE', 0)

        # Документы
        flag_doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
        df['SUM_FLAGS_DOCUMENTS'] = df[flag_doc_cols].sum(axis=1) if flag_doc_cols else 0

        return df

    def categorical_interaction_features(self, train_data, test_data):
        """Создание признаков через группировки категориальных колонок."""
        groups = [
            ['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'],
            ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE']
        ]

        aggregations = {
            'AMT_ANNUITY': ['mean', 'max'],
            'AMT_INCOME_TOTAL': ['mean', 'max'],
            'EXT_SOURCE_MEAN': ['mean', 'max']
        }

        for group_cols in groups:
            present = [c for c in group_cols if c in train_data.columns]
            if not present:
                continue

            available_aggs = {k: v for k, v in aggregations.items() if k in train_data.columns}
            if not available_aggs:
                continue

            grouped = train_data.groupby(present).agg(available_aggs)
            grouped.columns = [f"{col}_{agg}_AGG_{'_'.join(present)}" for col, agg in grouped.columns]
            grouped = grouped.reset_index()

            try:
                with open(f'Application_train_grouped_interactions_{"_".join(present)}.pkl', 'wb') as f:
                    pickle.dump(grouped, f)
            except Exception:
                pass

            train_data = train_data.merge(grouped, on=present, how='left')
            test_data = test_data.merge(grouped, on=present, how='left')

        return train_data, test_data

    def response_fit(self, data, column):
        """Response encoding."""
        mapping = {0: {}, 1: {}}
        total_counts = data[column].value_counts()
        for label in [0, 1]:
            class_counts = data.loc[data['TARGET'] == label, column].value_counts()
            mapping[label] = (class_counts / total_counts).fillna(0).to_dict()
        return mapping

    def response_transform(self, data, column, dict_mapping):
        """Применение response encoding."""
        data[f'{column}_0'] = data[column].map(dict_mapping.get(0, {}))
        data[f'{column}_1'] = data[column].map(dict_mapping.get(1, {}))

    def main(self):
        """Полный pipeline предобработки."""
        self.load_dataframes()
        self.data_cleaning()
        self.simple_imputation()

        if self.verbose:
            t0 = datetime.now()
            print("\nНачинаем генерацию признаков...")

        # СОХРАНИ TARGET ОТДЕЛЬНО ДЛЯ RESPONSE ENCODING
        self.application_train_target = self.application_train['TARGET'].copy()

        # УДАЛИ SK_ID_CURR И TARGET СРАЗУ
        self.application_train = self.application_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        self.application_test = self.application_test.drop(['SK_ID_CURR'], axis=1, errors='ignore')

        # NUMERIC FEATURE ENGINEERING
        self.application_train = self.numeric_feature_engineering(self.application_train)
        self.application_test = self.numeric_feature_engineering(self.application_test)

        if self.verbose:
            print(f"Numeric features done. Time: {datetime.now() - t0}")

        # CATEGORICAL INTERACTIONS
        if self.verbose:
            t1 = datetime.now()
            print("Создание признаков категориальных взаимодействий...")
        self.application_train, self.application_test = self.categorical_interaction_features(self.application_train,
                                                                                              self.application_test)
        if self.verbose:
            print(f"Categorical interactions done. Time: {datetime.now() - t1}")

        # RESPONSE ENCODING
        categorical_columns = list(set(self.application_train.select_dtypes(include=['object']).columns) |
                                   set(self.application_test.select_dtypes(include=['object']).columns))

        # Временный датафрейм с TARGET только для обучения response encoding
        temp_train_with_target = pd.concat([self.application_train, self.application_train_target], axis=1)

        for col in categorical_columns:
            mapping = self.response_fit(temp_train_with_target, col)
            try:
                with open(f'Response_coding_dict_{col}.pkl', 'wb') as f:
                    pickle.dump(mapping, f)
            except Exception:
                pass
            self.response_transform(self.application_train, col, mapping)
            self.response_transform(self.application_test, col, mapping)
            if col in self.application_train.columns:
                self.application_train = self.application_train.drop(columns=[col])
            if col in self.application_test.columns:
                self.application_test = self.application_test.drop(columns=[col])

        pkl_dir = "C://Users//User//credit-scoring//src//features//"

        if self.dump_to_pickle:
            try:
                with open(os.path.join(pkl_dir, "application_train_preprocessed.pkl"), 'wb') as f:
                    pickle.dump(self.application_train, f)
                with open(os.path.join(pkl_dir, "application_test_preprocessed.pkl"), 'wb') as f:
                    pickle.dump(self.application_test, f)
                with open(os.path.join(pkl_dir, "application_train_target.pkl"), 'wb') as f:
                    pickle.dump(self.application_train_target, f)
            except Exception as e:
                if self.verbose:
                    print(f"Ошибка при сохранении pickle: {e}")

        if self.verbose:
            print("\nПредобработка завершена.")
            print(f"Начальный размер train: {self.initial_shape}")
            print(f"Финальный размер train: {self.application_train.shape}")
            print(f"Финальный размер test: {self.application_test.shape}")

        return self.application_train, self.application_test
if __name__ == '__main__':
    file_path = 'C://Users//User//credit-scoring//data//raw//'
    preprocessor = ApplicationDataPreprocessor(file_directory=file_path)
    train, test = preprocessor.main()
    print(f"Тип train: {type(train)}")
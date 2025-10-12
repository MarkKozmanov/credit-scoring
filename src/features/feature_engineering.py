import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor


class ApplicationDataPreprocessor:
    """
    Препроцессор для datasets application_train и application_test.
    Выполняет очистку данных, проектирование признаков и предобработку
    специально для основных файлов с заявками.
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        """
        Инициализация препроцессора.

        Аргументы:
            file_directory: Путь к файлам данных (включая завершающий '/')
            verbose: Флаг вывода сообщений о прогрессе
            dump_to_pickle: Флаг сохранения обработанных данных в pickle файлы
        """
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory = file_directory

    def load_dataframes(self):
        """Загрузка DataFrame application_train и application_test."""
        if self.verbose:
            self.start_time = datetime.now()
            print('#' * 60)
            print('#      Предобработка данных заявок      #')
            print('#' * 60)
            print("\nЗагрузка application_train.csv и application_test.csv...")

        self.application_train = pd.read_csv(self.file_directory + 'application_train.csv')
        self.application_test = pd.read_csv(self.file_directory + 'application_test.csv')
        self.initial_shape = self.application_train.shape

        if self.verbose:
            print("DataFrame успешно загружены")
            print(f"Время загрузки: {datetime.now() - self.start_time}")

    def data_cleaning(self):
        """Выполнение операций очистки данных."""
        if self.verbose:
            print("\nВыполнение очистки данных...")

        self._remove_redundant_flag_columns()
        self._convert_days_to_years()
        self._handle_anomalous_values()
        self._handle_categorical_missing_values()
        self._convert_region_ratings_to_categorical()
        self._add_missing_values_count()

        if self.verbose:
            print("Очистка данных завершена")

    def _remove_redundant_flag_columns(self):
        """Удаление FLAG_DOCUMENT столбцов с низкой дисперсией."""
        flag_cols_to_drop = [
            'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_20'
        ]
        self.application_train = self.application_train.drop(flag_cols_to_drop, axis=1)
        self.application_test = self.application_test.drop(flag_cols_to_drop, axis=1)

    def _convert_days_to_years(self):
        """Конвертация дней в годы для возрастных признаков."""
        self.application_train['DAYS_BIRTH'] = self.application_train['DAYS_BIRTH'] * -1 / 365
        self.application_test['DAYS_BIRTH'] = self.application_test['DAYS_BIRTH'] * -1 / 365

    def _handle_anomalous_values(self):
        """Обработка аномальных значений в dataset."""
        # Обработка аномального значения в DAYS_EMPLOYED
        anomalous_value = 365243
        self.application_train['DAYS_EMPLOYED'] = self.application_train['DAYS_EMPLOYED'].replace(anomalous_value,
                                                                                                  np.nan)
        self.application_test['DAYS_EMPLOYED'] = self.application_test['DAYS_EMPLOYED'].replace(anomalous_value, np.nan)

        # Обработка выбросов в социальных кругах
        obs_columns = ['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']
        for col in obs_columns:
            self.application_train.loc[self.application_train[col] > 30, col] = np.nan
            self.application_test.loc[self.application_test[col] > 30, col] = np.nan

        # Удаление строк с невалидным полом
        self.application_train = self.application_train[self.application_train['CODE_GENDER'] != 'XNA']

    def _handle_categorical_missing_values(self):
        """Заполнение пропущенных значений в категориальных столбцах."""
        categorical_columns = self.application_train.select_dtypes(include='object').columns.tolist()
        self.application_train[categorical_columns] = self.application_train[categorical_columns].fillna('XNA')
        self.application_test[categorical_columns] = self.application_test[categorical_columns].fillna('XNA')

    def _convert_region_ratings_to_categorical(self):
        """Конвертация столбцов рейтингов регионов в категориальный тип."""
        region_columns = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']
        for col in region_columns:
            self.application_train[col] = self.application_train[col].astype('object')
            self.application_test[col] = self.application_test[col].astype('object')

    def _add_missing_values_count(self):
        """Добавление подсчета пропущенных значений для каждой строки."""
        self.application_train['MISSING_VALS_TOTAL_APP'] = self.application_train.isna().sum(axis=1)
        self.application_test['MISSING_VALS_TOTAL_APP'] = self.application_test.isna().sum(axis=1)

    def impute_ext_source_features(self):
        """Импутация пропущенных значений в EXT_SOURCE признаках с помощью XGBoost."""
        if self.verbose:
            start_time = datetime.now()
            print("\nИмпутация пропущенных значений EXT_SOURCE...")

        # Выбор числовых столбцов для моделирования (исключая целевую и ID столбцы)
        numeric_columns = self.application_test.select_dtypes(include=[np.number]).columns.tolist()
        columns_for_modelling = [
            col for col in numeric_columns
            if col not in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'SK_ID_CURR', 'TARGET']
        ]

        # Сохранение столбцов для будущего использования
        with open('columns_for_ext_values_predictor.pkl', 'wb') as f:
            pickle.dump(columns_for_modelling, f)

        # Импутация EXT_SOURCE столбцов в порядке от наименьших пропусков к наибольшим
        for ext_col in ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1']:
            self._impute_single_ext_source(ext_col, columns_for_modelling)
            columns_for_modelling.append(ext_col)  # Добавление импутированного столбца для следующего предсказания

        if self.verbose:
            print(f"Импутация EXT_SOURCE завершена за {datetime.now() - start_time}")

    def _impute_single_ext_source(self, column_name, feature_columns):
        """Импутация пропущенных значений для одного EXT_SOURCE столбца."""
        # Подготовка данных для моделирования
        train_mask = self.application_train[column_name].notna()
        X_train = self.application_train.loc[train_mask, feature_columns]
        y_train = self.application_train.loc[train_mask, column_name]

        X_train_missing = self.application_train.loc[~train_mask, feature_columns]
        X_test_missing = self.application_test.loc[self.application_test[column_name].isna(), feature_columns]

        # Обучение XGBoost модели
        xgb_model = XGBRegressor(
            n_estimators=1000,
            max_depth=3,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=59
        )
        xgb_model.fit(X_train, y_train)

        # Сохранение модели
        with open(f'nan_{column_name}_xgbr_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

        # Выполнение предсказаний
        if not X_train_missing.empty:
            self.application_train.loc[~train_mask, column_name] = xgb_model.predict(X_train_missing)

        if not X_test_missing.empty:
            self.application_test.loc[self.application_test[column_name].isna(), column_name] = xgb_model.predict(
                X_test_missing)

    def create_numeric_features(self, data):
        """
        Создание спроектированных числовых признаков на основе предметных знаний.

        Аргументы:
            data: DataFrame для проектирования признаков

        Возвращает:
            DataFrame с добавленными признаками
        """
        data = data.copy()

        # Финансовые соотношения и разницы
        data = self._create_financial_features(data)

        # Возрастные и рабочие признаки
        data = self._create_demographic_features(data)

        # Признаки владения автомобилем
        data = self._create_car_features(data)

        # Флаги контактов
        data = self._create_contact_features(data)

        # Семейные признаки
        data = self._create_family_features(data)

        # Признаки рейтинга регионов
        data = self._create_region_features(data)

        # EXT_SOURCE признаки
        data = self._create_ext_source_features(data)

        # Признаки квартир
        data = self._create_apartment_features(data)

        # Признаки социальных кругов
        data = self._create_social_features(data)

        # Флаги документов
        data = self._create_document_features(data)

        # Признаки дней
        data = self._create_days_features(data)

        # Признаки запросов
        data = self._create_enquiry_features(data)

        return data

    def _create_financial_features(self, data):
        """Создание финансовых соотношений и разниц."""
        epsilon = 0.00001  # для избежания деления на ноль

        data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / (data['AMT_INCOME_TOTAL'] + epsilon)
        data['CREDIT_ANNUITY_RATIO'] = data['AMT_CREDIT'] / (data['AMT_ANNUITY'] + epsilon)
        data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / (data['AMT_INCOME_TOTAL'] + epsilon)
        data['INCOME_ANNUITY_DIFF'] = data['AMT_INCOME_TOTAL'] - data['AMT_ANNUITY']
        data['CREDIT_GOODS_RATIO'] = data['AMT_CREDIT'] / (data['AMT_GOODS_PRICE'] + epsilon)
        data['CREDIT_GOODS_DIFF'] = data['AMT_CREDIT'] - data['AMT_GOODS_PRICE']
        data['GOODS_INCOME_RATIO'] = data['AMT_GOODS_PRICE'] / (data['AMT_INCOME_TOTAL'] + epsilon)
        data['INCOME_EXT_RATIO'] = data['AMT_INCOME_TOTAL'] / (data['EXT_SOURCE_3'] + epsilon)
        data['CREDIT_EXT_RATIO'] = data['AMT_CREDIT'] / (data['EXT_SOURCE_3'] + epsilon)

        return data

    def _create_demographic_features(self, data):
        """Создание возрастных и рабочих признаков."""
        epsilon = 0.00001

        data['AGE_EMPLOYED_DIFF'] = data['DAYS_BIRTH'] - data['DAYS_EMPLOYED']
        data['EMPLOYED_TO_AGE_RATIO'] = data['DAYS_EMPLOYED'] / (data['DAYS_BIRTH'] + epsilon)
        data['HOUR_PROCESS_CREDIT_MUL'] = data['AMT_CREDIT'] * data['HOUR_APPR_PROCESS_START']

        return data

    def _create_car_features(self, data):
        """Создание признаков владения автомобилем."""
        epsilon = 0.00001

        data['CAR_EMPLOYED_DIFF'] = data['OWN_CAR_AGE'] - data['DAYS_EMPLOYED']
        data['CAR_EMPLOYED_RATIO'] = data['OWN_CAR_AGE'] / (data['DAYS_EMPLOYED'] + epsilon)
        data['CAR_AGE_DIFF'] = data['DAYS_BIRTH'] - data['OWN_CAR_AGE']
        data['CAR_AGE_RATIO'] = data['OWN_CAR_AGE'] / (data['DAYS_BIRTH'] + epsilon)

        return data

    def _create_contact_features(self, data):
        """Создание признаков флагов контактов."""
        contact_columns = [
            'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
            'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL'
        ]
        data['FLAG_CONTACTS_SUM'] = data[contact_columns].sum(axis=1)
        return data

    def _create_family_features(self, data):
        """Создание семейных признаков."""
        epsilon = 0.00001

        data['CNT_NON_CHILDREN'] = data['CNT_FAM_MEMBERS'] - data['CNT_CHILDREN']
        data['CHILDREN_INCOME_RATIO'] = data['CNT_CHILDREN'] / (data['AMT_INCOME_TOTAL'] + epsilon)
        data['PER_CAPITA_INCOME'] = data['AMT_INCOME_TOTAL'] / (data['CNT_FAM_MEMBERS'] + 1)

        return data

    def _create_region_features(self, data):
        """Создание признаков рейтинга регионов."""
        data['REGIONS_RATING_INCOME_MUL'] = (data['REGION_RATING_CLIENT'] + data['REGION_RATING_CLIENT_W_CITY']) * data[
            'AMT_INCOME_TOTAL'] / 2
        data['REGION_RATING_MAX'] = data[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].max(axis=1)
        data['REGION_RATING_MIN'] = data[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].min(axis=1)
        data['REGION_RATING_MEAN'] = (data['REGION_RATING_CLIENT'] + data['REGION_RATING_CLIENT_W_CITY']) / 2
        data['REGION_RATING_MUL'] = data['REGION_RATING_CLIENT'] * data['REGION_RATING_CLIENT_W_CITY']

        # Флаги регионов
        region_flag_columns = [
            'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
            'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY'
        ]
        data['FLAG_REGIONS'] = data[region_flag_columns].sum(axis=1)

        return data

    def _create_ext_source_features(self, data):
        """Создание агрегированных EXT_SOURCE признаков."""
        data['EXT_SOURCE_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        data['EXT_SOURCE_MUL'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
        data['EXT_SOURCE_MAX'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
        data['EXT_SOURCE_MIN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
        data['EXT_SOURCE_VAR'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].var(axis=1)
        data['WEIGHTED_EXT_SOURCE'] = data['EXT_SOURCE_1'] * 2 + data['EXT_SOURCE_2'] * 3 + data['EXT_SOURCE_3'] * 4

        return data

    def _create_apartment_features(self, data):
        """Создание признаков квартир."""
        # AVG столбцы
        avg_columns = [col for col in data.columns if col.endswith('_AVG')]
        data['APARTMENTS_SUM_AVG'] = data[avg_columns].sum(axis=1)

        # MODE столбцы
        mode_columns = [col for col in data.columns if col.endswith('_MODE')]
        data['APARTMENTS_SUM_MODE'] = data[mode_columns].sum(axis=1)

        # MEDI столбцы
        medi_columns = [col for col in data.columns if col.endswith('_MEDI')]
        data['APARTMENTS_SUM_MEDI'] = data[medi_columns].sum(axis=1)

        # Взаимодействия дохода и квартир
        data['INCOME_APARTMENT_AVG_MUL'] = data['APARTMENTS_SUM_AVG'] * data['AMT_INCOME_TOTAL']
        data['INCOME_APARTMENT_MODE_MUL'] = data['APARTMENTS_SUM_MODE'] * data['AMT_INCOME_TOTAL']
        data['INCOME_APARTMENT_MEDI_MUL'] = data['APARTMENTS_SUM_MEDI'] * data['AMT_INCOME_TOTAL']

        return data

    def _create_social_features(self, data):
        """Создание признаков социальных кругов."""
        epsilon = 0.00001

        data['OBS_30_60_SUM'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] + data['OBS_60_CNT_SOCIAL_CIRCLE']
        data['DEF_30_60_SUM'] = data['DEF_30_CNT_SOCIAL_CIRCLE'] + data['DEF_60_CNT_SOCIAL_CIRCLE']
        data['OBS_DEF_30_MUL'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] * data['DEF_30_CNT_SOCIAL_CIRCLE']
        data['OBS_DEF_60_MUL'] = data['OBS_60_CNT_SOCIAL_CIRCLE'] * data['DEF_60_CNT_SOCIAL_CIRCLE']
        data['SUM_OBS_DEF_ALL'] = (data['OBS_30_CNT_SOCIAL_CIRCLE'] + data['DEF_30_CNT_SOCIAL_CIRCLE'] +
                                   data['OBS_60_CNT_SOCIAL_CIRCLE'] + data['DEF_60_CNT_SOCIAL_CIRCLE'])

        # Соотношения с кредитом
        data['OBS_30_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['OBS_30_CNT_SOCIAL_CIRCLE'] + epsilon)
        data['OBS_60_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['OBS_60_CNT_SOCIAL_CIRCLE'] + epsilon)
        data['DEF_30_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['DEF_30_CNT_SOCIAL_CIRCLE'] + epsilon)
        data['DEF_60_CREDIT_RATIO'] = data['AMT_CREDIT'] / (data['DEF_60_CNT_SOCIAL_CIRCLE'] + epsilon)

        return data

    def _create_document_features(self, data):
        """Создание признаков флагов документов."""
        document_columns = [col for col in data.columns if col.startswith('FLAG_DOCUMENT_')]
        data['SUM_FLAGS_DOCUMENTS'] = data[document_columns].sum(axis=1)
        return data

    def _create_days_features(self, data):
        """Создание признаков дней."""
        days_columns = ['DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']
        data['DAYS_DETAILS_CHANGE_MUL'] = data[days_columns].prod(axis=1)
        data['DAYS_DETAILS_CHANGE_SUM'] = data[days_columns].sum(axis=1)
        return data

    def _create_enquiry_features(self, data):
        """Создание признаков кредитных запросов."""
        epsilon = 0.00001
        enquiry_columns = [col for col in data.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
        data['AMT_ENQ_SUM'] = data[enquiry_columns].sum(axis=1)
        data['ENQ_CREDIT_RATIO'] = data['AMT_ENQ_SUM'] / (data['AMT_CREDIT'] + epsilon)
        return data

    def create_knn_target_features(self):
        """Создание KNN-признаков на основе соседей по целевой переменной."""
        if self.verbose:
            print("\nСоздание KNN-признаков на основе целевой переменной...")

        # Подготовка признаков для KNN
        knn_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_ANNUITY_RATIO']

        train_knn_data = self.application_train[knn_features].fillna(0)
        test_knn_data = self.application_test[knn_features].fillna(0)
        train_target = self.application_train['TARGET']

        # Обучение KNN модели
        knn = KNeighborsClassifier(n_neighbors=500, n_jobs=-1)
        knn.fit(train_knn_data, train_target)

        # Сохранение модели и тренировочных данных
        with open('KNN_model_TARGET_500_neighbors.pkl', 'wb') as f:
            pickle.dump(knn, f)
        with open('TARGET_MEAN_500_Neighbors_training_data.pkl', 'wb') as f:
            pickle.dump(train_knn_data, f)

        # Получение соседей и расчет среднего таргета
        train_neighbors = knn.kneighbors(train_knn_data, return_distance=False)
        test_neighbors = knn.kneighbors(test_knn_data, return_distance=False)

        self.application_train['TARGET_NEIGHBORS_500_MEAN'] = [
            self.application_train['TARGET'].iloc[indices].mean()
            for indices in train_neighbors
        ]
        self.application_test['TARGET_NEIGHBORS_500_MEAN'] = [
            self.application_train['TARGET'].iloc[indices].mean()
            for indices in test_neighbors
        ]

        if self.verbose:
            print("KNN-признаки созданы")

    def create_categorical_interactions(self, train_data, test_data):
        """
        Создание признаков категориальных взаимодействий.

        Аргументы:
            train_data: Тренировочный DataFrame
            test_data: Тестовый DataFrame

        Возвращает:
            Кортеж (train_data, test_data) с добавленными признаками
        """
        if self.verbose:
            start_time = datetime.now()
            print("Создание признаков категориальных взаимодействий...")

        # Определение стратегий группировки
        grouping_strategies = [
            ['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'],
            ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE'],
            ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
            ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        ]

        # Определение агрегаций
        aggregations = {
            'AMT_ANNUITY': ['mean', 'max', 'min'],
            'ANNUITY_INCOME_RATIO': ['mean', 'max', 'min'],
            'AGE_EMPLOYED_DIFF': ['mean', 'min'],
            'AMT_INCOME_TOTAL': ['mean', 'max', 'min'],
            'APARTMENTS_SUM_AVG': ['mean', 'max', 'min'],
            'APARTMENTS_SUM_MEDI': ['mean', 'max', 'min'],
            'EXT_SOURCE_MEAN': ['mean', 'max', 'min'],
            'EXT_SOURCE_1': ['mean', 'max', 'min'],
            'EXT_SOURCE_2': ['mean', 'max', 'min'],
            'EXT_SOURCE_3': ['mean', 'max', 'min']
        }

        # Создание признаков взаимодействий для каждой стратегии группировки
        for group_cols in grouping_strategies:
            train_data, test_data = self._create_single_interaction(
                train_data, test_data, group_cols, aggregations
            )

        if self.verbose:
            print(f"Признаки категориальных взаимодействий созданы за {datetime.now() - start_time}")

        return train_data, test_data

    def _create_single_interaction(self, train_data, test_data, group_cols, aggregations):
        """Создание признаков взаимодействий для одной стратегии группировки."""
        group_name = '_'.join(group_cols)

        # Расчет агрегаций на тренировочных данных
        grouped = train_data.groupby(group_cols).agg(aggregations)
        grouped.columns = [f'{metric}_{col}_AGG_{group_name}'.upper()
                           for col, metric in grouped.columns]

        # Сохранение результатов группировки
        with open(f'Application_train_grouped_interactions_{group_name}.pkl', 'wb') as f:
            pickle.dump(grouped, f)

        # Объединение с обоими datasets
        train_data = train_data.merge(grouped, on=group_cols, how='left')
        test_data = test_data.merge(grouped, on=group_cols, how='left')

        return train_data, test_data

    def encode_categorical_features(self):
        """Кодирование категориальных признаков с использованием response encoding."""
        if self.verbose:
            print("\nКодирование категориальных признаков...")

        categorical_columns = self.application_train.select_dtypes(include='object').columns.tolist()

        for col in categorical_columns:
            # Создание mapping для response encoding
            encoding_map = self._create_response_encoding(self.application_train, col)

            # Сохранение mapping
            with open(f'Response_coding_dict_{col}.pkl', 'wb') as f:
                pickle.dump(encoding_map, f)

            # Применение encoding к обоим datasets
            self._apply_response_encoding(self.application_train, col, encoding_map)
            self._apply_response_encoding(self.application_test, col, encoding_map)

            # Удаление оригинального категориального столбца
            self.application_train.drop(col, axis=1, inplace=True)
            self.application_test.drop(col, axis=1, inplace=True)

        if self.verbose:
            print("Кодирование категориальных признаков завершено")

    def _create_response_encoding(self, data, column):
        """
        Создание mapping для response encoding категориального столбца.

        Аргументы:
            data: DataFrame с данными
            column: Категориальный столбец для кодирования

        Возвращает:
            Словарь с encoding для каждого класса
        """
        encoding_map = {0: {}, 1: {}}

        for target_class in [0, 1]:
            class_data = data[data['TARGET'] == target_class]
            class_counts = class_data[column].value_counts()
            total_counts = data[column].value_counts()

            # Расчет вероятности для каждой категории
            for category in total_counts.index:
                class_count = class_counts.get(category, 0)
                encoding_map[target_class][category] = class_count / total_counts[category]

        return encoding_map

    def _apply_response_encoding(self, data, column, encoding_map):
        """
        Применение response encoding к столбцу.

        Аргументы:
            data: DataFrame для кодирования
            column: Имя столбца для кодирования
            encoding_map: Словарь mapping для encoding
        """
        data[f'{column}_0'] = data[column].map(encoding_map[0])
        data[f'{column}_1'] = data[column].map(encoding_map[1])

    def main(self):
        """
        Выполнение полного пайплайна предобработки.

        Возвращает:
            Кортеж (обработанные_train_данные, обработанные_test_данные)
        """
        # Загрузка данных
        self.load_dataframes()

        # Очистка данных
        self.data_cleaning()

        # Импутация EXT_SOURCE признаков
        self.impute_ext_source_features()

        # Проектирование признаков
        if self.verbose:
            print("\nНачало проектирования признаков...")

        # Проектирование числовых признаков
        self.application_train = self.create_numeric_features(self.application_train)
        self.application_test = self.create_numeric_features(self.application_test)

        # KNN-признаки на основе целевой переменной
        self.create_knn_target_features()

        # Категориальные взаимодействия
        self.application_train, self.application_test = self.create_categorical_interactions(
            self.application_train, self.application_test
        )

        # Кодирование категориальных признаков
        self.encode_categorical_features()

        # Финальная сводка
        if self.verbose:
            print('\nПредобработка завершена!')
            print(f"Начальный размер train: {self.initial_shape}")
            print(f"Финальный размер train: {self.application_train.shape}")
            print(f"Общее время: {datetime.now() - self.start_time}")

        # Сохранение результатов при необходимости
        if self.dump_to_pickle:
            self._save_processed_data()

        return self.application_train, self.application_test

    def _save_processed_data(self):
        """Сохранение обработанных данных в pickle файлы."""
        if self.verbose:
            print('\nСохранение обработанных данных в pickle файлы...')

        with open(self.file_directory + 'application_train_preprocessed.pkl', 'wb') as f:
            pickle.dump(self.application_train, f)
        with open(self.file_directory + 'application_test_preprocessed.pkl', 'wb') as f:
            pickle.dump(self.application_test, f)

        if self.verbose:
            print('Данные успешно сохранены!')
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ApplicationDataPreprocessor:
    """
    Класс предобработки таблиц application_train.csv и application_test.csv.
    Оставлена только логика, которая затрагивает исключительно эти два файла.

    Методы:
        - __init__
        - load_dataframes
        - data_cleaning
        - ext_source_values_predictor
        - numeric_feature_engineering
        - neighbors_EXT_SOURCE_feature
        - categorical_interaction_features
        - response_fit
        - response_transform
        - main
    """

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=True):
        """
        Инициализация.

        Параметры:
            file_directory (str): путь к csv файлам (заканчивается '/'), по умолчанию ''
            verbose (bool): печатать лог или нет
            dump_to_pickle (bool): сохранять ли в pickle результат
        """
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
        """
        Очистка данных: удаление бесполезных флагов, обработка DAYS_*, исправление выбросов,
        заполнение категорий и подсчёт числа пропусков.
        Работает только с application_train и application_test.
        """
        if self.verbose:
            print("\nВыполняется очистка данных...")

        # 1) удалить FLAG_DOCUMENT с низкой дисперсией (игнорировать, если отсутствуют)
        flag_cols_to_drop = [
            'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_20'
        ]
        self.application_train = self.application_train.drop(columns=flag_cols_to_drop, errors='ignore')
        self.application_test = self.application_test.drop(columns=flag_cols_to_drop, errors='ignore')

        # 2) DAYS_BIRTH -> годы (положительные)
        if 'DAYS_BIRTH' in self.application_train.columns:
            self.application_train['DAYS_BIRTH'] = pd.to_numeric(self.application_train['DAYS_BIRTH'], errors='coerce')
            self.application_train['DAYS_BIRTH'] = self.application_train['DAYS_BIRTH'].apply(lambda x: (-x) / 365 if pd.notna(x) else np.nan)
        if 'DAYS_BIRTH' in self.application_test.columns:
            self.application_test['DAYS_BIRTH'] = pd.to_numeric(self.application_test['DAYS_BIRTH'], errors='coerce')
            self.application_test['DAYS_BIRTH'] = self.application_test['DAYS_BIRTH'].apply(lambda x: (-x) / 365 if pd.notna(x) else np.nan)

        # 3) DAYS_EMPLOYED: заменить известную метку аномалии 365243 на NaN
        for df in (self.application_train, self.application_test):
            if 'DAYS_EMPLOYED' in df.columns:
                df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
                df['DAYS_EMPLOYED'] = pd.to_numeric(df['DAYS_EMPLOYED'], errors='coerce')
                df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: (-x) / 365 if pd.notna(x) else np.nan)

        # 4) OBS_*: значения > 30 считаем некорректными -> NaN (используем .loc для корректного присваивания)
        for col in ['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']:
            if col in self.application_train.columns:
                self.application_train.loc[self.application_train[col] > 30, col] = np.nan
            if col in self.application_test.columns:
                self.application_test.loc[self.application_test[col] > 30, col] = np.nan

        # 5) удалить строки с CODE_GENDER == 'XNA' в train
        if 'CODE_GENDER' in self.application_train.columns:
            self.application_train = self.application_train[self.application_train['CODE_GENDER'] != 'XNA']

        # 6) заполнить категориальные пропуски значением 'XNA' (согласованно для train/test)
        train_cats = set(self.application_train.select_dtypes(include=['object', 'category']).columns)
        test_cats = set(self.application_test.select_dtypes(include=['object', 'category']).columns)
        categorical_columns = sorted(list(train_cats.union(test_cats)))
        for col in categorical_columns:
            if col not in self.application_train.columns:
                self.application_train[col] = 'XNA'
            if col not in self.application_test.columns:
                self.application_test[col] = 'XNA'
            self.application_train[col] = self.application_train[col].fillna('XNA').astype(str)
            self.application_test[col] = self.application_test[col].fillna('XNA').astype(str)

        # 7) REGION_RATING_*: привести к числу (если собираемся с ними считать)
        for col in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']:
            if col in self.application_train.columns:
                self.application_train[col] = pd.to_numeric(self.application_train[col], errors='coerce')
            if col in self.application_test.columns:
                self.application_test[col] = pd.to_numeric(self.application_test[col], errors='coerce')

        # 8) подсчёт количества пропусков в строке
        self.application_train['MISSING_VALS_TOTAL_APP'] = self.application_train.isna().sum(axis=1)
        self.application_test['MISSING_VALS_TOTAL_APP'] = self.application_test.isna().sum(axis=1)

        if self.verbose:
            print("Очистка завершена.")

    def ext_source_values_predictor(self):
        """
        Импутация пропусков в EXT_SOURCE_1/2/3 с помощью XGBRegressor.
        Использует только числовые колонки, присутствующие и в train, и в test.
        """
        if self.verbose:
            t0 = datetime.now()
            print("\nИмпутация пропущенных EXT_SOURCE значений...")

        # получить пересечение числовых колонок train/test
        numeric_train = set(self.application_train.select_dtypes(include=[np.number]).columns.tolist())
        numeric_test = set(self.application_test.select_dtypes(include=[np.number]).columns.tolist())
        feature_candidates = sorted(list((numeric_train & numeric_test) - set(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'SK_ID_CURR', 'TARGET'])))

        # сохранить список фичей
        try:
            with open('columns_for_ext_values_predictor.pkl', 'wb') as f:
                pickle.dump(feature_candidates, f)
        except Exception:
            pass

        for ext_col in ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1']:
            # если столбца нет вообще — пропускаем
            if ext_col not in self.application_train.columns and ext_col not in self.application_test.columns:
                continue

            # отфильтровать features, которых действительно нет в df
            features = [f for f in feature_candidates if f in self.application_train.columns and f in self.application_test.columns]
            if not features:
                if self.verbose:
                    print(f"  Нет общих числовых фичей для моделирования {ext_col}, пропускаем.")
                continue

            # формируем тренировочные данные (в train те строки, где ext_col не пуст)
            train_mask = self.application_train[ext_col].notna()
            X_full = self.application_train.loc[train_mask, features].copy()
            y_full = pd.to_numeric(self.application_train.loc[train_mask, ext_col], errors='coerce').copy()

            # если мало данных — пропустить
            if X_full.shape[0] < 50:
                if self.verbose:
                    print(f"  Слишком мало данных для обучения модели {ext_col} ({X_full.shape[0]} строк).")
                continue

            # простая валидация
            X_tr, X_val, y_tr, y_val = train_test_split(X_full.fillna(X_full.median()), y_full, test_size=0.1, random_state=42)

            model = XGBRegressor(n_estimators=1000, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=59, verbosity=0)
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            except Exception:
                # fallback
                model.n_estimators = 200
                model.fit(X_tr, y_tr)

            # сохранить модель
            try:
                with open(f'nan_{ext_col}_xgbr_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
            except Exception:
                pass

            # предсказать и заполнить в train и test (с защитой)
            # train missing
            train_missing_idx = self.application_train[self.application_train[ext_col].isna()].index
            if len(train_missing_idx) > 0:
                X_train_missing = self.application_train.loc[train_missing_idx, features].fillna(X_full.median())
                try:
                    self.application_train.loc[train_missing_idx, ext_col] = model.predict(X_train_missing)
                except Exception:
                    pass

            # test missing
            if ext_col in self.application_test.columns:
                test_missing_idx = self.application_test[self.application_test[ext_col].isna()].index
                if len(test_missing_idx) > 0:
                    X_test_missing = self.application_test.loc[test_missing_idx, features].fillna(X_full.median())
                    try:
                        self.application_test.loc[test_missing_idx, ext_col] = model.predict(X_test_missing)
                    except Exception:
                        pass

            # добавить предсказанный столбец в кандидаты для следующих итераций
            if ext_col not in feature_candidates:
                feature_candidates.append(ext_col)

        if self.verbose:
            print(f"Импутация EXT_SOURCE завершена за {datetime.now() - t0}")

    def numeric_feature_engineering(self, data):
        """
        Создание числовых признаков (работает с DataFrame, возвращает новый DataFrame).
        Внутри все преобразования защищены: pd.to_numeric + fillna/epsilon.
        """
        df = data.copy()
        eps = 1e-8

        # приведение базовых колонок к числам (если есть)
        for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # финансовые соотношения
        df['CREDIT_INCOME_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('AMT_INCOME_TOTAL', 0) + eps)
        df['CREDIT_ANNUITY_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('AMT_ANNUITY', 0) + eps)
        df['ANNUITY_INCOME_RATIO'] = df.get('AMT_ANNUITY', 0) / (df.get('AMT_INCOME_TOTAL', 0) + eps)
        df['INCOME_ANNUITY_DIFF'] = df.get('AMT_INCOME_TOTAL', 0) - df.get('AMT_ANNUITY', 0)
        df['CREDIT_GOODS_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('AMT_GOODS_PRICE', 0) + eps)
        df['CREDIT_GOODS_DIFF'] = df.get('AMT_CREDIT', 0) - df.get('AMT_GOODS_PRICE', 0)
        df['GOODS_INCOME_RATIO'] = df.get('AMT_GOODS_PRICE', 0) / (df.get('AMT_INCOME_TOTAL', 0) + eps)
        df['INCOME_EXT_RATIO'] = df.get('AMT_INCOME_TOTAL', 0) / (df.get('EXT_SOURCE_3', 0) + eps)
        df['CREDIT_EXT_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('EXT_SOURCE_3', 0) + eps)

        # возраст и работа (предполагаем, что DAYS_BIRTH и DAYS_EMPLOYED уже в годах после data_cleaning)
        df['DAYS_BIRTH'] = pd.to_numeric(df.get('DAYS_BIRTH'), errors='coerce')
        df['DAYS_EMPLOYED'] = pd.to_numeric(df.get('DAYS_EMPLOYED'), errors='coerce')
        df['AGE_EMPLOYED_DIFF'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
        df['EMPLOYED_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + eps)

        if 'AMT_CREDIT' in df.columns and 'HOUR_APPR_PROCESS_START' in df.columns:
            df['HOUR_PROCESS_CREDIT_MUL'] = df['AMT_CREDIT'] * df['HOUR_APPR_PROCESS_START']
        else:
            df['HOUR_PROCESS_CREDIT_MUL'] = 0

        # автомобиль
        df['OWN_CAR_AGE'] = pd.to_numeric(df.get('OWN_CAR_AGE'), errors='coerce')
        df['CAR_EMPLOYED_DIFF'] = df['OWN_CAR_AGE'] - df.get('DAYS_EMPLOYED', 0)
        df['CAR_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / (df.get('DAYS_EMPLOYED', 0) + eps)
        df['CAR_AGE_DIFF'] = df.get('DAYS_BIRTH', 0) - df['OWN_CAR_AGE']
        df['CAR_AGE_RATIO'] = df['OWN_CAR_AGE'] / (df.get('DAYS_BIRTH', 0) + eps)

        # контакты
        contact_cols = [c for c in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL'] if c in df.columns]
        df['FLAG_CONTACTS_SUM'] = df[contact_cols].sum(axis=1) if contact_cols else 0

        # семья и доход на душу
        df['CNT_FAM_MEMBERS'] = pd.to_numeric(df.get('CNT_FAM_MEMBERS', 0), errors='coerce').fillna(0)
        df['CNT_CHILDREN'] = pd.to_numeric(df.get('CNT_CHILDREN', 0), errors='coerce').fillna(0)
        df['AMT_INCOME_TOTAL'] = pd.to_numeric(df.get('AMT_INCOME_TOTAL', 0), errors='coerce').fillna(0)
        df['CNT_NON_CHILDREN'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
        df['CHILDREN_INCOME_RATIO'] = df['CNT_CHILDREN'] / (df['AMT_INCOME_TOTAL'] + eps)
        df['PER_CAPITA_INCOME'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

        # рейтинг региона (числовые уже в data_cleaning)
        df['REGIONS_RATING_INCOME_MUL'] = ((df.get('REGION_RATING_CLIENT', 0) + df.get('REGION_RATING_CLIENT_W_CITY', 0)) * df.get('AMT_INCOME_TOTAL', 0) / 2)
        df['REGION_RATING_MAX'] = df[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].max(axis=1)
        df['REGION_RATING_MIN'] = df[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].min(axis=1)
        df['REGION_RATING_MEAN'] = (df.get('REGION_RATING_CLIENT', 0) + df.get('REGION_RATING_CLIENT_W_CITY', 0)) / 2
        df['REGION_RATING_MUL'] = df.get('REGION_RATING_CLIENT', 0) * df.get('REGION_RATING_CLIENT_W_CITY', 0)

        # флаги регионов
        region_flag_columns = [c for c in [
            'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
            'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY'
        ] if c in df.columns]
        df['FLAG_REGIONS'] = df[region_flag_columns].sum(axis=1) if region_flag_columns else 0

        # EXT_SOURCE агрегаты — безопасно
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
        for c in ext_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if ext_cols:
            df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
            if all(c in df.columns for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
                df['EXT_SOURCE_MUL'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
                df['EXT_SOURCE_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
                df['EXT_SOURCE_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
                df['EXT_SOURCE_VAR'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].var(axis=1)
                df['WEIGHTED_EXT_SOURCE'] = df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 4
        else:
            df['EXT_SOURCE_MEAN'] = np.nan

        # ---- АПАРТАМЕНТЫ: безопасное суммирование колонок *_AVG, *_MODE, *_MEDI ----
        avg_columns = [col for col in df.columns if col.endswith('_AVG')]
        mode_columns = [col for col in df.columns if col.endswith('_MODE')]
        medi_columns = [col for col in df.columns if col.endswith('_MEDI')]

        def conv_and_filter(frame, cols):
            """Преобразовать cols в числовые (coerce), отбросить полностью NaN столбцы."""
            if not cols:
                return pd.DataFrame(index=frame.index)
            conv = frame[cols].apply(pd.to_numeric, errors='coerce')
            conv = conv.dropna(axis=1, how='all')
            return conv

        avg_df = conv_and_filter(df, avg_columns)
        mode_df = conv_and_filter(df, mode_columns)
        medi_df = conv_and_filter(df, medi_columns)

        if self.verbose:
            print(f"Числовые AVG: {list(avg_df.columns)}")
            print(f"Числовые MODE: {list(mode_df.columns)}")
            print(f"Числовые MEDI: {list(medi_df.columns)}")

        df['APARTMENTS_SUM_AVG'] = avg_df.sum(axis=1) if not avg_df.empty else 0
        df['APARTMENTS_SUM_MODE'] = mode_df.sum(axis=1) if not mode_df.empty else 0
        df['APARTMENTS_SUM_MEDI'] = medi_df.sum(axis=1) if not medi_df.empty else 0

        income = pd.to_numeric(df.get('AMT_INCOME_TOTAL', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        df['INCOME_APARTMENT_AVG_MUL'] = df['APARTMENTS_SUM_AVG'] * income
        df['INCOME_APARTMENT_MODE_MUL'] = df['APARTMENTS_SUM_MODE'] * income
        df['INCOME_APARTMENT_MEDI_MUL'] = df['APARTMENTS_SUM_MEDI'] * income

        # OBS/DEF фичи
        for c in ['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df['OBS_30_60_SUM'] = df.get('OBS_30_CNT_SOCIAL_CIRCLE', 0) + df.get('OBS_60_CNT_SOCIAL_CIRCLE', 0)
        df['DEF_30_60_SUM'] = df.get('DEF_30_CNT_SOCIAL_CIRCLE', 0) + df.get('DEF_60_CNT_SOCIAL_CIRCLE', 0)
        df['OBS_DEF_30_MUL'] = df.get('OBS_30_CNT_SOCIAL_CIRCLE', 0) * df.get('DEF_30_CNT_SOCIAL_CIRCLE', 0)
        df['OBS_DEF_60_MUL'] = df.get('OBS_60_CNT_SOCIAL_CIRCLE', 0) * df.get('DEF_60_CNT_SOCIAL_CIRCLE', 0)
        df['SUM_OBS_DEF_ALL'] = (df.get('OBS_30_CNT_SOCIAL_CIRCLE', 0) + df.get('DEF_30_CNT_SOCIAL_CIRCLE', 0) +
                                 df.get('OBS_60_CNT_SOCIAL_CIRCLE', 0) + df.get('DEF_60_CNT_SOCIAL_CIRCLE', 0))
        df['OBS_30_CREDIT_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('OBS_30_CNT_SOCIAL_CIRCLE', 0) + eps)
        df['OBS_60_CREDIT_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('OBS_60_CNT_SOCIAL_CIRCLE', 0) + eps)
        df['DEF_30_CREDIT_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('DEF_30_CNT_SOCIAL_CIRCLE', 0) + eps)
        df['DEF_60_CREDIT_RATIO'] = df.get('AMT_CREDIT', 0) / (df.get('DEF_60_CNT_SOCIAL_CIRCLE', 0) + eps)

        # документы
        flag_doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
        df['SUM_FLAGS_DOCUMENTS'] = df[flag_doc_cols].sum(axis=1) if flag_doc_cols else 0

        # дни подробностей
        days_cols = [c for c in ['DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'] if c in df.columns]
        if days_cols:
            df['DAYS_DETAILS_CHANGE_MUL'] = df[days_cols].prod(axis=1)
            df['DAYS_DETAILS_CHANGE_SUM'] = df[days_cols].sum(axis=1)
        else:
            df['DAYS_DETAILS_CHANGE_MUL'] = 0
            df['DAYS_DETAILS_CHANGE_SUM'] = 0

        # запросы в бюро
        enquiry_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
        if enquiry_cols:
            df['AMT_ENQ_SUM'] = df[enquiry_cols].sum(axis=1)
            df['ENQ_CREDIT_RATIO'] = df['AMT_ENQ_SUM'] / (df.get('AMT_CREDIT', 0) + eps)
        else:
            df['AMT_ENQ_SUM'] = 0
            df['ENQ_CREDIT_RATIO'] = 0

        return df

    def neighbors_EXT_SOURCE_feature(self, n_neighbors=50):
        """
        Создаёт признак — средний TARGET по k ближайшим соседям (по EXT_SOURCE_*, CREDIT_ANNUITY_RATIO).
        Защищено от утечки: при вычислении для train исключаем саму точку.
        """
        if self.verbose:
            print("\nСоздание KNN-признака на основе EXT_SOURCE и CREDIT_ANNUITY_RATIO...")

        base_feats = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_ANNUITY_RATIO']
        feats = [f for f in base_feats if f in self.application_train.columns and f in self.application_test.columns]
        if not feats:
            if self.verbose:
                print("  Нет необходимых фичей для KNN. Пропускаем.")
            self.application_train['TARGET_NEIGHBORS_MEAN'] = np.nan
            self.application_test['TARGET_NEIGHBORS_MEAN'] = np.nan
            return

        train_X = self.application_train[feats].copy().fillna(self.application_train[feats].median())
        test_X = self.application_test[feats].copy().fillna(self.application_train[feats].median())

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_X)
        test_scaled = scaler.transform(test_X)

        n_train = train_scaled.shape[0]
        k = min(n_neighbors, max(1, n_train - 1))

        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(train_scaled, self.application_train['TARGET'].values)

        # TRAIN: получить k+1 соседей и удалить саму точку
        neighs_train = knn.kneighbors(train_scaled, n_neighbors=min(k + 1, n_train), return_distance=False)
        train_means = []
        for i, neigh in enumerate(neighs_train):
            neigh_list = list(neigh)
            if i in neigh_list:
                neigh_list.remove(i)
            else:
                # если сам отсутствует (маловероятно), обрезаем до k
                neigh_list = neigh_list[:k]
            if len(neigh_list) == 0:
                train_means.append(self.application_train['TARGET'].mean())
            else:
                train_means.append(self.application_train['TARGET'].iloc[neigh_list].mean())
        self.application_train['TARGET_NEIGHBORS_MEAN'] = train_means

        # TEST: взять k соседей из train
        neighs_test = knn.kneighbors(test_scaled, n_neighbors=k, return_distance=False)
        test_means = [self.application_train['TARGET'].iloc[inds].mean() for inds in neighs_test]
        self.application_test['TARGET_NEIGHBORS_MEAN'] = test_means

        try:
            with open('KNN_model_TARGET_neighbors.pkl', 'wb') as f:
                pickle.dump(knn, f)
        except Exception:
            pass

        if self.verbose:
            print("KNN-признак создан.")

    def categorical_interaction_features(self, train_data, test_data):
        """
        Создание признаков через группировки категориальных колонок (агрегации numeric -> group).
        Обрабатываются только группы, чьи колонки присутствуют в train.
        """
        groups = [
            ['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'],
            ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE'],
            ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
            ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        ]

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

        for group_cols in groups:
            present = [c for c in group_cols if c in train_data.columns]
            if not present:
                continue

            # фильтруем агрегации по наличию колонок
            available_aggs = {k: v for k, v in aggregations.items() if k in train_data.columns}
            if not available_aggs:
                continue

            grouped = train_data.groupby(present).agg(available_aggs)
            # нормализуем имена колонок
            grouped.columns = [f"{col.upper()}_{agg.upper()}_AGG_{'_'.join(present).upper()}" for col, agg in grouped.columns]
            grouped = grouped.reset_index()

            # сохраняем и мержим (join по ключам)
            try:
                with open(f'Application_train_grouped_interactions_{"_".join(present)}.pkl', 'wb') as f:
                    pickle.dump(grouped, f)
            except Exception:
                pass

            train_data = train_data.merge(grouped, on=present, how='left')
            test_data = test_data.merge(grouped, on=present, how='left')

        return train_data, test_data

    def response_fit(self, data, column):
        """
        Response encoding: посчитать частоту категории в классах 0/1.
        Возвращает словарь {0: {...}, 1: {...}} (в виде долей).
        """
        mapping = {0: {}, 1: {}}
        total_counts = data[column].value_counts()
        for label in [0, 1]:
            class_counts = data.loc[data['TARGET'] == label, column].value_counts()
            # безопасно делим: если категории отсутствуют в class_counts, get вернёт 0
            mapping[label] = (class_counts / total_counts).fillna(0).to_dict()
        return mapping

    def response_transform(self, data, column, dict_mapping):
        """
        Применяет mapping, создаёт два столбца: {column}_0 и {column}_1.
        Нераспознанные категории получают NaN -> можно заполнить позже.
        """
        data[f'{column}_0'] = data[column].map(dict_mapping.get(0, {}))
        data[f'{column}_1'] = data[column].map(dict_mapping.get(1, {}))

    def main(self):
        """Полный pipeline предобработки для application_train и application_test."""
        self.load_dataframes()
        self.data_cleaning()
        self.ext_source_values_predictor()

        if self.verbose:
            t0 = datetime.now()
            print("\nНачинаем генерацию признаков (numeric)...")

        self.application_train = self.numeric_feature_engineering(self.application_train)
        self.application_test = self.numeric_feature_engineering(self.application_test)

        # KNN-признак (neighbors)
        self.neighbors_EXT_SOURCE_feature(n_neighbors=50)

        if self.verbose:
            print(f"Numeric features done. Time: {datetime.now() - t0}")

        # categorical interactions
        if self.verbose:
            t1 = datetime.now()
            print("Создание признаков категориальных взаимодействий...")
        self.application_train, self.application_test = self.categorical_interaction_features(self.application_train, self.application_test)
        if self.verbose:
            print(f"Categorical interactions done. Time: {datetime.now() - t1}")

        # response encoding (на основе object колонок, объединение train/test)
        categorical_columns_application = sorted(list(set(self.application_train.select_dtypes(include=['object']).columns.tolist()) |
                                                    set(self.application_test.select_dtypes(include=['object']).columns.tolist())))
        for col in categorical_columns_application:
            mapping = self.response_fit(self.application_train, col)
            try:
                with open(f'Response_coding_dict_{col}.pkl', 'wb') as f:
                    pickle.dump(mapping, f)
            except Exception:
                pass
            self.response_transform(self.application_train, col, mapping)
            self.response_transform(self.application_test, col, mapping)
            # удалить исходный столбец
            if col in self.application_train.columns:
                self.application_train.pop(col)
            if col in self.application_test.columns:
                self.application_test.pop(col)

        if self.dump_to_pickle:
            try:
                with open("C://Users//User//credit-scoring//data//interim//application_train_preprocessed.pkl", 'wb') as f:
                    pickle.dump(self.application_train, f)
                with open("C://Users//User//credit-scoring//data//interim//application_test_preprocessed.pkl", 'wb') as f:
                    pickle.dump(self.application_test, f)
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
    preprocessor = ApplicationDataPreprocessor(file_directory = file_path)
    train, test = preprocessor.main()
    print(type(train))





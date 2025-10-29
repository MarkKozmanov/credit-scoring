import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

class BoostingEnsemble:
    """
    Комплексный класс для обучения и оценки ансамблевых моделей
    (XGBoost и LightGBM) с кросс-валидацией и детальным анализом.
    """

    def __init__(self, x_train, y_train, x_test, params,
                 num_folds=3, random_state=33, verbose=True,
                 save_model_to_pickle=False):
        """
        Инициализация BoostingEnsemble.
        """
        self.x_train = x_train
        self.y_train = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
        self.x_test = x_test
        self.params = params
        self.num_folds = num_folds
        self.verbose = verbose
        self.save_model = save_model_to_pickle
        self.models = []

        # Инициализация кросс-валидации
        self.stratified_cv = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=random_state
        )

        # Инициализация массивов предсказаний
        self._initialize_prediction_arrays()

    def _initialize_prediction_arrays(self):
        """Инициализация массивов для хранения предсказаний."""
        n_train = self.x_train.shape[0]
        n_test = self.x_test.shape[0]

        # Убираем train_preds_proba, так как он некорректно заполняется
        self.cv_preds_proba = np.zeros(n_train)
        self.test_preds_proba = np.zeros(n_test)
        self.best_threshold = 0

        # Инициализация DataFrame для важности признаков
        self.feature_importance = pd.DataFrame({
            'features': self.x_train.columns,
            'gain': np.zeros(len(self.x_train.columns))
        })

    def train(self, booster_type, verbose_eval=100, early_stopping_rounds=200,
              pickle_suffix=''):
        """
        Обучение бустинговой модели с кросс-валидацией.
        """
        self.booster_type = booster_type.lower()
        self._validate_booster_type()

        if self.verbose:
            self._print_training_start()
            start_time = datetime.now()

        # Инициализация списка моделей
        self.models = []

        # Выполнение обучения с кросс-валидацией
        for fold_num, (train_idx, val_idx) in enumerate(
                self.stratified_cv.split(self.x_train, self.y_train), 1
        ):
            if self.verbose:
                print(f"\nФолд {fold_num}/{self.num_folds}")

            # Обучение и оценка фолда
            model = self._train_fold(
                train_idx, val_idx, fold_num,
                verbose_eval, early_stopping_rounds, pickle_suffix
            )
            self.models.append(model)

        # Пост-обработка результатов
        self._post_process_training()

        if self.verbose:
            self._print_training_completion(start_time)

        gc.collect()
        return self.models

    def predict(self, x_data=None):
        """Предсказание на новых данных."""
        if x_data is None:
            x_data = self.x_test

        if not hasattr(self, 'models') or len(self.models) == 0:
            raise ValueError("Сначала обучите модели")

        all_predictions = []
        for model in self.models:
            if self.booster_type == 'xgboost':
                # Для новых версий XGBoost используем best_iteration
                if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                    pred = model.predict_proba(x_data, iteration_range=(0, model.best_iteration))[:, 1]
                else:
                    pred = model.predict_proba(x_data)[:, 1]
            else:
                # Для LightGBM
                if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
                    pred = model.predict_proba(x_data, num_iteration=model.best_iteration_)[:, 1]
                else:
                    pred = model.predict_proba(x_data)[:, 1]

            all_predictions.append(pred)

        avg_preds = np.mean(all_predictions, axis=0)
        return self.proba_to_class(avg_preds, self.best_threshold)

    def _validate_booster_type(self):
        """Проверка корректности типа бустера."""
        valid_boosters = ['xgboost', 'lightgbm']
        if self.booster_type not in valid_boosters:
            raise ValueError(f"booster_type должен быть одним из {valid_boosters}")

    def _print_training_start(self):
        """Вывод сообщения о начале обучения."""
        print(f"Обучение {self.booster_type} с {self.num_folds}-fold кросс-валидацией")
        print("Использование Out-of-Fold предсказаний для валидации")

    def _train_fold(self, train_idx, val_idx, fold_num,
                    verbose_eval, early_stopping_rounds, pickle_suffix):
        """Обучение одного фолда."""
        # Подготовка данных для фолда
        x_tr, y_tr, x_val, y_val = self._prepare_fold_data(train_idx, val_idx)

        # Инициализация и обучение модели
        model = self._initialize_model()

        # Параметры для XGBoost
        if self.booster_type == 'xgboost':
            fit_params = {
                'eval_set': [(x_val, y_val)],
                'eval_metric': 'auc',
                'early_stopping_rounds': early_stopping_rounds,
                'verbose': verbose_eval if self.verbose else False
            }

        # Параметры для LightGBM (без early stopping в fit)
        else:
            fit_params = {
                'eval_set': [(x_val, y_val)],
                'eval_metric': 'auc'
            }

        model.fit(x_tr, y_tr, **fit_params)

        # Генерация предсказаний
        self._generate_predictions(model, train_idx, val_idx, fold_num)

        # Расчет важности признаков
        self._calculate_feature_importance(model, fold_num)

        # Сохранение модели при необходимости
        if self.save_model:
            self._save_model(model, fold_num, pickle_suffix)

        return model

    def _prepare_fold_data(self, train_idx, val_idx):
        """Подготовка данных для одного фолда."""
        x_tr = self.x_train.iloc[train_idx]
        y_tr = self.y_train[train_idx]
        x_val = self.x_train.iloc[val_idx]
        y_val = self.y_train[val_idx]
        return x_tr, y_tr, x_val, y_val

    def _initialize_model(self):
        """Инициализация соответствующей бустинговой модели."""
        if self.booster_type == 'xgboost':
            return XGBClassifier(**self.params, random_state=42)
        else:
            return LGBMClassifier(**self.params, random_state=42, verbose=-1)

    def _generate_predictions(self, model, train_idx, val_idx, fold_num):
        """Генерация предсказаний для текущего фолда."""
        if self.booster_type == 'xgboost':
            self._generate_xgboost_predictions(model, train_idx, val_idx)
        else:
            self._generate_lightgbm_predictions(model, train_idx, val_idx)

        # Обновление оптимального порога на ВАЛИДАЦИОННЫХ данных
        self._update_optimal_threshold(val_idx)

    def _generate_xgboost_predictions(self, model, train_idx, val_idx):
        """Генерация предсказаний для XGBoost модели."""
        # Для XGBoost используем best_iteration если доступен
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            predict_params = {'iteration_range': (0, model.best_iteration)}
        else:
            predict_params = {}

        # Предсказания на валидационной выборке
        val_preds = model.predict_proba(self.x_train.iloc[val_idx], **predict_params)[:, 1]
        self.cv_preds_proba[val_idx] = val_preds

        # Тестовые предсказания (усредненные по фолдам)
        test_preds = model.predict_proba(self.x_test, **predict_params)[:, 1]
        self.test_preds_proba += test_preds / self.num_folds

    def _generate_lightgbm_predictions(self, model, train_idx, val_idx):
        """Генерация предсказаний для LightGBM модели."""
        # Для LightGBM используем best_iteration_ если доступен
        predict_params = {}
        if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
            predict_params['num_iteration'] = model.best_iteration_

        # Предсказания на валидационной выборке
        val_preds = model.predict_proba(self.x_train.iloc[val_idx], **predict_params)[:, 1]
        self.cv_preds_proba[val_idx] = val_preds

        # Тестовые предсказания (усредненные по фолдам)
        test_preds = model.predict_proba(self.x_test, **predict_params)[:, 1]
        self.test_preds_proba += test_preds / self.num_folds

    def _calculate_feature_importance(self, model, fold_num):
        """Расчет важности признаков для текущего фолда."""
        try:
            if self.booster_type == 'xgboost':
                importance_data = model.get_booster().get_score(importance_type='gain')
                fold_importance = pd.DataFrame({
                    'features': list(importance_data.keys()),
                    'gain': list(importance_data.values())
                })
            else:
                gain_values = model.booster_.feature_importance(importance_type='gain')
                fold_importance = pd.DataFrame({
                    'features': self.x_train.columns,
                    'gain': gain_values
                })

            # Агрегация важности признаков
            self.feature_importance = pd.concat([self.feature_importance, fold_importance],
                                                ignore_index=True)
        except Exception as e:
            if self.verbose:
                print(f"Ошибка при расчете важности признаков для фолда {fold_num}: {e}")

    def _update_optimal_threshold(self, val_idx):
        """Обновление оптимального порога с использованием ВАЛИДАЦИОННЫХ данных."""
        y_val_fold = self.y_train[val_idx]
        preds_fold = self.cv_preds_proba[val_idx]

        if len(np.unique(y_val_fold)) > 1:  # Проверяем, что есть оба класса
            fold_threshold = self.tune_threshold(y_val_fold, preds_fold)
            self.best_threshold += fold_threshold / self.num_folds

    def _save_model(self, model, fold_num, pickle_suffix):
        """Сохранение модели в pickle файл."""
        filename = f"model_{self.booster_type}_fold{fold_num}_{pickle_suffix}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        if self.verbose:
            print(f"Модель сохранена: {filename}")

    def _post_process_training(self):
        """Пост-обработка результатов обучения."""
        # Обработка важности признаков
        if not self.feature_importance.empty:
            self.feature_importance = (self.feature_importance
                                       .groupby('features', as_index=False)
                                       .mean()
                                       .sort_values('gain', ascending=False))

        # Конвертация вероятностей в метки классов
        self._convert_probabilities_to_classes()

    def _convert_probabilities_to_classes(self):
        """Конвертация вероятностей в метки классов с использованием оптимального порога."""
        # Используем только OOF предсказания для обучающей выборки
        self.cv_preds_class = self.proba_to_class(
            self.cv_preds_proba, self.best_threshold
        )
        self.test_preds_class = self.proba_to_class(
            self.test_preds_proba, self.best_threshold
        )

    def _print_training_completion(self, start_time):
        """Вывод сообщения о завершении обучения."""
        time_elapsed = datetime.now() - start_time
        print(f"\nОбучение завершено за {time_elapsed}")
        print(f"Оптимальный порог: {self.best_threshold:.4f}")

    def proba_to_class(self, probabilities, threshold):
        """
        Конвертация вероятностей в метки классов с использованием порога.
        """
        return (probabilities >= threshold).astype(int)

    def tune_threshold(self, true_labels, predicted_probas):
        """
        Поиск оптимального порога с использованием J-statistic Юдена.
        """
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probas)
        j_statistic = tpr - fpr
        best_idx = np.argmax(j_statistic)
        return thresholds[best_idx]

    def results(self, show_roc_auc=True, show_precision_recall=True,
                show_confusion_matrix=True, show_distributions=False):
        """
        Отображение комплексных результатов обучения.
        """
        print("=" * 80)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ")
        print("=" * 80)

        self._print_optimal_threshold()
        self._print_metrics(show_roc_auc, show_precision_recall)

        if show_confusion_matrix:
            self._plot_confusion_matrix()

        if show_distributions:
            self._plot_class_distributions()

        print("=" * 80)
        gc.collect()

    def _print_optimal_threshold(self):
        """Вывод информации об оптимальном пороге."""
        print(f"\nОптимальный порог (J-statistic Юдена): {self.best_threshold:.4f}")

    def _print_metrics(self, show_roc_auc, show_precision_recall):
        """Вывод метрик производительности."""
        # Метрики на обучающей выборке (предсказания на всех данных)
        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        # Получаем предсказания на всей обучающей выборке
        train_preds_proba, train_preds_class = self._get_train_predictions()
        self._print_set_metrics(
            self.y_train, train_preds_proba, train_preds_class,
            show_roc_auc, show_precision_recall
        )

        # Метрики на OOF предсказаниях
        print("\nOUT-OF-FOLD ПРЕДСКАЗАНИЯ:")
        self._print_set_metrics(
            self.y_train, self.cv_preds_proba, self.cv_preds_class,
            show_roc_auc, show_precision_recall
        )

    def _get_train_predictions(self):
        """Получение предсказаний на всей обучающей выборке усреднением по всем моделям."""
        all_train_preds = []
        for model in self.models:
            if self.booster_type == 'xgboost':
                if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                    pred = model.predict_proba(self.x_train, iteration_range=(0, model.best_iteration))[:, 1]
                else:
                    pred = model.predict_proba(self.x_train)[:, 1]
            else:
                if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
                    pred = model.predict_proba(self.x_train, num_iteration=model.best_iteration_)[:, 1]
                else:
                    pred = model.predict_proba(self.x_train)[:, 1]
            all_train_preds.append(pred)

        train_preds_proba = np.mean(all_train_preds, axis=0)
        train_preds_class = self.proba_to_class(train_preds_proba, self.best_threshold)
        return train_preds_proba, train_preds_class

    def _print_set_metrics(self, y_true, y_proba, y_pred,
                           show_roc_auc, show_precision_recall):
        """Вывод метрик для конкретного набора данных."""
        if show_roc_auc:
            auc_score = roc_auc_score(y_true, y_proba)
            print(f"  ROC-AUC: {auc_score:.4f}")

        if show_precision_recall:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")

    def _plot_confusion_matrix(self):
        """Построение матрицы ошибок для предсказаний кросс-валидации."""
        cm = confusion_matrix(self.y_train, self.cv_preds_class)
        cm_df = pd.DataFrame(
            cm,
            columns=['Предсказанный 0', 'Предсказанный 1'],
            index=['Фактический 0', 'Фактический 1']
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                    linewidths=0.5, annot_kws={'size': 14})
        plt.title('Матрица ошибок - Кросс-Валидация', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

    def _plot_class_distributions(self):
        """Построение сравнения распределений классов."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Исходное распределение
        sns.countplot(x=self.y_train, ax=axes[0])
        axes[0].set_title('Исходное распределение')
        axes[0].set_xlabel('Класс')

        # OOF предсказания
        sns.countplot(x=self.cv_preds_class, ax=axes[1])
        axes[1].set_title('OOF предсказания')
        axes[1].set_xlabel('Класс')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, num_features=20, figsize=(12, 10)):
        """
        Построение графика важности признаков.
        """
        # Выбор топ-признаков
        top_features = self.feature_importance.head(num_features)

        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='gain', y='features', hue='features', palette='viridis', legend=False)
        plt.title(f'Топ {num_features} самых важных признаков', fontsize=16, pad=20)
        plt.xlabel('Важность (gain)', fontsize=12)
        plt.ylabel('Признаки', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("=" * 80)
        gc.collect()

    def get_predictions(self, dataset_type='test'):
        """
        Получение предсказаний для указанного набора данных.
        """
        predictions_map = {
            'cv': (self.cv_preds_proba, self.cv_preds_class),
            'test': (self.test_preds_proba, self.test_preds_class)
        }

        if dataset_type not in predictions_map:
            raise ValueError("dataset_type должен быть 'cv' или 'test'")

        return predictions_map[dataset_type]

    def get_feature_importance(self, top_n=None):
        """
        Получение данных о важности признаков.
        """
        if top_n is None:
            return self.feature_importance
        return self.feature_importance.head(top_n)

    def save_final_model(self, model_path: str = None):
        """Сохранение финальной модели для продакшена"""
        import os

        if model_path is None:
            # Создаем папку models если ее нет
            model_dir = "C:/Users/User/credit-scoring/models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "lightgbm_model.pkl")
        else:
            # Создаем папку если указан путь
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Сохраняем первую модель из ансамбля
        if len(self.models) > 0:
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[0], f)
            print(f"✅ Model saved to {model_path}")

            # Также сохраняем threshold
            model_info = {
                'threshold': self.best_threshold,
                'feature_names': list(self.x_train.columns)
            }
            info_path = model_path.replace('.pkl', '_info.pkl')
            with open(info_path, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"✅ Model info saved to {info_path}")


if __name__ == "__main__":
    # Загрузка предобработанных данных и выбранных признаков
    with open("C:/Users/User/credit-scoring/src/features/selected_features_rfe.pkl", "rb") as f:
        selected_features = pickle.load(f)

    with open("C:/Users/User/credit-scoring/src/features/application_train_target.pkl", "rb") as f:
        y_train = pickle.load(f)

    with open("C:/Users/User/credit-scoring/src/features/application_train_preprocessed.pkl", "rb") as f:
        X_train_full = pickle.load(f)

    # Загрузка валидационной выборки
    X_val = pd.read_csv("C://Users//User//credit-scoring//data//interim//final_val_x.csv")

    # Выбор только отобранных признаков для train
    X_train = X_train_full[selected_features]

    # Подготовка валидационной выборки - добавляем отсутствующие признаки
    missing_features = [f for f in selected_features if f not in X_val.columns]
    print(f"Отсутствует признаков в val: {len(missing_features)}")

    # Добавляем отсутствующие признаки с нулевыми значениями
    for feature in missing_features:
        X_val[feature] = 0

    # Убираем лишние признаки, которых нет в selected_features
    extra_features = [f for f in X_val.columns if f not in selected_features]
    if extra_features:
        print(f"Убираем лишние признаки из val: {len(extra_features)}")
        X_val = X_val[selected_features]

    # Проверка совпадения признаков
    print(f"Количество признаков в train: {X_train.shape[1]}")
    print(f"Количество признаков в val: {X_val.shape[1]}")
    print(f"Совпадают ли признаки: {list(X_train.columns) == list(X_val.columns)}")

    # Параметры модели
    params = {
        'num_leaves': 6,
        'max_depth': 3,
        'min_split_gain': 0.1,
        'min_child_weight': 4,
        'min_child_samples': 6,
        'subsample': 0.5,
        'colsample_bytree': 0.5644927835566099,
        'reg_alpha': 0.2709917192816739,
        'reg_lambda': 0.1
    }

    # Инициализация и обучение модели
    light_boosting = BoostingEnsemble(
        x_train=X_train,
        y_train=y_train,
        x_test=X_val,
        params=params,
        save_model_to_pickle=False
    )
    light_boosting.train(booster_type='lightgbm')

    # Вывод результатов
    light_boosting.results()

    # Визуализация важности признаков
    light_boosting.plot_feature_importance()
    light_boosting.save_final_model("C:/Users/User/credit-scoring/models/lightgbm_model.pkl")
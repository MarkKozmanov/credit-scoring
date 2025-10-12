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


class BoostingEnsemble:
    """
    Комплексный класс для обучения и оценки ансамблевых моделей
    (XGBoost и LightGBM) с кросс-валидацией и детальным анализом.

    Возможности:
    - Обучение с кросс-валидацией и out-of-fold предсказаниями
    - Автоматическая оптимизация порога с использованием J-statistic Юдена
    - Анализ важности признаков
    - Комплексная визуализация результатов
    - Поддержка сохранения моделей
    """

    def __init__(self, x_train, y_train, x_test, params,
                 num_folds=3, random_state=33, verbose=True,
                 save_model_to_pickle=False):
        """
        Инициализация BoostingEnsemble.

        Args:
            x_train: Обучающая выборка (DataFrame)
            y_train: Целевая переменная (Series)
            x_test: Тестовая выборка (DataFrame)
            params: Гиперпараметры для ансамблевой модели
            num_folds: Количество фолдов CV (по умолчанию: 3)
            random_state: Random state для воспроизводимости (по умолчанию: 33)
            verbose: Флаг вывода прогресса (по умолчанию: True)
            save_model_to_pickle: Флаг сохранения моделей (по умолчанию: False)
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.params = params
        self.num_folds = num_folds
        self.verbose = verbose
        self.save_model = save_model_to_pickle

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

        self.train_preds_proba = np.zeros(n_train)
        self.cv_preds_proba = np.zeros(n_train)
        self.test_preds_proba = np.zeros(n_test)
        self.best_threshold = 0

        # Инициализация DataFrame для важности признаков
        self.feature_importance = pd.DataFrame({
            'features': self.x_train.columns,
            'gain': np.zeros(len(self.x_train.columns))
        })

    def train(self, booster_type, verbose_eval=400, early_stopping_rounds=200,
              pickle_suffix=''):
        """
        Обучение бустинговой модели с кросс-валидацией.

        Args:
            booster_type: Тип модели ('xgboost' или 'lightgbm')
            verbose_eval: Частота вывода прогресса (по умолчанию: 400)
            early_stopping_rounds: Количество итераций для ранней остановки (по умолчанию: 200)
            pickle_suffix: Суффикс для сохраняемых файлов моделей (по умолчанию: '')
        """
        self.booster_type = booster_type.lower()
        self._validate_booster_type()

        if self.verbose:
            self._print_training_start()
            start_time = datetime.now()

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

        # Пост-обработка результатов
        self._post_process_training()

        if self.verbose:
            self._print_training_completion(start_time)

        gc.collect()

    def _validate_booster_type(self):
        """Проверка корректности типа бустера."""
        valid_boosters = ['xgboost', 'lightgbm']
        if self.booster_type not in valid_boosters:
            raise ValueError(f"booster_type должен быть одним из {valid_boosters}")

    def _print_training_start(self):
        """Вывод сообщения о начале обучения."""
        print(f"Обучение {self.booster_type} с {self.num_folds}-fold кросс-валидацией")
        print("Использование Out-of-Fold предсказаний для валидации")

    def _train_fold(self, train_idx, val_idx, fold_num, verbose_eval,
                    early_stopping_rounds, pickle_suffix):
        """Обучение одного фолда."""
        # Подготовка данных для фолда
        x_tr, y_tr, x_val, y_val = self._prepare_fold_data(train_idx, val_idx)

        # Инициализация и обучение модели
        model = self._initialize_model()
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_tr, y_tr), (x_val, y_val)],
            eval_metric='auc',
            verbose=verbose_eval,
            early_stopping_rounds=early_stopping_rounds
        )

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
        y_tr = self.y_train.iloc[train_idx]
        x_val = self.x_train.iloc[val_idx]
        y_val = self.y_train.iloc[val_idx]
        return x_tr, y_tr, x_val, y_val

    def _initialize_model(self):
        """Инициализация соответствующей бустинговой модели."""
        if self.booster_type == 'xgboost':
            return XGBClassifier(**self.params)
        else:
            return LGBMClassifier(**self.params)

    def _generate_predictions(self, model, train_idx, val_idx, fold_num):
        """Генерация предсказаний для текущего фолда."""
        if self.booster_type == 'xgboost':
            self._generate_xgboost_predictions(model, train_idx, val_idx)
        else:
            self._generate_lightgbm_predictions(model, train_idx, val_idx)

        # Обновление оптимального порога
        self._update_optimal_threshold(train_idx)

    def _generate_xgboost_predictions(self, model, train_idx, val_idx):
        """Генерация предсказаний для XGBoost модели."""
        ntree_limit = model.get_booster().best_ntree_limit

        # Предсказания на обучающей выборке (усредненные по фолдам)
        self.train_preds_proba[train_idx] += (
                model.predict_proba(self.x_train.iloc[train_idx], ntree_limit=ntree_limit)[:, 1]
                / (self.num_folds - 1)
        )

        # Валидационные предсказания
        self.cv_preds_proba[val_idx] = model.predict_proba(
            self.x_train.iloc[val_idx], ntree_limit=ntree_limit
        )[:, 1]

        # Тестовые предсказания (усредненные по фолдам)
        self.test_preds_proba += (
                model.predict_proba(self.x_test, ntree_limit=ntree_limit)[:, 1]
                / self.num_folds
        )

    def _generate_lightgbm_predictions(self, model, train_idx, val_idx):
        """Генерация предсказаний для LightGBM модели."""
        num_iteration = model.best_iteration_

        # Предсказания на обучающей выборке (усредненные по фолдам)
        self.train_preds_proba[train_idx] += (
                model.predict_proba(self.x_train.iloc[train_idx], num_iteration=num_iteration)[:, 1]
                / (self.num_folds - 1)
        )

        # Валидационные предсказания
        self.cv_preds_proba[val_idx] = model.predict_proba(
            self.x_train.iloc[val_idx], num_iteration=num_iteration
        )[:, 1]

        # Тестовые предсказания (усредненные по фолдам)
        self.test_preds_proba += (
                model.predict_proba(self.x_test, num_iteration=num_iteration)[:, 1]
                / self.num_folds
        )

    def _calculate_feature_importance(self, model, fold_num):
        """Расчет важности признаков для текущего фолда."""
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

    def _update_optimal_threshold(self, train_idx):
        """Обновление оптимального порога с использованием текущего фолда."""
        y_train_fold = self.y_train.iloc[train_idx]
        preds_fold = self.train_preds_proba[train_idx]

        fold_threshold = self.tune_threshold(y_train_fold, preds_fold)
        self.best_threshold += fold_threshold / self.num_folds

    def _save_model(self, model, fold_num, pickle_suffix):
        """Сохранение модели в pickle файл."""
        filename = f"model_{self.booster_type}_fold{fold_num}_{pickle_suffix}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    def _post_process_training(self):
        """Пост-обработка результатов обучения."""
        # Обработка важности признаков
        self.feature_importance = (self.feature_importance
                                   .groupby('features', as_index=False)
                                   .mean()
                                   .sort_values('gain', ascending=False))

        # Конвертация вероятностей в метки классов
        self._convert_probabilities_to_classes()

    def _convert_probabilities_to_classes(self):
        """Конвертация вероятностей в метки классов с использованием оптимального порога."""
        self.train_preds_class = self.proba_to_class(
            self.train_preds_proba, self.best_threshold
        )
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

        Args:
            probabilities: Массив вероятностей
            threshold: Порог классификации

        Returns:
            Массив меток классов (0 или 1)
        """
        return (probabilities >= threshold).astype(int)

    def tune_threshold(self, true_labels, predicted_probas):
        """
        Поиск оптимального порога с использованием J-statistic Юдена.

        Аргументы:
            true_labels: Истинные метки классов
            predicted_probas: Предсказанные вероятности

        Возвращает:
            Оптимальное значение порога
        """
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probas)
        j_statistic = tpr - fpr
        best_idx = np.argmax(j_statistic)

        return thresholds[best_idx]

    def results(self, show_roc_auc=True, show_precision_recall=True,
                show_confusion_matrix=True, show_distributions=False):
        """
        Отображение комплексных результатов обучения.

        Args:
            show_roc_auc: Флаг отображения ROC-AUC scores
            show_precision_recall: Флаг отображения precision/recall scores
            show_confusion_matrix: Флаг отображения матрицы ошибок
            show_distributions: Флаг отображения распределений классов
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
        # Метрики на обучающей выборке
        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        self._print_set_metrics(
            self.y_train, self.train_preds_proba, self.train_preds_class,
            show_roc_auc, show_precision_recall
        )

        # Метрики кросс-валидации
        print("\nКРОСС-ВАЛИДАЦИЯ:")
        self._print_set_metrics(
            self.y_train, self.cv_preds_proba, self.cv_preds_class,
            show_roc_auc, show_precision_recall
        )

    def _print_set_metrics(self, y_true, y_proba, y_pred,
                           show_roc_auc, show_precision_recall):
        """Вывод метрик для конкретного набора данных."""
        if show_roc_auc:
            auc_score = roc_auc_score(y_true, y_proba)
            print(f"  ROC-AUC: {auc_score:.4f}")

        if show_precision_recall:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        plots_info = [
            (self.y_train, 'Исходные данные'),
            (self.cv_preds_class, 'Предсказания CV'),
            (self.test_preds_class, 'Тестовые предсказания')
        ]

        for idx, (data, title) in enumerate(plots_info):
            sns.countplot(x=data, ax=axes[idx])
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Класс')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, num_features=20, figsize=(12, 10)):
        """
        Построение графика важности признаков.

        Args:
            num_features: Количество топ-признаков для отображения
            figsize: Размер графика
        """
        # Выбор топ-признаков
        top_features = self.feature_importance.head(num_features)

        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='gain', y='features', palette='viridis')
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

        Args:
            dataset_type: 'train', 'cv', или 'test'

        Returns:
            Кортеж (вероятности, метки_классов)
        """
        predictions_map = {
            'train': (self.train_preds_proba, self.train_preds_class),
            'cv': (self.cv_preds_proba, self.cv_preds_class),
            'test': (self.test_preds_proba, self.test_preds_class)
        }

        if dataset_type not in predictions_map:
            raise ValueError("dataset_type должен быть 'train', 'cv', или 'test'")

        return predictions_map[dataset_type]

    def get_feature_importance(self, top_n=None):
        """
        Получение данных о важности признаков.

        Args:
            top_n: Количество топ-признаков для возврата (None для всех)

        Returns:
            DataFrame с важностью признаков
        """
        if top_n is None:
            return self.feature_importance
        return self.feature_importance.head(top_n)
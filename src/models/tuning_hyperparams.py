import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV





class ModelTuning:
    """
    Класс для настройки гиперпараметров, обучения моделей и оценки результатов.

    Этот класс выполняет randomized search CV для поиска оптимальных гиперпараметров,
    обучает модели на лучших параметрах и предоставляет результаты оценки.

    Атрибуты:
        base_model: Базовый классификатор/estimator
        x_train: Признаки обучающей выборки
        y_train: Целевые значения обучающей выборки
        x_test: Признаки тестовой выборки
        num_folds: Количество фолдов для кросс-валидации
        calibration: Флаг калибровки вероятностей
        calibration_method: Метод калибровки вероятностей
        calibration_cv: Количество фолдов для калибровки
    """

    def __init__(self, base_model, x_train, y_train, x_test,
                 calibration=False, calibration_method='isotonic',
                 calibration_cv=4, k_folds=4, random_state=982):
        """
        Инициализация класса с данными для обучения и конфигурацией.

        Args:
            base_model: Базовый классификатор/estimator
            x_train: Признаки обучающей выборки (numpy array)
            y_train: Целевые значения обучающей выборки (numpy array)
            x_test: Признаки тестовой выборки (numpy array)
            calibration: Флаг калибровки вероятностей (по умолчанию: False)
            calibration_method: 'isotonic' или 'sigmoid' (по умолчанию: 'isotonic')
            calibration_cv: Количество фолдов для калибровки (по умолчанию: 4)
            k_folds: Количество фолдов для обучения (по умолчанию: 4)
            random_state: Random state для воспроизводимости (по умолчанию: 982)
        """
        self.base_model = base_model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.num_folds = k_folds
        self.calibration = calibration

        # Инициализация кросс-валидации
        self.kfolds = StratifiedKFold(
            n_splits=k_folds,
            shuffle=True,
            random_state=random_state
        )

        # Настройки калибровки
        if self.calibration:
            self.calibration_method = calibration_method
            self.calibration_cv = calibration_cv

    def random_search_cv(self, hyperparams_dict, n_iter=30,
                         verbose=True, n_jobs=1, random_state=843):
        """
        Выполнение randomized search CV для настройки гиперпараметров.

        Args:
            hyperparams_dict: Словарь гиперпараметров для настройки
            n_iter: Количество случайных комбинаций параметров (по умолчанию: 30)
            verbose: Флаг вывода прогресса (по умолчанию: True)
            n_jobs: Количество параллельных jobs (по умолчанию: 1)
            random_state: Random state для воспроизводимости (по умолчанию: 843)

        Returns:
            None (устанавливает атрибуты best_model и tuning_results)
        """
        if verbose:
            print("\nВыполнение Randomized Search...")
            start = datetime.now()

        rscv = RandomizedSearchCV(
            estimator=self.base_model,
            param_distributions=hyperparams_dict,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=self.kfolds,
            return_train_score=True,
            verbose=2 if verbose else 0,
            n_jobs=n_jobs,
            random_state=random_state
        )
        rscv.fit(self.x_train, self.y_train)

        self.tuning_results = pd.DataFrame(rscv.cv_results_)
        self.best_model = rscv.best_estimator_

        if verbose:
            time_elapsed = datetime.now() - start
            print(f"Randomized Search завершен за {time_elapsed}")
            print(f"Лучший результат: {rscv.best_score_:.4f}")

        gc.collect()

    def train_on_best_params(self, verbose=True):
        """
        Обучение модели на лучших гиперпараметрах с кросс-валидацией.
        Генерирует out-of-fold предсказания и находит оптимальный порог.

        Args:
            verbose: Флаг вывода прогресса (по умолчанию: True)

        Returns:
            None (устанавливает cv_preds_probas, cv_preds_class, best_threshold_train)
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Сначала необходимо выполнить random_search_cv() для поиска лучшей модели")

        if verbose:
            print("Обучение классификатора на лучших параметрах...")
            print(f"{self.num_folds}-Fold Кросс-Валидация")
            start_time = datetime.now()

        # Инициализация массивов для предсказаний
        self.cv_preds_probas = np.zeros(self.x_train.shape[0])
        self.best_threshold_train = 0

        # Выполнение кросс-валидации
        for fold_num, (train_idx, val_idx) in enumerate(self.kfolds.split(self.x_train, self.y_train), 1):
            if verbose:
                print(f"  Обучение Фолда {fold_num}/{self.num_folds}")

            # Разделение данных для текущего фолда
            x_train_fold, x_val_fold = self.x_train[train_idx], self.x_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            # Обучение модели
            self.best_model.fit(x_train_fold, y_train_fold)

            # Получение предсказаний (с калибровкой или без)
            if not self.calibration:
                train_preds = self.best_model.predict_proba(x_train_fold)[:, 1]
                val_preds = self.best_model.predict_proba(x_val_fold)[:, 1]
            else:
                calibrated_clf = CalibratedClassifierCV(
                    self.best_model,
                    method=self.calibration_method,
                    cv=self.calibration_cv
                )
                calibrated_clf.fit(x_train_fold, y_train_fold)
                train_preds = calibrated_clf.predict_proba(x_train_fold)[:, 1]
                val_preds = calibrated_clf.predict_proba(x_val_fold)[:, 1]

            # Сохранение валидационных предсказаний
            self.cv_preds_probas[val_idx] = val_preds

            # Поиск оптимального порога для текущего фолда
            fold_threshold = self.tune_threshold(y_train_fold, train_preds)
            self.best_threshold_train += fold_threshold / self.num_folds

        # Конвертация вероятностей в классы
        self.cv_preds_class = self.proba_to_class(self.cv_preds_probas, self.best_threshold_train)

        if verbose:
            time_elapsed = datetime.now() - start_time
            print(f"Обучение завершено за {time_elapsed}")
            print(f"Оптимальный порог: {self.best_threshold_train:.4f}")

        gc.collect()

    def proba_to_class(self, probabilities, threshold):
        """
        Конвертация вероятностей в метки классов с использованием порога.

        Args:
            probabilities: Массив вероятностей для класса 1
            threshold: Пороговая вероятность для положительного класса

        Returns:
            Массив меток классов (0 или 1)
        """
        return (probabilities >= threshold).astype(int)

    def tune_threshold(self, true_labels, predicted_probas):
        """
        Поиск оптимального порога с использованием J-statistic (J = TPR - FPR).

        Ссылка: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

        Args:
            true_labels: Истинные метки классов
            predicted_probas: Предсказанные вероятности для положительного класса

        Returns:
            Оптимальная пороговая вероятность
        """
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probas)
        j_statistic = tpr - fpr
        best_idx = np.argmax(j_statistic)

        return thresholds[best_idx]

    def results_on_best_params(self, model_name):
        """
        Обучение на полных данных и вывод комплексных результатов.

        Args:
            model_name: Тип модели ('linear' или другой) для извлечения важности признаков

        Returns:
            None (выводит результаты и отображает графики)
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Сначала необходимо выполнить random_search_cv()")

        print("=" * 80)
        print("ОБУЧЕНИЕ НА ПОЛНЫХ ДАННЫХ И ОЦЕНКА РЕЗУЛЬТАТОВ")
        print("=" * 80)

        # Обучение на полных данных
        self.best_model.fit(self.x_train, self.y_train)

        # Получение предсказаний (с калибровкой или без)
        if not self.calibration:
            self.train_preds_probas = self.best_model.predict_proba(self.x_train)[:, 1]
            self.test_preds_probas = self.best_model.predict_proba(self.x_test)[:, 1]
        else:
            self.calibrated_classifier = CalibratedClassifierCV(
                self.best_model,
                method=self.calibration_method,
                cv=self.calibration_cv
            )
            self.calibrated_classifier.fit(self.x_train, self.y_train)
            self.train_preds_probas = self.calibrated_classifier.predict_proba(self.x_train)[:, 1]
            self.test_preds_probas = self.calibrated_classifier.predict_proba(self.x_test)[:, 1]

        # Конвертация в метки классов
        self.train_preds_class = self.proba_to_class(self.train_preds_probas, self.best_threshold_train)
        self.test_preds_class = self.proba_to_class(self.test_preds_probas, self.best_threshold_train)

        # Извлечение важности признаков
        if model_name.lower() == 'linear':
            self.feat_imp = self.best_model.coef_[0]
        else:
            self.feat_imp = self.best_model.feature_importances_

        # Отображение результатов
        self._print_metrics()
        self._plot_confusion_matrix()
        self._plot_class_distributions()

        gc.collect()

    def _print_metrics(self):
        """Вывод метрик производительности."""
        print(f"\nОптимальный порог (J-statistic): {self.best_threshold_train:.4f}\n")

        # Результаты на обучающей выборке
        train_auc = roc_auc_score(self.y_train, self.train_preds_probas)
        train_precision = precision_score(self.y_train, self.train_preds_class)
        train_recall = recall_score(self.y_train, self.train_preds_class)

        # Результаты кросс-валидации
        cv_auc = roc_auc_score(self.y_train, self.cv_preds_probas)
        cv_precision = precision_score(self.y_train, self.cv_preds_class)
        cv_recall = recall_score(self.y_train, self.cv_preds_class)

        print("РЕЗУЛЬТАТЫ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ:")
        print(f"  ROC-AUC: {train_auc:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")

        print("\nРЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ:")
        print(f"  ROC-AUC: {cv_auc:.4f}")
        print(f"  Precision: {cv_precision:.4f}")
        print(f"  Recall: {cv_recall:.4f}")
        print("=" * 80)

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
        """Построение распределений классов для оригинальных и предсказанных данных."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Оригинальное распределение
        sns.countplot(x=self.y_train, ax=axes[0])
        axes[0].set_title('Распределение классов в исходных данных')
        axes[0].set_xlabel('Класс')

        # Распределение предсказаний кросс-валидации
        sns.countplot(x=self.cv_preds_class, ax=axes[1])
        axes[1].set_title('Распределение классов в предсказаниях CV')
        axes[1].set_xlabel('Класс')

        # Распределение тестовых предсказаний
        sns.countplot(x=self.test_preds_class, ax=axes[2])
        axes[2].set_title('Распределение классов в тестовых предсказаниях')
        axes[2].set_xlabel('Класс')

        plt.tight_layout()
        plt.show()
        print("=" * 80)

    def feat_importances_show(self, feature_names, num_features=20, figsize=(12, 10)):
        """
        Отображение важности признаков.

        Args:
            feature_names: Массив названий признаков
            num_features: Количество топ-признаков для отображения (по умолчанию: 20)
            figsize: Размер графика (по умолчанию: (12, 10))
        """
        if not hasattr(self, 'feat_imp'):
            raise ValueError("Сначала необходимо выполнить results_on_best_params() для получения важности признаков")

        # Получение топ-признаков
        num_features = min(num_features, len(feature_names))
        top_indices = np.argsort(self.feat_imp)[::-1][:num_features]
        top_importances = self.feat_imp[top_indices]
        top_names = feature_names[top_indices]

        # Создание графика
        plt.figure(figsize=figsize)
        y_pos = np.arange(num_features)

        plt.barh(y_pos, top_importances)
        plt.yticks(y_pos, top_names)
        plt.xlabel('Важность признака')
        plt.title(f'Топ {num_features} самых важных признаков')
        plt.gca().invert_yaxis()  # Самые важные сверху
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("=" * 80)
        gc.collect()
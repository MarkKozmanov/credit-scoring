"""
Модули для обучения и настройки моделей кредитного скоринга.

Классы:
- ModelTuning: Настройка гиперпараметров и обучение моделей
- BoostingEnsemble: Обучение и оценка ансамблевых моделей (XGBoost, LightGBM)
"""

from .tuning_hyperparams import ModelTuning
from .train_boost_models import BoostingEnsemble

__all__ = [
    'ModelTuning',
    'BoostingEnsemble'
]
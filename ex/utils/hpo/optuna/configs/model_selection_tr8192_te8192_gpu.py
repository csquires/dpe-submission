"""StudyConfig for model_selection tr8192_te8192 on GPU lane."""
from ex.utils.hpo.optuna.configs._model_selection_base import make

CONFIG = make(tag="tr8192_te8192", tier="gpu")

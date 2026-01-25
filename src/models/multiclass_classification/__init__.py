from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier
from src.models.multiclass_classification.default_multiclass_classifier import make_default_multiclass_classifier


def make_multiclass_classifier(name: str, **kwargs) -> MulticlassClassifier:
    if name == "default":
        return make_default_multiclass_classifier(**kwargs)
    raise ValueError(f"Unknown multiclass classifier: {name}")
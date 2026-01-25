from src.models.binary_classification.binary_classifier import BinaryClassifier
from src.models.binary_classification.default_binary_classifier import make_default_binary_classifier
from src.models.binary_classification.gaussian_binary_classifier import make_gaussian_binary_classifier


def make_binary_classifier(name: str, **kwargs) -> BinaryClassifier:
    if name == "default":
        return make_default_binary_classifier(**kwargs)
    if name == "gaussian":
        return make_gaussian_binary_classifier(**kwargs)
    raise ValueError(f"Unknown binary classifier: {name}")


def make_pairwise_binary_classifiers(name: str, num_classes: int, **kwargs) -> list[BinaryClassifier]:
    return [make_binary_classifier(name, **kwargs) for _ in range(num_classes - 1)]
# model.py

from simpletransformers.ner import NERModel

def load_ner_model(
    model_dir: str = "outputs/",
    use_cuda: bool = True
) -> NERModel:
    """
    Load a pretrained NER model from a directory.

    Args:
        model_dir: Path to model directory (must be SimpleTransformers format)
        use_cuda: Whether to load onto GPU if available

    Returns:
        NERModel instance
    """
    return NERModel("bert", model_dir, use_cuda=use_cuda)


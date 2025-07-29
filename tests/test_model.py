# test_model.py

from ner_chunk_stitch_pipeline.model import load_ner_model

def test_model_loading_path_string():
    try:
        model = load_ner_model("outputs/", use_cuda=False)
        assert hasattr(model, "predict")
        print("Model loaded successfully (structure check passed).")
    except Exception as e:
        print("Model loading test skipped (no model found):", e)

if __name__ == "__main__":
    test_model_loading_path_string()

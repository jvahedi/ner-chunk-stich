# utils.py

def count_words(text: str) -> int:
    """
    Count the number of whitespace-separated words in a string.
    (Used instead of token count for simplicity and performance)

    Args:
        text: Input string

    Returns:
        Number of words in text
    """
    return len(text.split())

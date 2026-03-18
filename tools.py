from langchain.tools import tool
from datetime import datetime
import math


@tool
def calculator(expression: str) -> str:
    """Solve mathematical expressions. Supports basic arithmetic, powers, square roots, etc.
    Examples: '2 + 2', '10 * 5', 'math.sqrt(16)', '2 ** 10'
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Invalid expression: {e}"


@tool
def word_counter(text: str) -> str:
    """Counts words, characters, and sentences in a given text."""
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    sentences = len([s for s in text.split('.') if s.strip()])
    return (
        f"Words: {words} | "
        f"Characters: {chars} | "
        f"Characters (no spaces): {chars_no_spaces} | "
        f"Sentences: {sentences}"
    )


@tool
def unit_converter(query: str) -> str:
    """Convert between common units. Format: '<value> <from_unit> to <to_unit>'
    Supported: km/miles, kg/lbs, celsius/fahrenheit, meters/feet, liters/gallons
    Examples: '5 km to miles', '100 celsius to fahrenheit', '70 kg to lbs'
    """
    try:
        parts = query.lower().strip().split()
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[3]

        conversions = {
            ("km", "miles"): lambda x: x * 0.621371,
            ("miles", "km"): lambda x: x * 1.60934,
            ("kg", "lbs"): lambda x: x * 2.20462,
            ("lbs", "kg"): lambda x: x * 0.453592,
            ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
            ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
            ("meters", "feet"): lambda x: x * 3.28084,
            ("feet", "meters"): lambda x: x * 0.3048,
            ("liters", "gallons"): lambda x: x * 0.264172,
            ("gallons", "liters"): lambda x: x * 3.78541,
        }

        key = (from_unit, to_unit)
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        return f"Conversion from {from_unit} to {to_unit} not supported."
    except Exception as e:
        return f"Error: {e}. Use format: '<value> <from_unit> to <to_unit>'"


@tool
def get_current_datetime(query: str = "") -> str:
    """Get the current date and time. Use this when asked about today's date or current time."""
    now = datetime.now()
    return (
        f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
        f"Current time: {now.strftime('%I:%M %p')}"
    )


@tool
def text_analyzer(text: str) -> str:
    """Analyze text and return statistics: word frequency, average word length,
    most common words, and readability info."""
    words = text.lower().split()
    if not words:
        return "No text provided."

    word_freq = {}
    for word in words:
        clean = ''.join(c for c in word if c.isalpha())
        if clean:
            word_freq[clean] = word_freq.get(clean, 0) + 1

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    avg_len = sum(len(w) for w in word_freq) / len(word_freq) if word_freq else 0
    unique = len(word_freq)
    total = len(words)

    return (
        f"Total words: {total} | Unique words: {unique}\n"
        f"Average word length: {avg_len:.1f} chars\n"
        f"Top words: {', '.join(f'{w}({c})' for w, c in top_words)}"
    )

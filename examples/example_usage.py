import pandas as pd
from semantic_sense import AnomalyDetector

# Example DataFrame
df = pd.DataFrame({
    "Age": [25, 30, 29, 31, 200],
    "Amount": [100, 120, 110, 115, 9999],
    "Category": ["A", "A", "B", "B", "Z"]
})

print(df)

# Hybrid mode (default)
detector = AnomalyDetector()
results = detector.detect(df)
print(results)

# Text-only mode
detector_text = AnomalyDetector(mode="hybrid",  numeric_weight=1.5)
results_text = detector_text.detect(df)
print(results_text)

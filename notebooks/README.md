# Notebooks

Exploration only (e.g. `01_eda.ipynb` for exercise 1). Notebooks import from `heartnn`
instead of redefining logic, so anything reused lives in `src/` and is tested:

```python
from heartnn.data import load_clean_data
X, y = load_clean_data()
```

Clear outputs before committing large notebooks, or keep the rendered version small.

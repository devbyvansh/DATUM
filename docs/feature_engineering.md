# Feature engineering

Feature engineering is deterministic and runs only after a preparation plan has been approved. The original upload is never overwritten.

Supported registered operations include:

- Mean, median, mode, constant, forward-fill, and backward-fill imputation.
- Missing-value indicator columns.
- Text length and word-count columns.
- Date year, quarter, month, day, weekday, and hour columns.
- One-hot encoding with unknown-category handling.
- Standard, min-max, and robust numeric scaling.
- Safe ratios with zero denominators converted to missing values.
- Duplicate removal.

Operations are passed in the preparation configuration, recorded in the transformation audit trail, and checked for output-column collisions and invalid numeric values. Semantic target leakage detection remains heuristic; users should explicitly identify and review target columns before downstream model training.

# Profile Forecasting

This repository contains a PyTorch project for forecasting future profiles in a time series using a fixed number of past profiles as input.  
The project focuses on building a clean, extensible forecasting pipeline that can be improved and scaled as more data becomes available.  
The current implementation is a work in progress, with a particular interest in extrapolation to future time steps.

---

## Project Structure

```plaintext
.
├── main.py        # Entry point
├── data.py        # Data loading, standardization, dataset splits
├── model.py       # Neural network definition
├── engine.py      # Training, evaluation, prediction
├── plotting.py    # Plotting utilities
├── utils.py       # Helpers (seed, device)
├── prediction.png
├── README.md
└── approximated_data.csv
```

---

## Data

The input data is provided as a CSV file:  
- The first column represents a spatial coordinate (used as the x-axis for plotting).  
- Each remaining column represents a profile measured at a different time step.


### Notes:
- The data originates from an actual mining site in Belarus. However:
  - The exact mine is unknown.
  - The measurement procedure is not documented.
  - The data was already approximated before being used in this project.
- For these reasons, the dataset is treated as a generic profile time series, and no domain-specific interpretation is assumed.
- The emphasis is on **modeling**, **data handling**, and **forecasting**, rather than on geological conclusions.

---

## Problem Setup

- Each profile is represented as a 1D vector.
- The model uses the previous `n_hist` profiles as input.
- **Task:** Predict the next profile in time.

### Time Series Split:
- The time series is split chronologically into:
  - **Training**
  - **Validation**
  - **Test**
- No shuffling across time is performed.

---

## Standardization

- Profiles are standardized **per spatial position**:
  - Mean and standard deviation are computed **only on the training portion** of the time series.
  - The same statistics are applied to validation and test data.
- Model predictions are decoded back to the original scale before plotting and comparison.

---

## Model

- The current implementation uses a **compact Conv1D-based model**.
- The code is written to be **model-agnostic**, so alternative architectures (e.g., recurrent or transformer-based models) can be added without changing the overall pipeline.

---

### Run the Experiment:
```bash
python main.py
```

### This will:
1. Load and standardize the data.
2. Split it into **train**, **validation**, and **test** sets.
3. Train the model.
4. Evaluate on validation data.
5. Generate predictions on the test set.
6. Decode predictions back to original units.
7. Add the predicted profile as a new column.
8. Produce a plot for comparison.

## Example prediction

![Example prediction](profiles.png)

---

## Status and Future Work

This project is ongoing.  

### Planned and Possible Extensions:
- Improving extrapolation performance.
- Experimenting with different model architectures.
- Incorporating more data as it becomes available.
- Refining evaluation strategies for long-horizon forecasting.
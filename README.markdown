# Time Series Forecasting with LSTM

This project demonstrates the use of Long Short-Term Memory (LSTM) networks to forecast time series data. The dataset consists of time series observations recorded at 15-minute intervals, and the objective is to predict future values based on historical patterns.

## Data

The dataset is stored in the file `D_15m_with_time.csv` and includes the following columns:

- **Date**: The date of the observation (e.g., `2018-03-28`).
- **Time**: The time of the observation (e.g., `00:00:00`).
- **Count**: The numerical value to be predicted (e.g., event counts or measurements).

The dataset contains **35,424 samples**, with each sample representing a 15-minute interval, providing a high-resolution time series for analysis.

## Code Overview

The code, implemented in a Jupyter notebook (`base_15min.ipynb`), performs the following key steps:

1. **Data Preprocessing**:
   - Combines the `Date` and `Time` columns into a single `Datetime` column and sets it as the index.
   - Normalizes the `Count` column using `MinMaxScaler` to scale values between 0 and 1.
   - Creates lagged features (7 time steps) to capture temporal dependencies in the time series.

2. **Data Splitting**:
   - Splits the dataset into:
     - **Training set**: 80% (28,339 samples)
     - **Validation set**: 10% (3,542 samples)
     - **Test set**: 10% (3,543 samples)

3. **Hyperparameter Optimization**:
   - The **ARDE-N-BEATES algorithm** is employed to optimize the hyperparameters of the LSTM model.
   - This algorithm iteratively explores the hyperparameter space, testing combinations of parameters such as:
     - Number of LSTM units
     - Learning rate
     - Batch size
   - It evaluates each combination by training the LSTM model and selects the set that minimizes the validation loss, enhancing the model's forecasting accuracy.
   - This automated tuning process is essential for achieving optimal performance without manual trial-and-error.

4. **Model Definition and Training**:
   - Defines an LSTM model using the hyperparameters identified by ARDE-N-BEATES.
   - Trains the model on the training set, with early stopping based on validation loss to prevent overfitting.

5. **Prediction and Evaluation**:
   - Generates predictions on the test set.
   - Evaluates performance using the following metrics:
     - **Mean Squared Error (MSE)**: 2216.97
     - **Mean Absolute Percentage Error (MAPE)**: 7.07%
     - **Mean Absolute Error (MAE)**: 32.51
     - **Root Mean Squared Error (RMSE)**: 47.08

## Dependencies

The project requires the following Python libraries:

- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `keras`
- `tslearn`
- `json`
- `pickle`

Install them using pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn statsmodels keras tslearn
```

## Running the Code

To run the project:

1. Ensure Python 3.x is installed on your system.
2. Install the required dependencies listed above.
3. Place the `D_15m_with_time.csv` file in the same directory as the Jupyter notebook (`base_15min.ipynb`).
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `base_15min.ipynb` and execute the cells sequentially.

## Results

The LSTM model, optimized with ARDE-N-BEATES, achieved the following performance on the test set:
- **MSE**: 2216.97
- **MAPE**: 7.07%
- **MAE**: 32.51
- **RMSE**: 47.08

These metrics demonstrate the model's capability to predict time series values with reasonable accuracy, thanks to the effective hyperparameter tuning by ARDE-N-BEATES.

## Future Work

Potential improvements include:
- Exploring additional LSTM layers or bidirectional LSTMs.
- Testing other optimization algorithms alongside ARDE-N-BEATES.
- Adjusting the number of lagged features for better performance.
- Applying the model to additional time series datasets for validation.
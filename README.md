# VMD-PSO-BiLSTM for Time Series Forecasting

This project implements a hybrid deep learning model (VMD-PSO-BiLSTM) for time series forecasting, based on the provided "RAG quality.xlsx" dataset.

The model first decomposes the target signal into several Intrinsic Mode Functions (IMFs) using Variational Mode Decomposition (VMD). Then, it uses Particle Swarm Optimization (PSO) to find the optimal hyperparameters for a Bidirectional LSTM (BiLSTM) network. Finally, it trains a separate BiLSTM model for each IMF and aggregates their predictions to produce the final forecast.

##  methodology

The workflow is divided into four main parts:

1.  **Part 1: Data Loading and VMD Decomposition**
    * Loads the "RAG quality.xlsx" dataset.
    * Decomposes the original target signal into 5 IMFs using VMD.

2.  **Part 2: PSO Hyperparameter Optimization**
    * Uses Particle Swarm Optimization (PSO) to find the best hyperparameters for the BiLSTM model.
    * The parameters being optimized are: L2 regularization, learning rate, and the number of LSTM units.
    * The cost function for PSO is the Root Mean Squared Error (RMSE) of a model trained on the original, non-decomposed data.

3.  **Part 3: Final Model Training**
    * Iterates through each of the 5 IMFs.
    * For each IMF, a new BiLSTM model is built using the optimal hyperparameters found by PSO.
    * Each model is trained to predict its corresponding IMF, using the dataset's features as input.
    * The final prediction is the sum of the predictions from all 5 models.

4.  **Part 4: Performance Evaluation**
    * The aggregated predictions are compared against the true original signal.
    * A comprehensive set of regression metrics is calculated (RMSE, R², MSE, MAE, RPD, MBE, MAPE).
    * Results are visualized with prediction plots, a regression plot, an error histogram, and a performance radar chart.

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zsfbct/RAG-Retrieval-VMD-PSO-BiLSTM.git](https://github.com/zsfbct/RAG-Retrieval-VMD-PSO-BiLSTM.git)
    cd RAG-Retrieval-VMD-PSO-BiLSTM
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your data:**
    * Place your `RAG quality.xlsx` file in the root directory of this project.

5.  **Run the project:**
    ```bash
    python main.py
    ```

## 📂 Project Structure
import numpy as np
import pandas as pd
import pyswarms as ps
import tensorflow as tf
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Local project imports
from utils import calculate_metrics
from plotting import (
    plot_original_signal, plot_imfs, plot_fft_of_imfs,
    plot_prediction_comparison, plot_regression,
    plot_error_histogram, plot_radar_chart
)
from model import build_bilstm_model
from optimizer import cost_function

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    # ==========================================================================
    #  Data Loading and VMD Decomposition
    # ==========================================================================
    file_path = 'RAG quality.xlsx'
    try:
        data_df = pd.read_excel(file_path, header=0)
        data_df = data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        original_signal = data_df.iloc[:, -1].values
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please place the file in the project's root directory.")
        return

    plot_original_signal(original_signal)

    # VMD Parameters
    alpha = 356
    tau = 0.
    K = 5
    DC = 0
    init = 1
    tol = 1e-7

    imfs, u_hat, omega = VMD(original_signal, alpha, tau, K, DC, init, tol)
    print(f"VMD decomposition complete. Shape of IMFs: {imfs.shape}")

    imfs_df = pd.DataFrame(imfs.T, columns=[f'IMF_{i+1}' for i in range(K)])

    # Plot VMD results
    plot_imfs(original_signal, imfs)
    plot_fft_of_imfs(imfs)

    # ==========================================================================
    #  PSO Hyperparameter Optimization
    # ==========================================================================
    full_data_matrix = data_df.values
    num_samples = full_data_matrix.shape[0]
    train_size = int(0.7 * num_samples)

    pso_X = full_data_matrix[:, :-1]
    pso_y = full_data_matrix[:, -1].reshape(-1, 1)

    pso_X_train, pso_y_train = pso_X[:train_size], pso_y[:train_size]

    scaler_X_pso = MinMaxScaler()
    scaler_y_pso = MinMaxScaler()

    pso_X_train_norm = scaler_X_pso.fit_transform(pso_X_train)
    pso_y_train_norm = scaler_y_pso.fit_transform(pso_y_train)
    
    pso_X_train_reshaped = pso_X_train_norm.reshape((pso_X_train_norm.shape[0], 1, pso_X_train_norm.shape[1]))

    # Wrapper for PySwarms compatibility
    def pso_objective_function(x):
        # This wrapper passes the data and scaler to the cost function
        return cost_function(x, pso_X_train_reshaped, pso_y_train_norm, scaler_y_pso)

    # PSO Configuration
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.8}
    n_particles = 10
    max_iters = 15
    dimensions = 3
    bounds = (np.array([1e-8, 0.0001, 2]), np.array([1e-1, 0.1, 100]))

    print("Starting PSO for hyperparameter optimization...")
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(pso_objective_function, iters=max_iters)

    print("PSO finished.")
    print(f"Best cost (RMSE): {best_cost}")
    print(f"Best hyperparameters (L2, LR, Units): {best_pos}")

    l2_optimal = abs(best_pos[0])
    lr_optimal = abs(best_pos[1])
    units_optimal = abs(int(round(best_pos[2])))

    # ==========================================================================
    #  Final Model Training for Each IMF
    # ==========================================================================
    print("\nStarting final model training for each IMF...")

    X_features = data_df.iloc[:, :-1].values
    
    train_preds_imfs = []
    test_preds_imfs = []

    for i in range(K):
        print(f"--- Modeling for IMF {i+1}/{K} ---")
        
        y_target_imf = imfs_df.iloc[:, i].values.reshape(-1, 1)
        
        X_train, X_test = X_features[:train_size, :], X_features[train_size:, :]
        y_train, y_test = y_target_imf[:train_size], y_target_imf[train_size:]
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_norm = scaler_X.fit_transform(X_train)
        X_test_norm = scaler_X.transform(X_test)
        
        y_train_norm = scaler_y.fit_transform(y_train)
        y_test_norm = scaler_y.transform(y_test)
        
        X_train_reshaped = X_train_norm.reshape((X_train_norm.shape[0], 1, X_train_norm.shape[1]))
        X_test_reshaped = X_test_norm.reshape((X_test_norm.shape[0], 1, X_test_norm.shape[1]))
        
        input_shape_final = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
        
        model_final = build_bilstm_model(input_shape_final, units_optimal, l2_optimal)
        
        steps_per_epoch = int(np.ceil(len(X_train_reshaped) / 32))
        boundary_step = 350 * steps_per_epoch
        
        lr_schedule = PiecewiseConstantDecay(
            boundaries=[boundary_step],
            values=[lr_optimal, lr_optimal * 0.2]
        )
        
        optimizer_final = Adam(learning_rate=lr_schedule, clipnorm=1.0)
        model_final.compile(optimizer=optimizer_final, loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        
        model_final.fit(
            X_train_reshaped, y_train_norm,
            epochs=500,
            batch_size=32,
            validation_data=(X_test_reshaped, y_test_norm),
            verbose=0,
            callbacks=[early_stopping]
        )
        
        train_pred_norm = model_final.predict(X_train_reshaped, verbose=0)
        test_pred_norm = model_final.predict(X_test_reshaped, verbose=0)
        
        train_pred = scaler_y.inverse_transform(train_pred_norm.reshape(-1, 1))
        test_pred = scaler_y.inverse_transform(test_pred_norm.reshape(-1, 1))
        
        train_preds_imfs.append(train_pred.flatten())
        test_preds_imfs.append(test_pred.flatten())

    # Aggregate predictions
    train_pred_final = np.sum(np.array(train_preds_imfs), axis=0)
    test_pred_final = np.sum(np.array(test_preds_imfs), axis=0)

    y_true_train = original_signal[:train_size]
    y_true_test = original_signal[train_size:]

    # ==========================================================================
    #  Performance Evaluation and Visualization
    # ==========================================================================
    train_metrics = calculate_metrics(y_true_train, train_pred_final)
    test_metrics = calculate_metrics(y_true_test, test_pred_final)

    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    print("\n--- Performance Metrics Summary ---")
    print(metrics_df.round(4))

    # Visualization
    plot_prediction_comparison(
        y_true_train, 
        train_pred_final, 
        train_metrics, 
        title='Comparison of Training Set Prediction Results'
    )
    
    plot_prediction_comparison(
        y_true_test,
        test_pred_final,
        test_metrics,
        title='Comparison of Test Set Prediction Results'
    )
    
    plot_regression(y_true_test, test_pred_final, test_metrics)
    plot_error_histogram(y_true_test, test_pred_final)
    plot_radar_chart(train_metrics, test_metrics)

if __name__ == "__main__":
    main()
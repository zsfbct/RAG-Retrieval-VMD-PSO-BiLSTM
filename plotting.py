# ==============================================================================
# File: plotting.py
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np

# Set global plot style
plt.style.use('seaborn-v0_8-whitegrid')

def plot_original_signal(signal):
    """Plots the initial raw signal."""
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title('Original Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.show()

def plot_imfs(signal, imfs, title='VMD Decomposition Results'):
    """Plots the original signal and all its IMFs."""
    n_imfs = imfs.shape[0]
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(12, 2 * (n_imfs + 1)), sharex=True)
    
    axes[0].plot(signal)
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Amplitude')
    
    for i in range(n_imfs):
        axes[i+1].plot(imfs[i, :])
        axes[i+1].set_title(f'IMF {i+1}')
        axes[i+1].set_ylabel('Amplitude')
        
    axes[-1].set_xlabel('Time Step')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_fft_of_imfs(imfs, fs=1):
    """Plots the FFT spectrum for each IMF."""
    n_imfs = imfs.shape[0]
    
    plt.figure(figsize=(12, 2 * n_imfs))
    for i in range(n_imfs):
        ax = plt.subplot(n_imfs, 1, i + 1)
        N = len(imfs[i, :])
        yf = np.fft.fft(imfs[i, :])
        xf = np.fft.fftfreq(N, 1 / fs)[:N//2]
        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax.set_title(f'FFT Spectrum of IMF {i+1}')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_prediction_comparison(y_true, y_pred, metrics, title=''):
    """Plots the true vs. predicted values."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, 'r-*', label='Real Value', markersize=5)
    plt.plot(y_pred, 'b-o', label='VMD-PSO-BiLSTM Predicted Value', markersize=5, alpha=0.7)
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Value')
    title_str = (f"{title}\n"
                 f"(R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}, "
                 f"MSE = {metrics['MSE']:.4f}, RPD = {metrics['RPD']:.4f})")
    plt.title(title_str)
    plt.show()

def plot_regression(y_true, y_pred, metrics):
    """Plots a scatter plot of true vs. predicted values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    
    lims = [
        np.min([y_true.min(), y_pred.min()]),
        np.max([y_true.max(), y_pred.max()]),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    plt.xlabel('Real Value')
    plt.ylabel('Predicted Value')
    title_str = (f"Test Set Regression Plot\n"
                 f"(R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f})")
    plt.title(title_str)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_error_histogram(y_true, y_pred):
    """Plots a histogram of the prediction errors."""
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Histogram for Test Set')
    plt.axvline(errors.mean(), color='r', linestyle='dashed', linewidth=2, label=f'Mean Error: {errors.mean():.4f}')
    plt.legend()
    plt.show()

def plot_radar_chart(train_metrics, test_metrics):
    """Plots a radar chart comparing train and test metrics."""
    labels = ['MAE', 'RMSE', 'MSE', '1-R²', 'MAPE']
    num_vars = len(labels)

    # Use 1-R² and scale MAPE so smaller is better for all
    train_values = [train_metrics['MAE'], train_metrics['RMSE'], train_metrics['MSE'], 1 - train_metrics['R2'], train_metrics['MAPE']/100]
    test_values = [test_metrics['MAE'], test_metrics['RMSE'], test_metrics['MSE'], 1 - test_metrics['R2'], test_metrics['MAPE']/100]
    
    combined = np.array([train_values, test_values])
    max_vals = np.maximum(combined.max(axis=0), 1e-9)
    train_scaled = combined[0, :] / max_vals
    test_scaled = combined[1, :] / max_vals

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    train_scaled = np.concatenate((train_scaled, [train_scaled[0]]))
    test_scaled = np.concatenate((test_scaled, [test_scaled[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, train_scaled, color='red', alpha=0.25)
    ax.plot(angles, train_scaled, color='red', linewidth=2, label='Train Set')
    ax.fill(angles, test_scaled, color='blue', alpha=0.25)
    ax.plot(angles, test_scaled, color='blue', linewidth=2, label='Test Set')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Performance Metrics Comparison (Normalized)', size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from model import build_bilstm_model # Import from our model.py

def cost_function(params, X_train, y_train, scaler_y):
    """
    Cost function for PSO. Trains a BiLSTM model and returns RMSE.
    params: A numpy array with shape (n_particles, n_dimensions)
    """
    n_particles = params.shape[0]
    rmse_scores = np.zeros(n_particles)
    
    input_shape = (X_train.shape[1], X_train.shape[2])

    for i in range(n_particles):
        # Unpack hyperparameters for the i-th particle
        l2_reg = abs(params[i, 0])
        learn_rate = abs(params[i, 1])
        num_units = abs(int(round(params[i, 2])))

        # Ensure num_units is at least 1
        if num_units < 1:
            num_units = 1

        model = build_bilstm_model(input_shape, num_units, l2_reg)

        # Configure training options
        optimizer = Adam(learning_rate=learn_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Train the model (silently)
        model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=32)

        # Predict and calculate RMSE
        y_pred_norm = model.predict(X_train, verbose=0)
        
        # Use the provided scaler to inverse transform
        y_pred = scaler_y.inverse_transform(y_pred_norm)
        y_true = scaler_y.inverse_transform(y_train)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_scores[i] = rmse

    return rmse_scores
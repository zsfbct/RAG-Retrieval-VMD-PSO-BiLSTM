from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras import regularizers

def build_bilstm_model(input_shape, units, l2_reg):
    """
    Builds a Bidirectional LSTM model.
    """
    model = Sequential()
    model.add(Bidirectional(
        LSTM(units, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg)),
        input_shape=input_shape
    ))
    model.add(Dense(1)) # Output layer
    return model
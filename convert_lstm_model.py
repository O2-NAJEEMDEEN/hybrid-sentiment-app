
import tensorflow as tf

# Load the legacy HDF5 model
model = tf.keras.models.load_model("models/lstm_model.h5")

# Export as TensorFlow SavedModel format (for Keras 3+ compatibility)
model.export("models/lstm_model_v3")

print("✅ Model exported successfully to: models/lstm_model_v3/")

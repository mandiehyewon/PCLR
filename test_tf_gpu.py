import tensorflow as tf

# Check if TensorFlow is built with GPU support
if tf.test.is_built_with_cuda():
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow is using the following GPU(s): {gpus}")
        # Set memory growth for GPUs if required
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU devices found. TensorFlow will run on the CPU.")
else:
    print("TensorFlow is not built with GPU support. It will run on the CPU.")
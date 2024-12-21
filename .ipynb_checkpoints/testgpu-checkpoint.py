import tensorflow as tf

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available!")
    print("Detected GPUs:")
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("No GPU detected. TensorFlow is using the CPU.")

# Perform a simple computation on the GPU (if available)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    tensor1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    tensor2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    result = tf.add(tensor1, tensor2)

print("Result of Tensor Addition:")
print(result)


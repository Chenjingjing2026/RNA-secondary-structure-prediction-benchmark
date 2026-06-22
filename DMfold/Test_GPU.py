import tensorflow as tf
import subprocess

def check_gpu():
    """Checks if TensorFlow can see the GPU and prints GPU information (TensorFlow 1.x)."""

    print("TensorFlow version:", tf.__version__)

    if tf.test.is_gpu_available():
        print("GPU is available.")
        print("GPU Device Name:", tf.test.gpu_device_name())
    else:
        print("GPU is not available.")

def get_cuda_version():
    """Gets the CUDA version using nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        cuda_version = output.decode("utf-8").strip()
        return cuda_version
    except FileNotFoundError:
        return "nvidia-smi not found.  Make sure it's in your PATH."
    except subprocess.CalledProcessError as e:
        return "Error running nvidia-smi: {}".format(e)

def get_cudnn_version():
    """Attempts to infer the cuDNN version (best-effort)."""
    try:
        # cuDNN doesn't have a direct command-line query.
        # Try to find it by looking at the cuDNN library file.
        cudnn_path = tf.sysconfig.get_lib() # This might not always be correct
        if cudnn_path:
            # This is a very rough way to check, and might not be accurate.
            # It's better to check the installation documentation for your specific setup.
            import os
            for filename in os.listdir(cudnn_path):
                if "libcudnn" in filename and ".so" in filename:
                    return filename # Return the filename as an approximation
            return "cuDNN library found, but version could not be inferred from filename."
        else:
            return "cuDNN path not found.  Make sure cuDNN is installed correctly."

    except Exception as e:
        return "Error getting cuDNN version: {}".format(e)

if __name__ == "__main__":
    check_gpu()
    cuda_version = get_cuda_version()
    cudnn_version = get_cudnn_version()

    print("\nCUDA Version:", cuda_version)
    print("cuDNN Version:", cudnn_version)
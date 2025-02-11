import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# If needed, initialize the CUDA driver (usually done at program start)
# import pycuda.autoinit

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_image_paths, batch_size, input_shape, max_samples, cache_file):
        """
        :param calibration_image_paths: List of file paths to calibration images.
        :param batch_size: Number of images per calibration batch.
        :param input_shape: Expected model input shape as (C, H, W).
        :param max_samples: Maximum number of calibration samples to use.
        :param cache_file: Path to the file for storing calibration cache.
        """
        # Call the base class __init__ as required.
        super(MyCalibrator, self).__init__()

        self.calibration_image_paths = calibration_image_paths
        self.batch_size = batch_size
        self.input_shape = input_shape  # (C, H, W)
        self.max_samples = max_samples
        self.cache_file = cache_file
        self.current_index = 0

        # Allocate device memory for one batch.
        total_size = int(np.prod((self.batch_size, *self.input_shape)) * np.dtype(np.float32).itemsize)
        self.device_input = cuda.mem_alloc(total_size)

        # Load calibration data (limit the number of images to max_samples)
        self.data = self.load_calibration_data()

    def load_calibration_data(self):
        """
        Loads and preprocesses calibration images.
        Preprocessing:
          - Reads the image using OpenCV.
          - Converts from BGR to RGB.
          - Resizes the image to (width, height) derived from input_shape (input_shape is (C, H, W)).
          - Normalizes pixel values to [0, 1].
          - Transposes the image to (C, H, W).
          - Adds a batch dimension resulting in shape (1, C, H, W).
        Returns a list of preprocessed images.
        """
        # Limit the paths to max_samples
        limited_paths = self.calibration_image_paths[:self.max_samples]
        data = []
        # Extract target dimensions from input_shape
        # input_shape is (C, H, W)
        _, height, width = self.input_shape
        total_images = len(limited_paths)
        for i, path in enumerate(limited_paths):
            image = cv2.imread(path)
            if image is None:
                print(f"[WARNING] Could not load image at {path}. Skipping.")
                continue
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize image to (width, height)
            image = cv2.resize(image, (width, height))
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            # Change to channel-first format: (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            # Add a batch dimension -> shape becomes (1, C, H, W)
            image = np.expand_dims(image, axis=0)
            data.append(image)
            print(f"[DEBUG] Processed calibration image {i+1}/{total_images}: {path}")
        return data

    def get_batch(self, names):
        """
        Returns the next batch of preprocessed calibration data.
        If there are no more images, returns None.
        """
        if self.current_index + self.batch_size > len(self.data):
            return None

        # Concatenate self.batch_size images (each with shape (1, C, H, W)) along axis 0.
        batch = np.ascontiguousarray(
            np.concatenate(self.data[self.current_index:self.current_index + self.batch_size], axis=0)
        )
        self.current_index += self.batch_size

        # Copy the batch data to device memory.
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """
        If a calibration cache exists, return its contents. Otherwise, return None.
        """
        try:
            with open(self.cache_file, "rb") as f:
                print("[DEBUG] Calibration cache found. Loading cache...")
                return f.read()
        except Exception:
            print("[DEBUG] No calibration cache found.")
            return None

    def write_calibration_cache(self, cache):
        """
        Writes the calibration cache to a file.
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[DEBUG] Calibration cache written to {self.cache_file}")


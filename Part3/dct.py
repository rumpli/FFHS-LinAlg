import os
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt

def get_dct_array(n: int) -> np.ndarray:
    """Returns the transformation matrix used for DCT and inversed DCT of length n.

    Args:
        n (int): Length of the matrix.

    Returns:
        np.ndarray: Transformation matrix.
    """
    transformation_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                transformation_matrix[i][j] = np.sqrt(1 / n)
            else:
                transformation_matrix[i][j] = np.sqrt(2 / n) * np.cos((np.pi * i * (1 / 2 + j)) / n)
    return transformation_matrix

def dct_1d(v: np.ndarray) -> np.ndarray:
    """Computes the 1D Discrete Cosine Transform (DCT) of a vector.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: DCT of the input vector.
    """
    A = get_dct_array(len(v))
    return A @ v

def idct_1d(y: np.ndarray) -> np.ndarray:
    """Computes the inverse 1D Discrete Cosine Transform (DCT) of a vector.

    Args:
        y (np.ndarray): Input vector.

    Returns:
        np.ndarray: Inverse DCT of the input vector.
    """
    A = get_dct_array(len(y))
    return A.T @ y

def dct_2d_scipy(image: np.ndarray) -> np.ndarray:
    """Computes the 2D Discrete Cosine Transform (DCT) of an image using scipy.fftpack.dct.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: 2D DCT of the input image.
    """
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct_2d_scipy(dct_image: np.ndarray) -> np.ndarray:
    """Computes the inverse 2D Discrete Cosine Transform (DCT) of an image using scipy.fftpack.idct.

    Args:
        dct_image (np.ndarray): Input DCT image.

    Returns:
        np.ndarray: Inverse 2D DCT of the input image.
    """
    return idct(idct(dct_image.T, norm='ortho').T, norm='ortho')

def compress_dct(dct_image: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """Sets a percentage of high-frequency DCT coefficients to zero.

    Args:
        dct_image (np.ndarray): 2D DCT coefficients.
        keep_ratio (float): Ratio of coefficients to keep (0-1).

    Returns:
        np.ndarray: Compressed DCT image.
    """
    h, w = dct_image.shape
    keep_h = int(h * keep_ratio)
    keep_w = int(w * keep_ratio)
    compressed_dct = np.zeros_like(dct_image)
    compressed_dct[:keep_h, :keep_w] = dct_image[:keep_h, :keep_w]
    return compressed_dct

def process_image_with_dct(image_path: str, keep_ratio: float, threshold: float = 1e-10) -> None:
    """Loads an image, performs a 2D DCT, compresses it, and displays the original, transformed, and reconstructed images.

    Args:
        image_path (str): Path to the image file.
        keep_ratio (float): Ratio of coefficients to keep during compression.
        threshold (float): Threshold to ignore small differences.
    """
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    dct_image = dct_2d_scipy(image_array)
    compressed_dct_image = compress_dct(dct_image, keep_ratio=keep_ratio)
    reconstructed_image = idct_2d_scipy(compressed_dct_image)
    difference_image = np.abs(image_array - reconstructed_image)

    difference_image[difference_image < threshold] = 0

    plt.figure(figsize=(30, 20))
    fontsize = 30

    plt.subplot(1, 2, 1)
    plt.title("Original Image", fontsize=fontsize)
    plt.imshow(image_array, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image ({}% of coefficients)".format(keep_ratio * 100), fontsize=fontsize)
    plt.imshow(np.round(reconstructed_image), cmap='gray')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(30, 20))

    plt.subplot(1, 3, 1)
    plt.title("Difference Image", fontsize=fontsize)
    plt.imshow(difference_image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("2D DCT (log scaled)", fontsize=fontsize)
    plt.imshow(np.log(abs(dct_image) + 1), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Compressed DCT ({:.0f}% kept)".format(keep_ratio * 100), fontsize=fontsize)
    plt.imshow(np.log(abs(compressed_dct_image) + 1), cmap='gray')

    plt.tight_layout()
    plt.show()

def ask_user_for_keep_ratio() -> float:
    """Prompts the user to enter the keep ratio for DCT compression as a percentage (0-100).

    Returns:
        float: The keep ratio (0-1).
    """
    while True:
        try:
            keep_ratio_percentage = float(input("Enter the keep ratio for DCT compression (0-100): "))
            if 0 <= keep_ratio_percentage <= 100:
                return keep_ratio_percentage / 100
            print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")

def main() -> None:
    """Main function to demonstrate the DCT on vectors and images."""
    vectors = {
        "Linear Vector": np.array([0, 10, 20, 30, 40, 50, 60, 70]),
        "Changing Signals Vector": np.array([1,-1,1,-1,1,-1,1,-1]),
        "Constant Vector": np.array([10, 10, 10, 10, 10, 10, 10, 10]),
        "Impuls Vector": np.array([0, 0, 0, 100, 0, 0, 0, 0])
    }
    image_directory = "./images"
    image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.png', '.jpeg'))]
    keep_ratio = ask_user_for_keep_ratio()

    print("\n--- 1D DCT Example ---")

    for name, v in vectors.items():
        dct_v = dct_1d(v)
        reconstructed_v = idct_1d(dct_v)

        print(f"\n--- {name} ---")
        print("Original vector:", v)
        print("DCT (custom implementation):", np.round(dct_v, 3))
        print("Reconstructed vector:", np.round(reconstructed_v, 3))
        print("Transformation matrix:", np.round(get_dct_array(len(v)), 3))

    print("\n--- 2D DCT Example ---\n")
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        process_image_with_dct(image_path, keep_ratio=keep_ratio)

if __name__ == '__main__':
    main()

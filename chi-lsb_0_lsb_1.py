import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from PIL import Image
import matplotlib.pyplot as plt

def extract_lsb(image):
    """Extract the RGB LSB of the image"""
    img_data = np.array(image)
    r_lsb = img_data[:, :, 0] & 1  # Red channel LSB
    g_lsb = img_data[:, :, 1] & 1  # Green channel LSB
    b_lsb = img_data[:, :, 2] & 1  # Blue channel LSB
    return r_lsb, g_lsb, b_lsb

def chi_square_test(l):
    """Calculate the Chi-square test for the LSB"""
    counts = np.bincount(l)
    expected = [len(l.flatten()) / 2] * 2  # Assume 0 and 1 are uniformly distributed
    chi2_stat, p_val = chisquare(counts, expected)
    return p_val

def analyze_p_values(image_path):
    """Gradually expand the sample range and calculate the trend of p-values"""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    total_pixels = width * height

    # Extract LSBs
    r_lsb, g_lsb, b_lsb = extract_lsb(img)

    sample_percentages = np.arange(1, 101, 1)  # From 1% to 100%
    p_values_r, p_values_g, p_values_b = [], [], []

    for percent in sample_percentages:
        num_pixels = int((percent / 100) * total_pixels)  # Calculate the corresponding number of pixels
        y_limit = int((percent / 100) * height)  # Restrict the range starting from the top of the image

        # Sample the upper region
        r_sample = r_lsb[:y_limit, :].flatten()
        g_sample = g_lsb[:y_limit, :].flatten()
        b_sample = b_lsb[:y_limit, :].flatten()

        # Calculate p-value
        p_values_r.append(chi_square_test(r_sample))
        p_values_g.append(chi_square_test(g_sample))
        p_values_b.append(chi_square_test(b_sample))

    # Plot the trend of p-values
    plt.figure(figsize=(10, 5))
    plt.plot(sample_percentages, p_values_r, label="Red Channel", color="red")
    plt.plot(sample_percentages, p_values_g, label="Green Channel", color="green")
    plt.plot(sample_percentages, p_values_b, label="Blue Channel", color="blue")
    
    plt.axhline(y=0.05, color='black', linestyle='dashed', label="Significance threshold (0.05)")
    plt.xlabel("Sample Percentage of Image (%)")
    plt.ylabel("p-value")
    plt.title("LSB p-values vs. Sample Percentage")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    image_path = input("Please enter the image path: ")
    analyze_p_values(image_path)

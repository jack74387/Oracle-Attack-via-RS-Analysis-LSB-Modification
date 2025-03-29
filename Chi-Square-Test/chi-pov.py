import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def get_povs_frequencies(img, channel_name):
    unique, counts = np.unique(img, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    
    even_freqs = {i: freq_dict.get(i, 0) for i in range(0, 256, 2)}
    odd_freqs = {i+1: freq_dict.get(i+1, 0) for i in range(0, 256, 2)}
    
    total_sample_size = sum(even_freqs.values()) + sum(odd_freqs.values())
    return even_freqs, odd_freqs, total_sample_size

def compute_chi_square(even_freqs, odd_freqs, min_expected_freq=4):
    chi_square_stat = 0
    total_pairs = 0
    sample_size = sum(even_freqs.values()) + sum(odd_freqs.values())
    
    for i in range(0, 256, 2):
        n0 = even_freqs.get(i, 0)
        n1 = odd_freqs.get(i+1, 0)
        total = n0 + n1
        
        if total == 0:
            continue
        
        expected = total / 2
        if expected <= min_expected_freq:
            continue
        
        chi_square_stat += ((n0 - expected) ** 2 / expected) + ((n1 - expected) ** 2 / expected)
        total_pairs += 1
    
    return chi_square_stat, total_pairs, sample_size

def detect_stego(image_path, threshold=0.05, min_expected_freq=4):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Unable to read image, please check file path!")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channels = ['Red', 'Green', 'Blue']
    chi_square_results = []
    sample_ratios = np.linspace(0.1, 1.0, 10)  # different size of sample
    
    plt.figure(figsize=(10, 6))
    
    for i, channel_name in enumerate(channels):
        p_values = []
        
        for ratio in sample_ratios:
            sampled_img = img[:, :, i].flatten()
            sampled_size = int(len(sampled_img) * ratio)
            sampled_img = np.random.choice(sampled_img, sampled_size, replace=False)
            
            even_freqs, odd_freqs, _ = get_povs_frequencies(sampled_img, channel_name)
            chi_stat, total_pairs, _ = compute_chi_square(even_freqs, odd_freqs, min_expected_freq)
            
            if total_pairs == 0:
                # p_values.append(1.0)
                continue
            df = max(1, total_pairs - 1)
            p_value = chi2.sf(chi_stat, df=df)  
            # p_value = chi2.sf(chi_stat, df=127)
            p_values.append(p_value)
        
        plt.plot(sample_ratios * 100, p_values, marker='o', label=f'{channel_name} channel', color=channel_name.lower())
        
    plt.xlabel("Sample Ratio (%)")
    plt.ylabel("P value")
    plt.title("Chi-Square Test p-value vs Sample Ratio (RGB Channels)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
image_path = input("📂 Please enter the image file name: ")
detect_stego(image_path)

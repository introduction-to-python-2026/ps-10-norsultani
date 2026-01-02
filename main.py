from image_utils import load_image, edge_detection

def main():
    print("Starting image processing...")

    # 1. Load an image
    # First, let's ensure a sample image is available. This part assumes 'sample_image.jpg' is already downloaded.
    # In a real GitHub scenario, you might have the image alongside main.py or provide a path.
    image_path = 'sample_image.jpg'
    try:
        original_image = load_image(image_path)
        print(f"Image loaded successfully with shape: {original_image.shape}, dtype: {original_image.dtype}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Please ensure it exists.")
        return
    except Exception as e:
        print(f"An error occurred during image loading: {e}")
        return

    # 2. Suppress noise using a median filter
    # Convert to uint8 for skimage if it's not already, as median expects specific types.
    image_for_median = original_image.astype(np.uint8)
    clean_image = median(image_for_median, ball(3))
    print(f"Noise suppressed. Clean image shape: {clean_image.shape}, dtype: {clean_image.dtype}")

    # 3. Run the noise-free image through the edge_detection function
    edge_magnitude_image = edge_detection(clean_image)
    print(f"Edge detection applied. Resulting shape: {edge_magnitude_image.shape}, dtype: {edge_magnitude_image.dtype}")

    # 4. Convert the resulting edgeMAG array into a binary array
    # A fixed threshold is used here. For more advanced use, this could be dynamic.
    threshold = 50 # This value might need adjustment based on the image
    # Ensure the output is 0 or 255 for proper image saving and display
    edge_binary = (edge_magnitude_image > threshold).astype(np.uint8) * 255
    print(f"Binary edge image created. Shape: {edge_binary.shape}, dtype: {edge_binary.dtype}")

    # 5. Display the binary image and save it as .png file
    output_filename = 'my_edges.png'
    plt.imshow(edge_binary, cmap='gray')
    plt.title('Binary Edge Detected Image')
    plt.axis('off')
    plt.savefig(output_filename)
    print(f"Binary edge image saved as {output_filename}")
    print("Image processing complete.")

if _name_ == '_main_':
    main()


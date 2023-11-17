import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

CLASSES = ['no_data', 'clouds', 'artificial', 'cultivated', 'broadleaf', 'coniferous', 'herbaceous', 'natural', 'snow', 'water']

image_path = r"/Users/anton/Desktop/challenge_small/dataset/test1/images/10087.jpg"
model_path = r"/Users/anton/Desktop/UNET_Skyscan.h5"

model = load_model(model_path)

image = load_img(image_path, target_size=(256, 256))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array)
predicted_class = np.argmax(predictions, axis=-1)[0]

# Create a color map for visualization
colors = np.array([
    [0, 0, 0],  # 'no_data'
    [128, 128, 128],  # 'clouds'
    [255, 0, 0],  # 'artificial'
    [0, 255, 0],  # 'cultivated'
    [0, 0, 255],  # 'broadleaf'
    [255, 255, 0],  # 'coniferous'
    [255, 165, 0],  # 'herbaceous'
    [0, 255, 255],  # 'natural'
    [255, 255, 255],  # 'snow'
    [255, 255, 255],  # 'water'
], dtype=np.uint8)

# Map the predicted class to colors
colored_mask = colors[predicted_class]

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

# Display the predicted mask with colored legend
plt.subplot(1, 2, 2)
plt.imshow(colored_mask)
plt.title('Predicted Mask with Legend')

# Create a legend for the classes
legend_labels = [f"{i}: {CLASSES[i]}" for i in range(len(CLASSES))]
legend_colors = [colors[i] / 255.0 for i in range(len(CLASSES))]
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, legend_colors)]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title='Legend')

# Show the plot
plt.show()
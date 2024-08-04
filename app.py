import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Net
import keras
import numpy as np
import copy
import pickle
import shap
import skimage.io
import skimage.transform
import skimage.segmentation
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img

import os
if os.path.exists("checkpoint/shape_model.keras"):
    import numpy as np
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.inception_v3 import preprocess_input
    import shap
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers, models
    import pickle

    dataset_dir = 'dataset/'

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_input, 
        validation_split=0.2  # Use 20% of the data for validation
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(299, 299),  # Resize images to match model expected input
        batch_size=32,
        class_mode='sparse' , 
        subset='training'  
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(299, 299),
        batch_size=32,
        class_mode='sparse' ,
        subset='validation' 
    )

    def load_image_paths_labels(dataset_dir):
        image_paths = []
        labels = []
        class_names = os.listdir(dataset_dir)
        label_map = {class_name: i for i, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(dataset_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(label_map[class_name])

        return image_paths, labels, label_map

    dataset_dir = 'dataset'
    image_paths, labels, label_map = load_image_paths_labels(dataset_dir)

    # Split the data - 80% for training, 20% for testing
    x_train_paths, x_test_paths, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    def load_images(image_paths, target_size=(28,28)):
        images = []
        for img_path in image_paths:
            img = load_img(img_path, target_size=target_size, color_mode='grayscale')
            img_array = img_to_array(img)
            img_array /= 255.0  # Normalize to [0, 1]
            images.append(img_array)
        return np.array(images)

    # Load the datasets
    x_train = load_images(x_train_paths)
    x_test = load_images(x_test_paths)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Check the shapes
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    inputs = tf.keras.Input(shape=(28,28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="sparse_CNN")

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    history = model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    class_names = os.listdir("dataset")

    background = x_train[np.random.choice(x_train.shape[0], 5000, replace=False)]
    # Use DeepExplainer to explain predictions of the model
    # e = shap.DeepExplainer(model, background)

    explainer = shap.DeepExplainer(model, background)

    x_test_dict = dict()
    for i, l in enumerate(y_test):
        if len(x_test_dict)==10:
            break
        if l not in x_test_dict.keys():
            x_test_dict[l] = x_test[i]

    # Convert to list preserving order of classes
    x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]

    # Convert to tensor
    x_test_each_class = np.asarray(x_test_each_class)

    # Print shape of tensor
    print(f"x_test_each_class tensor has shape: {x_test_each_class.shape}")

    predictions = model.predict(x_test_each_class)

    shap_values = explainer.shap_values(x_test_each_class)
    # Plot shap values
    # shap.image_plot(shap_values, -x_test_each_class)

    model.save("shape_model.keras")
    

@st.cache_data
def load_lime_model():
    return keras.applications.inception_v3.InceptionV3()

@st.cache_data
def load_shap_model():
    return keras.models.load_model("checkpoint/shape_model.keras")

@st.cache_data
def load_pretrained_model():
    pre_trained_model_path = "checkpoint/model_epoch_30.pth"
    pre_trained_model_state_dict = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))['state_dict']

    model = Net(num_conv_in_channel=1,
                num_conv_out_channel=256,
                num_primary_unit=8,
                primary_unit_size=1152,
                num_classes=4,
                output_unit_size=16,
                num_routing=3,
                use_reconstruction_loss=True,
                regularization_scale=0.0005,
                input_width=28,
                input_height=28,
                cuda_enabled=False)

    model.load_state_dict(pre_trained_model_state_dict)
    model.eval()
    return model

@st.cache_data()
def load_data():
    with open("checkpoint/train_test_data.pkl", "rb") as f:
        x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded = pickle.load(f)
    return x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded

def create_shap_explainer(shap_model, x_train_loaded):
    background = x_train_loaded[np.random.choice(x_train_loaded.shape[0], 50, replace=False)]
    return shap.DeepExplainer(shap_model, background)

def get_shap_values(explainer, x_test_loaded, y_test_loaded):
    x_test_dict = {}
    for i, label in enumerate(y_test_loaded):
        if len(x_test_dict) == 10:
            break
        if label not in x_test_dict:
            x_test_dict[label] = x_test_loaded[i]
    x_test_each_class = np.array([x_test_dict[i] for i in sorted(x_test_dict)])
    return explainer.shap_values(x_test_each_class)

def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image

def generate_lime_explanation(uploaded_file):
    Xi = skimage.io.imread(uploaded_file)
    Xi = skimage.transform.resize(Xi, (299, 299))
    Xi = (Xi - 0.5) * 2
    if Xi.ndim == 2:
        Xi = skimage.color.gray2rgb(Xi)

    superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4, max_dist=200, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]
    marked_img = skimage.segmentation.mark_boundaries(Xi / 2 + 0.5, superpixels)
    
    num_perturb = 150
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    img_lime = perturb_image(Xi / 2 + 0.5, perturbations[0], superpixels)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(marked_img)
    ax[0].set_title('Marked Boundaries')
    ax[0].axis('off')

    ax[1].imshow(img_lime)
    ax[1].set_title('LIME Explanation')
    ax[1].axis('off')

    return fig

def generate_shap_explanation(uploaded_file, shap_model, x_train_loaded):
    explainer = create_shap_explainer(shap_model, x_train_loaded)
    shap_values = get_shap_values(explainer, x_test_loaded, y_test_loaded)

    img = load_img(uploaded_file, target_size=(28, 28), color_mode="grayscale")
    image_array = np.array(img) / 255.0
    batch_img = image_array[np.newaxis, ...]
    single_class_img = np.repeat(batch_img, 4, axis=0)[..., np.newaxis]
    corrected_shap_values = np.sum(shap_values, axis=-1)

    num_images = single_class_img.shape[0]
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))
    for i in range(num_images):
        axes[i, 0].imshow(single_class_img[i].squeeze(), cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(corrected_shap_values[i], cmap="coolwarm")
        axes[i, 1].axis("off")

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=axes.ravel().tolist(), orientation="horizontal")
    cbar.set_label("SHAP value")

    return fig

# Load resources
lime_model = load_lime_model()
shap_model = load_shap_model()
model = load_pretrained_model()
x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded = load_data()

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.Grayscale(), 
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)) 
])

st.title("Image Classification using CapsNet")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    image = transform(image)
    image = image.unsqueeze(0) 
    with torch.no_grad():
        output = model(image)
        v_magnitude = torch.sqrt((output**2).sum(dim=2, keepdim=True))
        pred = v_magnitude.data.max(1, keepdim=True)[1]

    predicted_class = class_names[pred.item()]
    st.markdown(f"<h3 style='color: green; text-align: center;'>Predicted Class: {predicted_class}</h3>", unsafe_allow_html=True)

    lime_fig = generate_lime_explanation(uploaded_file)
    st.pyplot(lime_fig)

    shap_fig = generate_shap_explanation(uploaded_file, shap_model, x_train_loaded)
    st.pyplot(shap_fig)
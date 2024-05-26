import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r"D:/Internships/Prodigy Infotech/FOOD_KCAL/food_model.keras")

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizing the image
    return img_array

# Function to predict the food item
def predict_food(img_path, class_labels):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    
    # Get the predicted class with the highest probability
    predicted_class_idx = np.argmax(predictions, axis=-1)[0]
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class

# Function to get calorie information based on the predicted class
def get_calorie_info(food_class, calorie_mapping):
    return calorie_mapping.get(food_class, "Calorie information not available")

# Define the class labels (replace with your actual class labels)
class_labels = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
    "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros",
    "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
    "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette",
    "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
    "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak",
    "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]

# Average calorie values for each food item
calorie_mapping = {
    'apple_pie': 237,
    'baby_back_ribs': 292,
    'baklava': 334,
    'beef_carpaccio': 122,
    'beef_tartare': 192,
    'beet_salad': 148,
    'beignets': 289,
    'bibimbap': 583,
    'bread_pudding': 291,
    'breakfast_burrito': 305,
    'bruschetta': 120,
    'caesar_salad': 481,
    'cannoli': 218,
    'caprese_salad': 235,
    'carrot_cake': 326,
    'ceviche': 142,
    'cheesecake': 401,
    'cheese_plate': 600,
    'chicken_curry': 293,
    'chicken_quesadilla': 460,
    'chicken_wings': 203,
    'chocolate_cake': 352,
    'chocolate_mousse': 355,
    'churros': 237,
    'clam_chowder': 200,
    'club_sandwich': 320,
    'crab_cakes': 250,
    'creme_brulee': 262,
    'croque_madame': 500,
    'cup_cakes': 305,
    'deviled_eggs': 64,
    'donuts': 452,
    'dumplings': 138,
    'edamame': 120,
    'eggs_benedict': 500,
    'escargots': 170,
    'falafel': 333,
    'filet_mignon': 275,
    'fish_and_chips': 595,
    'foie_gras': 446,
    'french_fries': 365,
    'french_onion_soup': 369,
    'french_toast': 280,
    'fried_calamari': 150,
    'fried_rice': 238,
    'frozen_yogurt': 214,
    'garlic_bread': 206,
    'gnocchi': 350,
    'greek_salad': 220,
    'grilled_cheese_sandwich': 287,
    'grilled_salmon': 412,
    'guacamole': 240,
    'gyoza': 200,
    'hamburger': 354,
    'hot_and_sour_soup': 95,
    'hot_dog': 290,
    'huevos_rancheros': 359,
    'hummus': 166,
    'ice_cream': 207,
    'lasagna': 290,
    'lobster_bisque': 248,
    'lobster_roll_sandwich': 436,
    'macaroni_and_cheese': 310,
    'macarons': 95,
    'miso_soup': 40,
    'mussels': 146,
    'nachos': 346,
    'omelette': 154,
    'onion_rings': 280,
    'oysters': 50,
    'pad_thai': 357,
    'paella': 300,
    'pancakes': 227,
    'panna_cotta': 350,
    'peking_duck': 337,
    'pho': 290,
    'pizza': 285,
    'pork_chop': 221,
    'poutine': 740,
    'prime_rib': 400,
    'pulled_pork_sandwich': 415,
    'ramen': 436,
    'ravioli': 230,
    'red_velvet_cake': 350,
    'risotto': 166,
    'samosa': 262,
    'sashimi': 200,
    'scallops': 75,
    'seaweed_salad': 106,
    'shrimp_and_grits': 250,
    'spaghetti_bolognese': 344,
    'spaghetti_carbonara': 379,
    'spring_rolls': 150,
    'steak': 679,
    'strawberry_shortcake': 285,
    'sushi': 130,
    'tacos': 226,
    'takoyaki': 342,
    'tiramisu': 240,
    'tuna_tartare': 170,
    'waffles': 310
}

# Test the model with a sample image
img_path = r"D:\Internships\Prodigy Infotech\FOOD_KCAL\food-101\images\hamburger\440931.jpg"  
predicted_class = predict_food(img_path, class_labels)
calorie_info = get_calorie_info(predicted_class, calorie_mapping)

print(f"Predicted Food Item: {predicted_class}")
print(f"Calorie Content for {predicted_class}: {calorie_info} kcal")

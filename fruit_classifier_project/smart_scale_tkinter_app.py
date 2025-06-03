import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
import threading
import time
import random
import sys

# --- Localization Import ---
from translations import get_translation, set_language, LANGUAGES
import translations  # Import the module to access CURRENT_LANG

# --- Configuration Import ---
from config import (
    MODEL_PATH as APP_MODEL_PATH, # Renaming to avoid conflict if user defines MODEL_PATH locally
    IMG_WIDTH,
    IMG_HEIGHT,
    TOP_N_PREDICTIONS_DISPLAY,
    MAX_SEARCH_RESULTS_DISPLAY,
    CLASS_NAMES_RAW,
    PRODUCT_DATA as APP_PRODUCT_DATA, # Renaming
    DEFAULT_WINDOW_GEOMETRY,
    USER_DATA_SUBDIR,
    CART_THUMBNAIL_SIZE,
    CART_ROW_HEIGHT
)

# --- UI Dimensions ---
CAMERA_DISPLAY_WIDTH = 900  # Fixed width for the camera display area
CAMERA_DISPLAY_HEIGHT = 600 # Fixed height for the camera display area

# --- Configuration ---
# MODEL_PATH = "fruit_classifier_mobilenet_best_v2.keras"  # Updated Default model 
# MOVED TO CONFIG.PY

# --- Argument Parsing for Model Path (Early in Script) ---
# Use APP_MODEL_PATH from config as the base default
current_model_path_to_use = APP_MODEL_PATH 
if len(sys.argv) > 2:
    if sys.argv[1] == "--model":
        cli_model_path = sys.argv[2]
        if cli_model_path.endswith(".keras"):
            current_model_path_to_use = cli_model_path # Override default if valid arg provided
            print(f"Command-line override: Using model '{current_model_path_to_use}'")
        else:
            print(f"Warning: Provided model path '{cli_model_path}' via --model does not end with .keras. Using config default: '{current_model_path_to_use}'")
    else:
        print(f"Warning: Unrecognized argument '{sys.argv[1]}'. To specify a model, use '--model <path_to_model.keras>'. Using config default: '{current_model_path_to_use}'")
elif len(sys.argv) == 2: # Handles cases like `python script.py --model` (missing path) or `python script.py some_other_arg`
    if sys.argv[1] == "--model":
        print(f"Warning: '--model' flag used but no path provided. Using config default: '{current_model_path_to_use}'")
    else:
        print(f"Warning: Unrecognized argument '{sys.argv[1]}'. Using config default: '{current_model_path_to_use}'")

# IMG_WIDTH = 100  # Should match the model's expected input 
# IMG_HEIGHT = 100 # Should match the model's expected input
# MOVED TO CONFIG.PY
IMAGE_SIZE_TUPLE = (IMG_WIDTH, IMG_HEIGHT) # For model input processing
# TOP_N_PREDICTIONS_DISPLAY = 5 # For correction dialog 
# MAX_SEARCH_RESULTS_DISPLAY = 10 # For correction dialog search
# MOVED TO CONFIG.PY

# --- Setup Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_CORRECTED_DATA_DIR = os.path.join(BASE_DIR, USER_DATA_SUBDIR) # Use from config
if not os.path.exists(USER_CORRECTED_DATA_DIR):
    os.makedirs(USER_CORRECTED_DATA_DIR)
    print(f"Created directory for user corrected data: {USER_CORRECTED_DATA_DIR}")

# --- Provided Class Names (Processed) ---
# User provided a 1-indexed list with format "1: Apple 10"
# CLASS_NAMES_RAW = """ ... """ # MOVED TO CONFIG.PY
CLASS_NAMES = [line.split(': ', 1)[1] for line in CLASS_NAMES_RAW.strip().split('\n') if ': ' in line]
NUM_CLASSES = len(CLASS_NAMES)

# --- Product Data (Prices per kg, Average weight in kg) ---
# PRODUCT_DATA = { ... } # MOVED TO CONFIG.PY

# --- SmartScaleApp Class ---
def get_base_product_name(detailed_class_name):
    name_lower = detailed_class_name.lower()
    # Prioritize more specific names if they imply different products/pricing often
    if "cabbage" in name_lower: return "Cabbage" # Before generic "green" or "red"
    if "apple" in name_lower: return "Apple"
    if "apricot" in name_lower: return "Apricot"
    if "avocado" in name_lower: return "Avocado"
    if "banana" in name_lower: return "Banana"
    if "beans" in name_lower: return "Beans"
    if "beetroot" in name_lower: return "Beetroot"
    if "blackberrie" in name_lower: return "Blackberry" # Spelling from class list
    if "blueberry" in name_lower: return "Blueberry"
    if "cactus fruit" in name_lower: return "Cactus fruit"
    if "caju seed" in name_lower: return "Caju seed"
    if "cantaloupe" in name_lower: return "Cantaloupe"
    if "carambula" in name_lower: return "Carambula"
    if "carrot" in name_lower: return "Carrot"
    if "cauliflower" in name_lower: return "Cauliflower"
    if "cherimoya" in name_lower: return "Cherimoya"
    if "cherry" in name_lower: return "Cherry" # Covers all cherry types
    if "chestnut" in name_lower: return "Chestnut"
    if "clementine" in name_lower: return "Clementine"
    if "cocos" in name_lower: return "Cocos"
    if "corn" in name_lower: return "Corn"
    if "cucumber" in name_lower: return "Cucumber"
    if "dates" in name_lower: return "Dates"
    if "eggplant" in name_lower: return "Eggplant"
    if "fig" in name_lower: return "Fig"
    if "ginger root" in name_lower: return "Ginger Root"
    if "gooseberry" in name_lower: return "Gooseberry"
    if "granadilla" in name_lower: return "Granadilla"
    if "grapefruit" in name_lower: return "Grapefruit"
    if "grape" in name_lower: return "Grape"  # Must be after grapefruit
    if "guava" in name_lower: return "Guava"
    if "hazelnut" in name_lower: return "Hazelnut"
    if "huckleberry" in name_lower: return "Huckleberry"
    if "kaki" in name_lower: return "Kaki"
    if "kiwi" in name_lower: return "Kiwi"
    if "kohlrabi" in name_lower: return "Kohlrabi"
    if "kumquats" in name_lower: return "Kumquats"
    if "lemon" in name_lower: return "Lemon"
    if "limes" in name_lower: return "Limes"
    if "lychee" in name_lower: return "Lychee"
    if "mandarine" in name_lower: return "Mandarine" # Covers Mandarine
    if "mango" in name_lower: return "Mango"
    if "mangostan" in name_lower: return "Mangostan"
    if "maracuja" in name_lower: return "Maracuja" # Passion fruit synonym
    if "melon piel de sapo" in name_lower: return "Melon Piel de Sapo"
    if "mulberry" in name_lower: return "Mulberry"
    if "nectarine" in name_lower: return "Nectarine"
    if "nut forest" in name_lower: return "Nut Forest"
    if "nut pecan" in name_lower: return "Nut Pecan"
    if "onion" in name_lower: return "Onion"
    if "orange" in name_lower: return "Orange"
    if "papaya" in name_lower: return "Papaya"
    if "passion fruit" in name_lower: return "Passion Fruit"
    if "peach" in name_lower: return "Peach"
    if "pear" in name_lower: return "Pear"
    if "pepino" in name_lower: return "Pepino"
    if "pepper" in name_lower: return "Pepper" # Covers all Pepper types
    if "physalis" in name_lower: return "Physalis"
    if "pineapple" in name_lower: return "Pineapple"
    if "pistachio" in name_lower: return "Pistachio"
    if "pitahaya" in name_lower: return "Pitahaya"
    if "plum" in name_lower: return "Plum"
    if "pomegranate" in name_lower: return "Pomegranate"
    if "pomelo sweetie" in name_lower: return "Pomelo Sweetie"
    if "potato" in name_lower: return "Potato"
    if "quince" in name_lower: return "Quince"
    if "rambutan" in name_lower: return "Rambutan"
    if "raspberry" in name_lower: return "Raspberry"
    if "redcurrant" in name_lower: return "Redcurrant"
    if "salak" in name_lower: return "Salak"
    if "strawberry" in name_lower: return "Strawberry"
    if "tamarillo" in name_lower: return "Tamarillo"
    if "tangelo" in name_lower: return "Tangelo"
    if "tomato" in name_lower: return "Tomato" # Covers all tomato types
    if "walnut" in name_lower: return "Walnut"
    if "watermelon" in name_lower: return "Watermelon"
    if "zucchini" in name_lower: return "Zucchini"
    return "Unknown"

def save_image_with_label(frame_to_save, correct_class_name, user_data_dir, image_size_tuple):
    print(f"Feedback received: Correct label is '{correct_class_name}'. Saving image.")
    # Resize the frame to the target size (e.g., 100x100) before saving
    resized_for_saving = cv2.resize(frame_to_save, image_size_tuple, interpolation=cv2.INTER_AREA)

    class_specific_dir = os.path.join(user_data_dir, correct_class_name.replace(' ', '_').replace('/', '_')) # Sanitize name
    if not os.path.exists(class_specific_dir):
        os.makedirs(class_specific_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Count existing files to make a unique name
    safe_class_name = correct_class_name.replace(' ', '_').replace('/', '_')
    existing_files = [f for f in os.listdir(class_specific_dir) if f.startswith(safe_class_name)]
    image_filename = f"{safe_class_name}_{timestamp}_{len(existing_files)+1}.png"
    image_save_path = os.path.join(class_specific_dir, image_filename)
    
    try:
        cv2.imwrite(image_save_path, resized_for_saving) # Save the resized image
        print(f"Image saved to: {image_save_path} (resized to {image_size_tuple[0]}x{image_size_tuple[1]})")
    except Exception as e:
        print(f"Error saving image {image_save_path}: {e}")
        messagebox.showerror("Save Error", f"Could not save corrected image for '{correct_class_name}'. Check console.")

class SmartScaleApp(tk.Tk):
    def __init__(self, model, class_names_list):
        super().__init__()
        self.model = model
        self.class_names = class_names_list
        self.title(get_translation("title"))
        self.geometry(DEFAULT_WINDOW_GEOMETRY) # Use from config
        self.configure(bg="#f0f0f0")

        self.cap = cv2.VideoCapture(0)
        self.camera_active = True
        self.last_captured_frame_for_processing = None # For model input
        self.last_captured_frame_for_saving = None # Potentially original aspect ratio for saving if needed, or just use processed one.
                                                   # For now, will save the processed one.
        self.current_item_details = None 
        self.raw_predictions = None # To store the full prediction array
        self.shopping_cart = []
        self.cart_item_photoimages = [] # To keep references to PhotoImage objects for the cart

        self._init_ui()
        self._update_camera_feed()

    def _init_ui(self):
        # --- Top Bar for Logo and Language ---
        top_bar_frame = ttk.Frame(self, padding=(10, 5, 10, 0)) # Add some padding
        top_bar_frame.pack(fill=tk.X, side=tk.TOP)

        # Replicate logo style from UI.py, text will be set by _update_texts
        self.logo_placeholder = tk.Label(top_bar_frame, 
                                         bg="gray", fg="white",
                                         font=("Helvetica", 24), width=13, height=2)
        self.logo_placeholder.pack(side=tk.LEFT, padx=(0, 20))

        self.language_button = ttk.Button(top_bar_frame, text="Español", command=self._toggle_language)
        self.language_button.pack(side=tk.RIGHT)
        # --- End Top Bar ---

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Left Panel: Camera and Item Info
        # Adjust left_panel width to be more suitable for the fixed camera size + controls
        left_panel_width = CAMERA_DISPLAY_WIDTH + 40 # Added padding for controls below
        left_panel = ttk.Frame(main_frame, width=left_panel_width)
        left_panel.pack_propagate(False) 
        left_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 10))

        # Create a frame for the camera display with fixed size
        camera_frame_container = ttk.Frame(left_panel, width=CAMERA_DISPLAY_WIDTH, height=CAMERA_DISPLAY_HEIGHT)
        camera_frame_container.pack_propagate(False) # Prevent children from resizing this frame
        # Pack it so it doesn't expand with left_panel's height changes. pady provides spacing.
        camera_frame_container.pack(pady=5, anchor=tk.N) 

        self.camera_label = ttk.Label(camera_frame_container, background="black")
        # camera_label fills the fixed-size camera_frame_container
        self.camera_label.pack(expand=True, fill=tk.BOTH)
        
        self.scan_button = ttk.Button(left_panel, text="Scan Item & Get Weight", command=self._scan_item_and_get_weight)
        self.scan_button.pack(pady=5, fill=tk.X)

        item_info_frame = ttk.LabelFrame(left_panel, text="Current Item Details", padding="10")
        item_info_frame.pack(pady=5, fill=tk.X)
        self.item_info_frame_widget = item_info_frame # Store reference

        self.item_name_label = ttk.Label(item_info_frame, text="Item: --", font=("Arial", 12))
        self.item_name_label.pack(anchor=tk.W)
        self.item_confidence_label = ttk.Label(item_info_frame, text="Confidence: --", font=("Arial", 10))
        self.item_confidence_label.pack(anchor=tk.W)
        self.item_weight_label = ttk.Label(item_info_frame, text="Weight (kg): --", font=("Arial", 10))
        self.item_weight_label.pack(anchor=tk.W)
        self.item_unit_price_label = ttk.Label(item_info_frame, text="Unit Price (€/kg): --", font=("Arial", 10))
        self.item_unit_price_label.pack(anchor=tk.W)
        self.item_total_price_label = ttk.Label(item_info_frame, text="Item Total (€): --", font=("Arial", 12, "bold"))
        self.item_total_price_label.pack(anchor=tk.W, pady=(5,0))

        # Buttons for workflow
        action_buttons_frame = ttk.Frame(left_panel)
        action_buttons_frame.pack(fill=tk.X, pady=5)

        self.add_to_cart_button = ttk.Button(action_buttons_frame, text="Add to Cart", command=self._add_to_cart, state=tk.DISABLED)
        self.add_to_cart_button.pack(side=tk.LEFT, expand=True, padx=(0,2))
        
        self.correct_button = ttk.Button(action_buttons_frame, text="Correct / Alternatives", command=self._show_correction_dialog, state=tk.DISABLED)
        self.correct_button.pack(side=tk.LEFT, expand=True, padx=(2,0))

        # Right Panel: Shopping Cart
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0)) # Right panel will take remaining space

        cart_frame = ttk.LabelFrame(right_panel, text="Shopping Cart", padding="10") # This text will be updated by _update_texts
        cart_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        # --- Style for Treeview Row Height ---
        style = ttk.Style()
        style.configure("Treeview", rowheight=CART_ROW_HEIGHT) # Use from config
        # --- End Style ---

        # Define columns for data, image will be in the #0 tree column
        cols = ("Item Name", "Weight (kg)", "Unit Price (€)", "Total (€)") 
        self.cart_tree = ttk.Treeview(cart_frame, columns=cols, show='tree headings', selectmode="none")
        
        # Setup the #0 column for images
        self.cart_tree.column("#0", width=60, minwidth=60, stretch=tk.NO, anchor=tk.W) # Adjusted width
        self.cart_tree.heading("#0", text="Image") # Placeholder, will be translated in _update_texts

        # Adjusted column widths for other data
        col_widths = {
            "Item Name": 120,
            "Weight (kg)": 80,
            "Unit Price (€)": 80,
            "Total (€)": 80
        }
        for col_key in cols:
            width = col_widths.get(col_key, 100) # Default to 100 if somehow not in map
            self.cart_tree.heading(col_key, text=col_key) # Placeholders, will be translated
            self.cart_tree.column(col_key, width=width, anchor=tk.CENTER)
        self.cart_tree.pack(expand=True, fill=tk.BOTH)

        self.total_cart_price_label = ttk.Label(right_panel, text="Cart Total: €0.00", font=("Arial", 14, "bold"))
        self.total_cart_price_label.pack(pady=5, anchor=tk.E)
        
        cart_buttons_frame = ttk.Frame(right_panel)
        cart_buttons_frame.pack(fill=tk.X, pady=5)

        self.clear_cart_button = ttk.Button(cart_buttons_frame, text="Clear Cart", command=self._clear_cart)
        self.clear_cart_button.pack(side=tk.LEFT, expand=True, padx=(0,2))
        
        self.checkout_button = ttk.Button(cart_buttons_frame, text="Checkout", command=self._checkout)
        self.checkout_button.pack(side=tk.RIGHT, expand=True, padx=(2,0))

        self._update_texts() # Initial text setup

    def _update_camera_feed(self):
        if self.camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_captured_frame_for_processing = frame.copy() 
                # self.last_captured_frame_for_saving = frame.copy() # REMOVE THIS LINE
                
                # Display frame (resized for UI)
                frame_rgb = cv2.cvtColor(self.last_captured_frame_for_processing, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Calculate aspect ratio for display
                # Use the fixed camera display dimensions for thumbnailing
                img.thumbnail((CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT), Image.Resampling.LANCZOS)
                
                img_tk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = img_tk
                self.camera_label.configure(image=img_tk)
        
        self.after(30, self._update_camera_feed) # Update ~30 FPS

    def _scan_item_and_get_weight(self):
        if self.last_captured_frame_for_processing is None or self.model is None:
            messagebox.showerror(get_translation("error_title"), get_translation("camera_error_text"))
            return
        
        # Capture the frame for saving at the moment of scanning
        self.last_captured_frame_for_saving = self.last_captured_frame_for_processing.copy()

        frame_to_process = self.last_captured_frame_for_processing
        cart_thumbnail_img = self._create_cart_thumbnail(self.last_captured_frame_for_saving)
        
        # Preprocess for model
        img_resized = cv2.resize(frame_to_process, IMAGE_SIZE_TUPLE, interpolation=cv2.INTER_NEAREST)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Predict
        try:
            self.raw_predictions = self.model.predict(img_batch, verbose=0)[0] # Store raw predictions
            top_index = np.argmax(self.raw_predictions)
            confidence = self.raw_predictions[top_index] * 100
            predicted_class_name_detailed = self.class_names[top_index]
        except Exception as e:
            messagebox.showerror(get_translation("prediction_error_title"), get_translation("prediction_error_text_template", error=e))
            self.current_item_details = None
            self.raw_predictions = None
            self.add_to_cart_button.config(state=tk.DISABLED)
            self.correct_button.config(state=tk.DISABLED)
            return

        base_name = get_base_product_name(predicted_class_name_detailed)
        product_info = APP_PRODUCT_DATA.get(base_name)

        if not product_info:
            messagebox.showwarning(get_translation("unknown_item_warning_title"), get_translation("unknown_item_warning_text_template", item=base_name))
            self.current_item_details = None
            self.raw_predictions = None
            self._clear_current_item_display()
            return

        # Simulate weight using a normal distribution for more realistic variation
        avg_w = product_info["avg_weight"]
        min_w = product_info["typical_min_weight"]
        max_w = product_info["typical_max_weight"]

        mu = avg_w
        if max_w > min_w:
            sigma = (max_w - min_w) / 4.0 # Assume range is roughly 4 standard deviations
        elif avg_w > 0: # If range is 0, but avg is not, use a small percentage of avg as sigma
            sigma = avg_w * 0.05 
        else: # If range is 0 and avg is 0, sigma is 0
            sigma = 0.0
        
        if sigma > 0:
            sim_weight = random.normalvariate(mu, sigma)
            # Clamp the weight to be within the defined min/max typical weights
            sim_weight = max(min_w, min(sim_weight, max_w))
            # Ensure weight is not negative (especially if min_w could be 0 or very low)
            sim_weight = max(0.001, sim_weight) # Ensure a very small positive weight if it would be zero or less
        elif avg_w > 0: # If sigma ended up 0 but avg_w is positive, use avg_w (or min_w/max_w if they are same)
            sim_weight = avg_w
        else: # If all are zero (e.g. for "Unknown" item)
            sim_weight = 0.0
            
        unit_price = product_info["price_per_kg"]
        item_total = sim_weight * unit_price

        self.current_item_details = {
            "name": base_name,
            "detailed_name": predicted_class_name_detailed,
            "confidence": confidence,
            "weight": sim_weight,
            "unit_price": unit_price,
            "item_total": item_total,
            "thumbnail": cart_thumbnail_img # Store the PhotoImage for the cart
        }
        # Mark as not corrected initially when scanned
        self.current_item_details["_is_corrected"] = False

        # Update UI labels for current item
        self._update_current_item_display_text() # Use the new method for text updates
        
        self.add_to_cart_button.config(state=tk.NORMAL)
        self.correct_button.config(state=tk.NORMAL) # Enable correction button

    def _clear_current_item_display(self):
        self.item_name_label.config(text=get_translation("item_label") + "--")
        self.item_confidence_label.config(text=get_translation("confidence_label") + "--")
        self.item_weight_label.config(text=get_translation("weight_label") + "--")
        self.item_unit_price_label.config(text=get_translation("unit_price_label") + "--")
        self.item_total_price_label.config(text=get_translation("item_total_label") + "--")
        self.current_item_details = None
        self.raw_predictions = None
        self.add_to_cart_button.config(state=tk.DISABLED)
        self.correct_button.config(state=tk.DISABLED)

    def _add_to_cart(self):
        if self.current_item_details:
            self.shopping_cart.append(self.current_item_details)
            if self.current_item_details.get("thumbnail"):
                 self.cart_item_photoimages.append(self.current_item_details["thumbnail"]) # Keep reference
            self._update_cart_display()
            self._clear_current_item_display()
        else:
            messagebox.showwarning(get_translation("no_item_message_title"), get_translation("no_item_message_text"))

    def _update_cart_display(self):
        # Clear existing items
        for i in self.cart_tree.get_children():
            self.cart_tree.delete(i)
        
        cart_total = 0.0
        for item_idx, item in enumerate(self.shopping_cart):
            thumbnail_obj = item.get("thumbnail")
            # Ensure the image object is valid, otherwise, don't pass image to Treeview
            image_param = thumbnail_obj if thumbnail_obj else '' 

            self.cart_tree.insert("", tk.END, image=image_param, values=(
                item["name"],
                f"{item['weight']:.3f}",
                f"{item['unit_price']:.2f}",
                f"{item['item_total']:.2f}"
            ))
            cart_total += item["item_total"]
        
        self.total_cart_price_label.config(text=get_translation("cart_total_label") + f"€{cart_total:.2f}")

    def _clear_cart(self):
        if messagebox.askyesno(get_translation("confirm_clear_cart_title"), get_translation("confirm_clear_cart_text")):
            self.shopping_cart = []
            self.cart_item_photoimages = [] # Clear stored photoimages
            self._update_cart_display()
            self._clear_current_item_display()

    def _checkout(self):
        if not self.shopping_cart:
            messagebox.showinfo(get_translation("empty_cart_title"), get_translation("empty_cart_text"))
            return
        
        cart_total = sum(item['item_total'] for item in self.shopping_cart)
        messagebox.showinfo(get_translation("checkout_message_title"), get_translation("checkout_message_text_template", total=cart_total))
        # Here you would typically clear the cart or integrate with a payment system
        self.shopping_cart = []
        self._update_cart_display()
        self._clear_current_item_display()

    def _show_correction_dialog(self):
        if self.raw_predictions is None or self.last_captured_frame_for_saving is None:
            messagebox.showwarning(get_translation("no_item_message_title"), get_translation("no_item_message_text")) # Reused "No Item" as it fits
            return
        # Pass self (main_app) to the dialog so it can call back _handle_correction
        dialog = CorrectionDialog(self, self.class_names, self.raw_predictions, self)
        # The dialog will be modal (grab_set) and handle its own lifecycle.

    def _handle_correction(self, corrected_detailed_class_name):
        if self.last_captured_frame_for_saving is None:
            messagebox.showerror(get_translation("error_title"), get_translation("original_frame_missing_error"))
            return

        print(f"Handling correction: New class is '{corrected_detailed_class_name}'")
        save_image_with_label(self.last_captured_frame_for_saving, 
                              corrected_detailed_class_name, 
                              USER_CORRECTED_DATA_DIR, 
                              IMAGE_SIZE_TUPLE) # Save the image resized to model's input size

        # Update current item details based on the corrected label
        base_name = get_base_product_name(corrected_detailed_class_name)
        product_info = APP_PRODUCT_DATA.get(base_name)

        if not product_info:
            messagebox.showwarning(get_translation("unknown_item_warning_title"), get_translation("corrected_item_unknown_text_template", item=base_name))
            return

        # Handle weight and thumbnail for corrected item
        cart_thumbnail_img = None # Initialize
        if self.current_item_details and 'weight' in self.current_item_details:
            sim_weight = self.current_item_details['weight'] # Keep existing weight
            cart_thumbnail_img = self.current_item_details.get('thumbnail') # Keep existing thumbnail
            if cart_thumbnail_img is None and self.last_captured_frame_for_saving is not None: # If no old thumb, make one
                 cart_thumbnail_img = self._create_cart_thumbnail(self.last_captured_frame_for_saving)
        else: # Fallback if no prior weight, re-simulate with normal distribution
            avg_w = product_info["avg_weight"]
            min_w = product_info["typical_min_weight"]
            max_w = product_info["typical_max_weight"]
            mu = avg_w
            if max_w > min_w:
                sigma = (max_w - min_w) / 4.0
            elif avg_w > 0:
                sigma = avg_w * 0.05
            else:
                sigma = 0.0
            
            if sigma > 0:
                sim_weight = random.normalvariate(mu, sigma)
                sim_weight = max(min_w, min(sim_weight, max_w))
                sim_weight = max(0.001, sim_weight) 
            elif avg_w > 0:
                sim_weight = avg_w
            else:
                sim_weight = 0.0
            # Create thumbnail as new weight is generated
            if self.last_captured_frame_for_saving is not None:
                cart_thumbnail_img = self._create_cart_thumbnail(self.last_captured_frame_for_saving)
        
        unit_price = product_info["price_per_kg"]
        item_total = sim_weight * unit_price
        # The confidence for a user-corrected item is effectively 100% from user's perspective
        corrected_confidence = 100.0 

        self.current_item_details = {
            "name": base_name,
            "detailed_name": corrected_detailed_class_name,
            "confidence": corrected_confidence, # User corrected
            "weight": sim_weight,
            "unit_price": unit_price,
            "item_total": item_total,
            "thumbnail": cart_thumbnail_img # Store/update the PhotoImage
        }
        self.current_item_details["_is_corrected"] = True # Mark as corrected

        # Update UI labels for current item
        self._update_current_item_display_text() # Use the new method for text updates
        
        self.add_to_cart_button.config(state=tk.NORMAL)
        # self.correct_button might be left enabled or disabled depending on desired flow after correction.
        # For now, leave it enabled if user wants to correct again.

    def on_closing(self):
        self.camera_active = False
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

    def _toggle_language(self):
        if translations.CURRENT_LANG == "en":
            new_lang = "es"
        else:
            new_lang = "en"
        
        if set_language(new_lang):
            self._update_texts()
            # Update dialogs too if they are open or need dynamic text updates
            # This might require passing the main_app reference to dialogs to call an update method, or re-initializing them

    def _update_texts(self):
        # Update all UI elements that have translatable text
        self.title(get_translation("title"))
        self.logo_placeholder.config(text=get_translation("logo_placeholder"))
        # Update language button text to show the target language name
        self.language_button.config(text=get_translation("switch_to_language"))

        self.scan_button.config(text=get_translation("scan_button"))
        # Update item_info_frame label text
        if hasattr(self, 'item_info_frame_widget'): # Check if widget exists
            self.item_info_frame_widget.config(text=get_translation("current_item_details_label"))
        # self.item_name_label and other item details are updated dynamically in _scan_item_and_get_weight or _handle_correction
        # However, their base text (e.g., "Item: ") needs to be handled if we reconstruct the full string.
        # For simplicity, we will assume dynamic updates will re-translate. If not, those also need explicit updates here.

        self.add_to_cart_button.config(text=get_translation("add_to_cart_button"))
        self.correct_button.config(text=get_translation("correct_button"))
        
        # Update cart_frame label text
        if hasattr(self.cart_tree, 'master') and isinstance(self.cart_tree.master, ttk.LabelFrame):
            self.cart_tree.master.config(text=get_translation("shopping_cart_label"))
        
        # Update cart tree column headers
        self.cart_tree.heading("#0", text=get_translation("col_image"))
        cols_keys = { # Mapping display name (or original key) to translation key
            "Item Name": "col_item_name",
            "Weight (kg)": "col_weight_kg",
            "Unit Price (€)": "col_unit_price_eur",
            "Total (€)": "col_total_eur"
        }
        for original_header in self.cart_tree["columns"]:
            translation_key = cols_keys.get(original_header, original_header) # Fallback to original if no key
            self.cart_tree.heading(original_header, text=get_translation(translation_key))

        # Preserve value in total_cart_price_label
        current_total_text = self.total_cart_price_label.cget("text")
        currency_symbol_and_value = ""
        if '€' in current_total_text:
            currency_symbol_and_value = '€' + current_total_text.split('€')[-1]
        elif '$' in current_total_text: # Example for other currencies if ever needed
            currency_symbol_and_value = '$' + current_total_text.split('$')[-1]
        
        self.total_cart_price_label.config(text=get_translation("cart_total_label") + currency_symbol_and_value)

        self.clear_cart_button.config(text=get_translation("clear_cart_button"))
        self.checkout_button.config(text=get_translation("checkout_button"))
        
        # Update dynamic labels if an item is currently displayed (important after language switch)
        if self.current_item_details:
            self._update_current_item_display_text() # New method to only update text parts
        else:
            self._clear_current_item_display() # This should use translations for default state

    def _update_current_item_display_text(self):
        if not self.current_item_details: return

        name_display = self.current_item_details["name"]
        detailed_name = self.current_item_details.get("detailed_name", "")

        if detailed_name and detailed_name != name_display:
            name_display += f" ({detailed_name})"
        
        # Check if the item was marked as corrected. 
        is_corrected_flag = self.current_item_details.get("_is_corrected", False)

        if is_corrected_flag:
            name_display += get_translation("item_corrected_suffix")

        self.item_name_label.config(text=get_translation("item_label") + name_display)
        
        confidence_value = self.current_item_details['confidence']
        confidence_text = f"{confidence_value:.2f}%"
        if is_corrected_flag:
            confidence_text += get_translation("confidence_user_suffix")
        self.item_confidence_label.config(text=get_translation("confidence_label") + confidence_text)

        self.item_weight_label.config(text=get_translation("weight_label") + f"{self.current_item_details['weight']:.3f}")
        self.item_unit_price_label.config(text=get_translation("unit_price_label") + f"{self.current_item_details['unit_price']:.2f}")
        self.item_total_price_label.config(text=get_translation("item_total_label") + f"{self.current_item_details['item_total']:.2f}")

    def _create_cart_thumbnail(self, frame, size=CART_THUMBNAIL_SIZE): # Use from config
        if frame is None: return None
        try:
            thumbnail = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(thumbnail_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            return img_tk
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None

# Correction Dialog Class (New)
class CorrectionDialog(tk.Toplevel):
    def __init__(self, parent, all_class_names, raw_preds, main_app_ref):
        super().__init__(parent)
        self.all_class_names = all_class_names
        self.raw_predictions = raw_preds
        self.main_app = main_app_ref # Reference to the main SmartScaleApp instance

        self.title(get_translation("correction_dialog_title"))
        self.geometry("500x600") # Could also be moved to config if desired
        self.transient(parent) # Dialog stays on top of parent
        self.grab_set() # Modal behavior

        self._init_correction_ui()
        # Add language support to dialog if needed by translating title, button texts etc.
        # For now, keeping it simple. The main app's _get_translation would be passed or used.

    def _init_correction_ui(self):
        main_dialog_frame = ttk.Frame(self, padding="10")
        main_dialog_frame.pack(expand=True, fill=tk.BOTH)

        # Top N Predictions Display
        top_n_frame = ttk.LabelFrame(main_dialog_frame, text=get_translation("top_5_preds_label"), padding="10")
        top_n_frame.pack(pady=5, fill=tk.X)

        top_indices = np.argsort(self.raw_predictions)[-TOP_N_PREDICTIONS_DISPLAY:][::-1]
        
        for i, class_idx in enumerate(top_indices):
            class_name = self.all_class_names[class_idx]
            confidence = self.raw_predictions[class_idx] * 100
            btn_text = f"{i+1}: {class_name} ({confidence:.2f}%)"
            # lambda needs to capture class_name correctly for each button
            action = lambda cn=class_name: self._user_selected_class(cn)
            ttk.Button(top_n_frame, text=btn_text, command=action).pack(fill=tk.X, pady=2)

        # Search Section
        search_frame = ttk.LabelFrame(main_dialog_frame, text=get_translation("search_label"), padding="10")
        search_frame.pack(pady=10, fill=tk.X)

        search_entry_frame = ttk.Frame(search_frame)
        search_entry_frame.pack(fill=tk.X)
        ttk.Label(search_entry_frame, text=get_translation("query_label")).pack(side=tk.LEFT, padx=(0,5))
        self.search_query_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_entry_frame, textvariable=self.search_query_var)
        self.search_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        self.search_entry.bind("<Return>", self._perform_search)
        ttk.Button(search_entry_frame, text=get_translation("search_button"), command=self._perform_search).pack(side=tk.LEFT)
        
        self.search_results_listbox = tk.Listbox(search_frame, height=10)
        self.search_results_listbox.pack(pady=5, expand=True, fill=tk.BOTH)
        self.search_results_listbox.bind("<<ListboxSelect>>", self._on_search_result_select)

        # Footer buttons
        footer_frame = ttk.Frame(main_dialog_frame)
        footer_frame.pack(fill=tk.X, pady=10)
        # Potentially translate "Cancel"
        ttk.Button(footer_frame, text=get_translation("cancel_button"), command=self.destroy).pack(side=tk.RIGHT)
    
    def _perform_search(self, event=None): # event is passed by bind
        query = self.search_query_var.get().lower()
        self.search_results_listbox.delete(0, tk.END)
        if not query:
            return
        
        results_count = 0
        for class_name in self.all_class_names:
            if query in class_name.lower():
                self.search_results_listbox.insert(tk.END, class_name)
                results_count += 1
                if results_count >= TOP_N_PREDICTIONS_DISPLAY: # Corrected to use TOP_N_PREDICTIONS_DISPLAY from config for search results as well, or use MAX_SEARCH_RESULTS_DISPLAY if it was intended for this listbox
                    # Assuming MAX_SEARCH_RESULTS_DISPLAY is for this listbox
                    if results_count >= MAX_SEARCH_RESULTS_DISPLAY: 
                         self.search_results_listbox.insert(tk.END, f"...and more (max {MAX_SEARCH_RESULTS_DISPLAY} shown)")
                         break
        if results_count == 0:
            self.search_results_listbox.insert(tk.END, get_translation("no_results_search")) # Potentially translate

    def _on_search_result_select(self, event=None):
        widget = event.widget
        selection = widget.curselection()
        if selection:
            index = selection[0]
            selected_class_name = widget.get(index)
            if not selected_class_name.startswith("...") and not selected_class_name.startswith("No matching") :
                 # Confirm selection
                # Potentially translate message and title
                confirm_text = get_translation("confirm_correction_text_template", item=selected_class_name)
                if messagebox.askyesno(get_translation("confirm_correction_title"), confirm_text, parent=self):
                    self._user_selected_class(selected_class_name)

    def _user_selected_class(self, class_name):
        print(f"User selected '{class_name}' as correct.")
        self.main_app._handle_correction(class_name) # Call the main app's handler
        self.destroy() # Close the dialog

def load_tf_model(model_path):
    try:
        # Check if model_path is relative, if so, make it relative to script directory
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, model_path)

        if not os.path.exists(model_path):
            messagebox.showerror(get_translation("model_load_error_title"), get_translation("model_not_found_error_template", path=model_path))
            return None
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        # Perform a dummy prediction to ensure model is fully loaded/compiled (optional)
        dummy_input = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        print("Model warmed up.")
        return model
    except Exception as e:
        messagebox.showerror(get_translation("model_load_error_title"), get_translation("model_load_error_text_template", error=e))
        return None

if __name__ == "__main__":
    if NUM_CLASSES == 0:
        messagebox.showerror(get_translation("startup_error_title"), get_translation("startup_error_text"))
    else:
        print(f"Loaded {NUM_CLASSES} class names. First few: {CLASS_NAMES[:5]}")
        # Use current_model_path_to_use which includes CLI override logic
        model_to_load = load_tf_model(current_model_path_to_use)

        if model_to_load:
            app = SmartScaleApp(model_to_load, CLASS_NAMES)
            app.protocol("WM_DELETE_WINDOW", app.on_closing)
            app.mainloop()
        else:
            print("Failed to load the model. Application cannot start.")
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(get_translation("app_start_error_title"), get_translation("app_start_error_text"))
            root.destroy() 
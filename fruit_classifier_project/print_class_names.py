import os
import sys

# --- Argument Parsing (adapted from interactive_fruit_classifier.py) ---
# Defaults
DATASET_FOLDER_ARG = 'fruits-360-original-size' # Subfolder within the archive name
DATASET_ARCHIVE_NAME_ARG = 'fruits-360_original-size'
# Unused arguments from original script, kept for structural similarity during copy if needed later
# IMG_WIDTH_ARG = 224
# IMG_HEIGHT_ARG = 224
# MODEL_PATH_ARG = None

print("Parsing arguments for dataset selection...")
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "res" and i + 1 < len(sys.argv):
        if sys.argv[i+1] == "100":
            # IMG_WIDTH_ARG = 100
            # IMG_HEIGHT_ARG = 100
            DATASET_FOLDER_ARG = 'fruits-360'
            DATASET_ARCHIVE_NAME_ARG = 'fruits-360_100x100'
            print("Configuration: Using 100x100 resolution dataset for class names.")
        else:
            print(f"Warning: Unrecognized value '{sys.argv[i+1]}' for 'res' argument. Using default (original-size) dataset for class names.")
        i += 2
    # Silently ignore other arguments like "model" as they are not relevant for this script
    # elif arg == "model" and i + 1 < len(sys.argv):
    #     i += 2 # Consume the argument and its value
    else:
        if arg not in ["model"]: # Avoid warning for common but unused args from the other script
            print(f"Warning: Unrecognized argument or missing value ignored: {arg}")
        i += 1

if "res" not in sys.argv:
    print("Configuration: No 'res' argument provided. Using default (original-size) dataset for class names.")


# --- Setup Directories (adapted from interactive_fruit_classifier.py) ---
# Assumes this script is in the same directory as the dataset archives
# or that the dataset archives are in a path relative to this script's location.
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load Class Names (adapted from interactive_fruit_classifier.py) ---
# Construct the path to the 'Training' directory based on parsed arguments
dataset_train_path = os.path.join(base_dir, DATASET_ARCHIVE_NAME_ARG, DATASET_FOLDER_ARG, 'Training')

class_names = []
print(f"\nAttempting to load class names from: {dataset_train_path}")
try:
    if not os.path.isdir(dataset_train_path):
        # Try to give a more helpful message if the base archive directory is missing
        base_archive_path = os.path.join(base_dir, DATASET_ARCHIVE_NAME_ARG)
        if not os.path.isdir(base_archive_path):
            raise FileNotFoundError(f"Base dataset archive directory not found: {base_archive_path}. Please ensure '{DATASET_ARCHIVE_NAME_ARG}' exists in the same directory as this script, or adjust paths.")
        raise FileNotFoundError(f"Training directory for class names not found: {dataset_train_path}. Check if '{DATASET_FOLDER_ARG}/Training' exists within '{base_archive_path}'.")

    loaded_class_names = sorted(os.listdir(dataset_train_path))
    # Filter to include only directories, as these represent class names
    class_names = [name for name in loaded_class_names if os.path.isdir(os.path.join(dataset_train_path, name))]

    if not class_names:
        raise ValueError(f"No class subdirectories found in {dataset_train_path}. The directory is empty or contains no subdirectories.")

    NUM_CLASSES = len(class_names)
    print(f"Successfully found {NUM_CLASSES} class names from '{DATASET_ARCHIVE_NAME_ARG}/{DATASET_FOLDER_ARG}'.\n")

    print("--- Class Names ---")
    for idx, name in enumerate(class_names):
        print(f"{idx + 1}: {name}")

except FileNotFoundError as fnfe:
    print(f"Error: {fnfe}")
    print("Please ensure the dataset archives ('fruits-360_original-size' and/or 'fruits-360_100x100')")
    print("are unzipped in the same directory as this script, or that the paths are correctly specified.")
    print("For example, if script is in 'fruit_classifier_project', archives should be 'fruit_classifier_project/fruits-360_original-size/', etc.")
    print("You can specify the 100x100 dataset by running: python print_class_names.py res 100")
except ValueError as ve:
    print(f"Error: {ve}")
    print(f"Please check the structure of your dataset directory: {dataset_train_path}")
except Exception as e:
    print(f"An unexpected error occurred while loading class names: {e}")
    print("Please check your dataset path and structure.")
    exit()

if not class_names:
    print("\nNo class names were loaded. Please review error messages above and check your dataset path and structure.") 
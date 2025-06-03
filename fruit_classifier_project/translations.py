import tkinter as tk

# --- Language Configuration ---
LANGUAGES = {
    "en": {
        "title": "Smart Fruit Scale",
        "scan_button": "Scan Item & Get Weight",
        "current_item_details_label": "Current Item Details",
        "item_label": "Item: ",
        "confidence_label": "Confidence: ",
        "weight_label": "Weight (kg): ",
        "unit_price_label": "Unit Price (€/kg): ",
        "item_total_label": "Item Total (€): ",
        "add_to_cart_button": "Add to Cart",
        "correct_button": "Correct / Alternatives",
        "shopping_cart_label": "Shopping Cart",
        "cart_total_label": "Cart Total: ",
        "clear_cart_button": "Clear Cart",
        "checkout_button": "Checkout",
        "no_item_message_title": "No Item",
        "no_item_message_text": "No item scanned to add to cart.",
        "confirm_clear_cart_title": "Confirm",
        "confirm_clear_cart_text": "Are you sure you want to clear the cart?",
        "empty_cart_title": "Empty Cart",
        "empty_cart_text": "Shopping cart is empty.",
        "checkout_message_title": "Checkout",
        "checkout_message_text_template": "Proceeding to checkout.\nTotal amount: €{total:.2f}\nThank you for shopping!",
        "correction_dialog_title": "Correct Classification",
        "top_5_preds_label": "Top 5 Predictions",
        "search_label": "Search for Correct Label",
        "query_label": "Query:",
        "search_button": "Search",
        "cancel_button": "Cancel",
        "no_results_search": "No matching classes found.",
        "confirm_correction_title": "Confirm Correction",
        "confirm_correction_text_template": "Set '{item}' as the correct label?",
        "error_title": "Error",
        "camera_error_text": "Camera frame not available or model not loaded.",
        "prediction_error_text_template": "Failed to predict: {error}",
        "unknown_item_warning_title": "Unknown Item",
        "unknown_item_warning_text_template": "'{item}' not found in product database.",
        "corrected_item_unknown_text_template": "Corrected item '{item}' not in product database. Cannot update price.",
        "original_frame_missing_error": "Original frame for saving is missing.",
        "model_load_error_title": "Model Load Error",
        "model_load_error_text_template": "Error loading model: {error}",
        "startup_error_title": "Startup Error",
        "startup_error_text": "Class names could not be loaded. Please check CLASS_NAMES_RAW.",
        "app_start_error_title": "Application Start Error",
        "app_start_error_text": "Failed to load the TensorFlow model. The application cannot start. Please check console for details and ensure the model file is correctly specified and accessible.",
        "model_not_found_error_template": "Model file not found at: {path}",
        "logo_placeholder": "[Company Logo]",
        "language_button_en": "Español",
        "language_button_es": "English",
        "switch_to_language": "Español",
        "item_corrected_suffix": " [Corrected]",
        "confidence_user_suffix": " (User)",
        "col_image": "Image",
        "col_item_name": "Item Name",
        "col_weight_kg": "Weight (kg)",
        "col_unit_price_eur": "Unit Price (€)",
        "col_total_eur": "Total (€)",
    },
    "es": {
        "title": "Báscula Inteligente de Frutas",
        "scan_button": "Escanear Artículo y Pesar",
        "current_item_details_label": "Detalles del Artículo Actual",
        "item_label": "Artículo: ",
        "confidence_label": "Confianza: ",
        "weight_label": "Peso (kg): ",
        "unit_price_label": "Precio Unit. (€/kg): ",
        "item_total_label": "Total Artículo (€): ",
        "add_to_cart_button": "Añadir al Carrito",
        "correct_button": "Corregir / Alternativas",
        "shopping_cart_label": "Carrito de Compras",
        "cart_total_label": "Total Carrito: ",
        "clear_cart_button": "Vaciar Carrito",
        "checkout_button": "Pagar",
        "no_item_message_title": "Sin Artículo",
        "no_item_message_text": "No hay ningún artículo escaneado para añadir al carrito.",
        "confirm_clear_cart_title": "Confirmar",
        "confirm_clear_cart_text": "¿Seguro que quieres vaciar el carrito?",
        "empty_cart_title": "Carrito Vacío",
        "empty_cart_text": "El carrito de compras está vacío.",
        "checkout_message_title": "Pagar",
        "checkout_message_text_template": "Procediendo al pago.\nImporte total: €{total:.2f}\n¡Gracias por su compra!",
        "correction_dialog_title": "Corregir Clasificación",
        "top_5_preds_label": "Top 5 Predicciones",
        "search_label": "Buscar Etiqueta Correcta",
        "query_label": "Consulta:",
        "search_button": "Buscar",
        "cancel_button": "Cancelar",
        "no_results_search": "No se encontraron clases coincidentes.",
        "confirm_correction_title": "Confirmar Corrección",
        "confirm_correction_text_template": "¿Establecer '{item}' como la etiqueta correcta?",
        "error_title": "Error",
        "camera_error_text": "Imagen de cámara no disponible o modelo no cargado.",
        "prediction_error_text_template": "Fallo al predecir: {error}",
        "unknown_item_warning_title": "Artículo Desconocido",
        "unknown_item_warning_text_template": "'{item}' no encontrado en la base de datos de productos.",
        "corrected_item_unknown_text_template": "Artículo corregido '{item}' no está en la base de datos. No se puede actualizar el precio.",
        "original_frame_missing_error": "Falta el marco original para guardar.",
        "model_load_error_title": "Error al Cargar Modelo",
        "model_load_error_text_template": "Error cargando el modelo: {error}",
        "startup_error_title": "Error de Inicio",
        "startup_error_text": "No se pudieron cargar los nombres de las clases. Revisa CLASS_NAMES_RAW.",
        "app_start_error_title": "Error al Iniciar Aplicación",
        "app_start_error_text": "Fallo al cargar el modelo de TensorFlow. La aplicación no puede iniciar. Revisa la consola y asegúrate que el modelo esté bien especificado y accesible.",
        "model_not_found_error_template": "Archivo de modelo no encontrado en: {path}",
        "logo_placeholder": "[Logo de Empresa]",
        "language_button_en": "Español",
        "language_button_es": "English",
        "switch_to_language": "English",
        "item_corrected_suffix": " [Corregido]",
        "confidence_user_suffix": " (Usuario)",
        "col_image": "Imagen",
        "col_item_name": "Nombre Artículo",
        "col_weight_kg": "Peso (kg)",
        "col_unit_price_eur": "Precio Unit. (€)",
        "col_total_eur": "Total (€)",
    }
}

CURRENT_LANG = "en" # Default language

def get_translation(key, **kwargs):
    # Fallback to English if a key is missing in the current language
    translation = LANGUAGES.get(CURRENT_LANG, LANGUAGES["en"]).get(key, LANGUAGES["en"].get(key, key))
    if kwargs:
        try:
            return translation.format(**kwargs)
        except KeyError:
            # Fallback if formatting fails (e.g. template expects a var not provided)
            return LANGUAGES["en"].get(key, key).format(**kwargs) if LANGUAGES["en"].get(key, key) else key
    return translation

def set_language(lang_code):
    global CURRENT_LANG
    if lang_code in LANGUAGES:
        CURRENT_LANG = lang_code
        print(f"Language changed to {CURRENT_LANG}")
        return True
    return False 
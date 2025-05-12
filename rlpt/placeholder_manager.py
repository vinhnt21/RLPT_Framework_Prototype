import json
import os
from typing import Dict, Tuple, Optional
from config import PLACEHOLDER_MAPPING_FILE, PII_ENTITY_TYPES

class PlaceholderManager:
    """
    Manages the placeholder_mapping.json file.
    Handles loading, saving, and generating new placeholders for PII.
    Ensures that placeholders are unique and consistently mapped.
    """
    def __init__(self, mapping_file_path: str = PLACEHOLDER_MAPPING_FILE):
        self.mapping_file_path = mapping_file_path
        self.placeholder_map = self._load_mapping()

    def _load_mapping(self) -> Dict:
        """Loads the placeholder mapping from the JSON file."""
        if os.path.exists(self.mapping_file_path):
            try:
                with open(self.mapping_file_path, 'r') as f:
                    # Ensure all PII_ENTITY_TYPES are present as top-level keys
                    loaded_map = json.load(f)
                    for pii_type in PII_ENTITY_TYPES:
                        if pii_type not in loaded_map:
                            loaded_map[pii_type] = {}
                    return loaded_map
            except json.JSONDecodeError:
                print(f"Warning: {self.mapping_file_path} is corrupted. Initializing with an empty map.")
                return {pii_type: {} for pii_type in PII_ENTITY_TYPES} # Initialize all types
        else:
            # Create the file with the basic structure if it doesn't exist
            initial_map = {pii_type: {} for pii_type in PII_ENTITY_TYPES}
            self._save_mapping(initial_map)
            return initial_map

    def _save_mapping(self, mapping_data: Optional[Dict] = None):
        """Saves the current placeholder mapping to the JSON file."""
        data_to_save = mapping_data if mapping_data is not None else self.placeholder_map
        with open(self.mapping_file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    def get_or_create_placeholder(self, pii_value: str, pii_type: str) -> str:
        """
        Gets an existing placeholder for a PII value or creates a new one.
        Args:
            pii_value (str): The actual PII string (e.g., "John Doe").
            pii_type (str): The type of PII (e.g., "PERSON_NAME").

        Returns:
            str: The placeholder tag (e.g., "<PERSON_NAME_1>").
        """
        if pii_type not in self.placeholder_map:
            self.placeholder_map[pii_type] = {} # Initialize if type is new

        # Check if PII value already has a placeholder
        for placeholder, value in self.placeholder_map[pii_type].items():
            if value == pii_value:
                return placeholder

        # Create a new placeholder
        new_index = len(self.placeholder_map[pii_type]) + 1
        new_placeholder = f"<{pii_type.upper()}_{new_index}>"
        self.placeholder_map[pii_type][new_placeholder] = pii_value
        self._save_mapping()
        return new_placeholder

    def get_original_value(self, placeholder: str) -> Optional[str]:
        """
        Retrieves the original PII value for a given placeholder.
        Args:
            placeholder (str): The placeholder tag (e.g., "<PERSON_NAME_1>").

        Returns:
            Optional[str]: The original PII value, or None if not found.
        """
        # Extract PII type from placeholder, e.g., "<PERSON_NAME_1>" -> "PERSON_NAME"
        try:
            pii_type_from_placeholder = placeholder.strip("<>").split('_')[0] # More robust extraction
            if pii_type_from_placeholder in self.placeholder_map and \
               placeholder in self.placeholder_map[pii_type_from_placeholder]:
                return self.placeholder_map[pii_type_from_placeholder][placeholder]
        except IndexError:
            print(f"Warning: Could not parse PII type from placeholder '{placeholder}'")

        # Fallback: search across all types if direct lookup fails (less efficient)
        for pii_type in self.placeholder_map:
            if placeholder in self.placeholder_map[pii_type]:
                return self.placeholder_map[pii_type][placeholder]
        return None

    def get_all_mappings(self) -> Dict:
        """Returns the entire placeholder map."""
        return self.placeholder_map 
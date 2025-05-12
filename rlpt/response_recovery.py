import re
from typing import Dict, Optional
from rlpt.placeholder_manager import PlaceholderManager

class ResponseRecoverer:
    """
    Recovers original PII in a text that contains placeholders.
    Uses the PlaceholderManager to look up original values.
    """
    def __init__(self, placeholder_manager: PlaceholderManager):
        self.placeholder_manager = placeholder_manager

    def recover_text(self, text_with_placeholders: str, specific_mappings: Optional[Dict[str,str]] = None) -> str:
        """
        Replaces placeholders in the text with their original PII values.
        Args:
            text_with_placeholders (str): The text containing placeholders.
            specific_mappings (Optional[Dict[str,str]]): If provided, uses these mappings first.
                                                        This is useful if the sanitization step returned a
                                                        context-specific map. Falls back to global manager.
        Returns:
            str: The text with original PII values restored.
        """
        recovered_text = text_with_placeholders

        # Regex to find all placeholders like <TYPE_INDEX> e.g. <PERSON_NAME_1>, <EMAIL_23>
        # This regex is more general to catch various placeholder formats.
        placeholder_pattern = re.compile(r"<([A-Z_]+_\d+)>")
        found_placeholders = set(placeholder_pattern.findall(text_with_placeholders)) # Get unique placeholder tags like "PERSON_NAME_1"

        for tag_content in found_placeholders:
            placeholder = f"<{tag_content}>" # Reconstruct the full placeholder, e.g., <PERSON_NAME_1>
            original_value = None

            if specific_mappings and placeholder in specific_mappings:
                original_value = specific_mappings[placeholder]

            if original_value is None: # Fallback to global placeholder manager
                original_value = self.placeholder_manager.get_original_value(placeholder)

            if original_value is not None:
                # Replace all occurrences of this placeholder
                recovered_text = recovered_text.replace(placeholder, original_value)
            else:
                print(f"Warning: No original value found for placeholder '{placeholder}'. It will remain in the text.")

        return recovered_text 
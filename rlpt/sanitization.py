import json
from typing import Dict, List, Tuple
from langchain_core.language_models.llms import BaseLLM # For type hinting
from rlpt.prompts import ner_prompt_template
from rlpt.placeholder_manager import PlaceholderManager
from config import PII_ENTITY_TYPES # Import from config

class Sanitizer:
    """
    Handles the detection and sanitization of PII in text.
    Uses a local LLM for NER and the PlaceholderManager to replace PII with placeholders.
    """
    def __init__(self, local_llm: BaseLLM, placeholder_manager: PlaceholderManager):
        """
        Initializes the Sanitizer.
        Args:
            local_llm: An instance of a Langchain compatible local LLM (e.g., Ollama).
            placeholder_manager: An instance of PlaceholderManager.
        """
        self.local_llm = local_llm
        self.placeholder_manager = placeholder_manager

    def _detect_pii_with_llm(self, text: str) -> Dict[str, List[str]]:
        """
        Uses the local LLM to detect PII entities in the text.
        Args:
            text (str): The input text to scan for PII.
        Returns:
            Dict[str, List[str]]: A dictionary where keys are PII types
                                  (e.g., "PERSON_NAME") and values are lists of
                                  detected PII strings.
        """
        formatted_prompt = ner_prompt_template.format_messages(text_input=text)
        try:
            # Assuming local_llm.invoke returns a AIMessage or similar with 'content'
            response = self.local_llm.invoke(formatted_prompt)

            # Extract content based on Langchain version/model
            if hasattr(response, 'content'):
                llm_output_str = response.content
            elif isinstance(response, str):
                llm_output_str = response
            else:
                print(f"Unexpected LLM response type: {type(response)}")
                return {}

            # Clean the output if it contains markdown code block markers
            llm_output_str = llm_output_str.strip()
            if llm_output_str.startswith("```json"):
                llm_output_str = llm_output_str[7:]
            if llm_output_str.endswith("```"):
                llm_output_str = llm_output_str[:-3]
            llm_output_str = llm_output_str.strip()

            detected_pii = json.loads(llm_output_str)

            # Validate structure and filter for known PII_ENTITY_TYPES
            validated_pii = {}
            if isinstance(detected_pii, dict):
                for pii_type, values in detected_pii.items():
                    if pii_type in PII_ENTITY_TYPES and isinstance(values, list):
                        # Ensure values are strings
                        validated_pii[pii_type] = [str(v) for v in values if isinstance(v, (str, int, float))]
            return validated_pii
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM NER output: {e}")
            print(f"LLM Raw Output: '{llm_output_str}'")
            return {}
        except Exception as e:
            print(f"Error during PII detection with LLM: {e}")
            return {}

    def sanitize_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Sanitizes the input text by replacing detected PII with placeholders.
        Args:
            text (str): The input text.
        Returns:
            Tuple[str, Dict[str, str]]:
                - The sanitized text with PII replaced by placeholders.
                - A dictionary mapping placeholders used in this text to their original values.
        """
        detected_pii_groups = self._detect_pii_with_llm(text)
        sanitized_text = text
        current_placeholders_map = {} # Tracks placeholders specific to this sanitization pass

        if not detected_pii_groups:
            return sanitized_text, current_placeholders_map

        # Sort PII by length (longest first) to avoid issues with substrings
        # e.g. "John Doe" and "Doe" -> replace "John Doe" first
        all_pii_items = []
        for pii_type, pii_values in detected_pii_groups.items():
            for pii_value in pii_values:
                all_pii_items.append((pii_type, pii_value))

        # Sort by length of PII value, descending
        all_pii_items.sort(key=lambda item: len(item[1]), reverse=True)

        for pii_type, pii_value in all_pii_items:
            if pii_value in sanitized_text: # Check if the PII is still in the text (might have been part of a longer PII)
                placeholder = self.placeholder_manager.get_or_create_placeholder(pii_value, pii_type)
                # Replace all occurrences of this specific PII value
                sanitized_text = sanitized_text.replace(pii_value, placeholder)
                current_placeholders_map[placeholder] = pii_value

        return sanitized_text, current_placeholders_map 
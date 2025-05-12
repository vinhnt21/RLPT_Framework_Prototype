from langchain_core.prompts import ChatPromptTemplate
# PII_ENTITY_TYPES = ["PERSON_NAME", "EMAIL", "PHONE_NUMBER", "ADDRESS", "URL_PERSONAL", "ID_NUM", "USERNAME"]


PROMPT_NER_SYSTEM = """You are a PII detection tool. Extract ONLY actual personal identifiers from text, not generic terms. Identify:
- PERSON_NAME: Real full/partial names (e.g., "John Smith", "Dr. Lee")
- EMAIL: Valid email addresses only (e.g., "user@example.com")
- PHONE_NUMBER: Actual phone numbers with digits (e.g., "+1-555-0100")
- ADDRESS: Specific locations with numbers/streets (e.g., "123 Main St")
- URL_PERSONAL: Personal website URLs (e.g., "https://johnsmith.me")
- ID_NUM: Identification numbers (e.g., "SSN123-45-6789")
- USERNAME: Distinct user handles (e.g., "@johndoe")

Return ONLY a valid JSON object with entity types as keys and arrays of found strings as values. If nothing found, return {{}}. No explanations."""

PROMPT_NER_USER_TEMPLATE = """Extract actual PII from: {text_input}"""

# Prompt for local LLM to perform NER
# This prompt guides the LLM to identify PII and return it in a structured JSON format.
# The system message sets the role and expected output format.
# The user message template will take the text input.
ner_prompt_template = ChatPromptTemplate.from_messages([
    ("system", PROMPT_NER_SYSTEM),
    ("human", PROMPT_NER_USER_TEMPLATE)
])

# Example usage (for testing):
# formatted_prompt = ner_prompt_template.format_messages(text_input="My name is John Doe and my email is john.doe@example.com.")
# print(formatted_prompt) 
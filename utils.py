import streamlit as st
import json
import os
import io
import sys
import anthropic
import time
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
import concurrent.futures

# Add OpenAI import
try:
    import openai
except ImportError:
    pass

# Load environment variables from .env file in local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Global client variables for lazy loading
_anthropic_client = None
_openai_client = None

# Enhanced API key handling with multiple fallback mechanisms
def get_api_key(provider="anthropic"):
    """
    Get the API key from various sources in the following priority order:
    1. Environment variable
    2. Streamlit secrets (for deployment)
    3. Session state (if manually entered in UI)
    
    Args:
        provider (str): The API provider ("anthropic" or "openai")
    """
    env_var_names = {
        "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "openai": ["OPENAI_API_KEY"]
    }
    
    session_state_key = {
        "anthropic": "user_anthropic_api_key",
        "openai": "user_openai_api_key"
    }
    
    # Check environment variables first (loaded from .env file in local development)
    for env_var in env_var_names.get(provider, []):
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    
    # Check Streamlit secrets (for deployment)
    try:
        # Use the in operator to check if the key exists
        if provider in st.secrets:
            if "api_key" in st.secrets[provider]:
                api_key = st.secrets[provider]["api_key"]
                if api_key:
                    return api_key
    except:
        # Silently continue if secrets aren't available
        pass
    
    # Check session state (if user manually entered it)
    state_key = session_state_key.get(provider)
    if state_key in st.session_state and st.session_state[state_key]:
        return st.session_state[state_key]
    
    # No API key found
    return None

# Initialize Anthropic client with lazy loading
def initialize_anthropic_client():
    global _anthropic_client
    
    if _anthropic_client is not None:
        return _anthropic_client
        
    api_key = get_api_key("anthropic")
    if api_key:
        try:
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
            return _anthropic_client
        except Exception as e:
            st.error(f"Error initializing Anthropic client: {str(e)}")
            return None
    return None

# Initialize OpenAI client with lazy loading
def initialize_openai_client():
    global _openai_client
    
    if _openai_client is not None:
        return _openai_client
        
    api_key = get_api_key("openai")
    if api_key:
        try:
            import openai
            _openai_client = openai.OpenAI(api_key=api_key)
            # Test the client with a simple request
            test_response = _openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=5
            )
            # If we get here, the client is working
            return _openai_client
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    return None

def check_password():
    """
    Returns `True` if the user had the correct password.
    """
    # Load password from environment variables or secrets
    correct_password = None
    
    # Try from environment variables first
    correct_password = os.getenv("QA_PASSWORD")
    
    # If not found in environment, try getting from secrets
    if not correct_password:
        try:
            # Access secrets in a way that won't trigger unnecessary warnings
            if "QA_PASSWORD" in st.secrets:
                correct_password = st.secrets["QA_PASSWORD"]
        except:
            pass
    
    if not correct_password:
        st.error("No password set. Configure QA_PASSWORD in environment variables or Streamlit secrets.")
        return False
    
    # Create a session state object if it doesn't exist yet
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False
    
    if "password_attempts" not in st.session_state:
        st.session_state.password_attempts = 0

    if st.session_state.password_entered:
        return True

    # Show input for password
    st.markdown("# Customer Support QA Engine Login")
    password = st.text_input("Enter the application password", type="password")
    
    if password:
        if password == correct_password:
            st.session_state.password_entered = True
            st.rerun()
            return True
        else:
            st.session_state.password_attempts += 1
            if st.session_state.password_attempts >= 3:
                st.error("Too many incorrect password attempts. Please try again later.")
                time.sleep(3)  # Add a delay to discourage brute force attempts
            else:
                st.error(f"Incorrect password. Attempts: {st.session_state.password_attempts}/3")
            return False
    else:
        return False
    
# Helper function to parse JSON from API responses with better error handling
def parse_json_response(response_text):
    """Parse JSON from API responses with better error handling"""
    try:
        # Try direct JSON parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try one more approach - look for ```json blocks
                import re
                json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_blocks:
                    try:
                        return json.loads(json_blocks[0])
                    except json.JSONDecodeError:
                        st.error("Failed to extract valid JSON from API response")
                        st.code(response_text, language="text")
                        st.write("Raw response from API:")
                        st.write(response_text)
                        return None
                else:
                    st.error("Failed to extract valid JSON from API response")
                    st.code(response_text, language="text")
                    st.write("Raw response from API:")
                    st.write(response_text)
                    return None
        else:
            st.error("Could not find JSON structure in API response")
            st.write("Raw response from API:")
            st.write(response_text)
            return None
        
def detect_language_from_sample(text_sample):
    """
    Directly detect language from a text sample based on character presence.
    This function prioritizes detecting specific languages with unique scripts.
    """
    if not text_sample or len(text_sample.strip()) < 10:
        return "unknown", "Unknown"
    
    # Count characters in each script range
    text_len = len(text_sample.strip())
    
    # Thai detection (ก-๛)
    thai_chars = sum(1 for char in text_sample if '\u0E00' <= char <= '\u0E7F')
    if thai_chars > 10:  # If there are at least 10 Thai characters
        return "th", "Thai"
    
    # Chinese detection
    chinese_chars = sum(1 for char in text_sample if 
                        ('\u4E00' <= char <= '\u9FFF') or ('\u3400' <= char <= '\u4DBF'))
    if chinese_chars > 10:  # If there are at least 10 Chinese characters
        return "zh", "Chinese"
    
    # Vietnamese detection - look for specific Vietnamese diacritics
    vietnamese_diacritics = sum(1 for char in text_sample if 
                               char in 'ăâêôơưđ' or char in 'ĂÂÊÔƠƯĐ' or
                               '\u0300' <= char <= '\u036F')  # Combining diacritical marks
    if vietnamese_diacritics > 10:
        return "vi", "Vietnamese"
    
    # Default to English for everything else
    return "en", "English"

# Then modify the detect_language_cached function to use this direct detection:

def detect_language_cached(text_sample, provider="anthropic"):
    """Cached version of language detection to improve performance"""
    if not text_sample:
        return "unknown", "Unknown"
    
    # Create a hash of the text sample for caching
    import hashlib
    text_hash = hashlib.md5(text_sample.encode('utf-8')).hexdigest()[:8]
    cache_key = f"lang_detect_{text_hash}"
    
    # Check if we have a cached result
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Use our direct detection logic instead of API-based detection
    result = detect_language_from_sample(text_sample)
    st.session_state[cache_key] = result
    return result

def detect_language(text, model_provider="anthropic"):
    try:
        # Use the model provider chosen in the sidebar if available
        if "model_provider" in st.session_state:
            if st.session_state.model_provider == "OpenAI GPT":
                model_provider = "openai"
            else:
                model_provider = "anthropic"
                
        # Check if API key is available for selected provider
        if model_provider == "anthropic":
            client = initialize_anthropic_client()
            if not client:
                st.warning("Anthropic API key is required for language detection. Trying other providers...")
                model_provider = "openai"
                client = initialize_openai_client()
                if not client:
                    st.warning("No API keys available for language detection. Using English as default.")
                    return "en", "English (default)"
        elif model_provider == "openai":
            client = initialize_openai_client()
            if not client:
                st.warning("OpenAI API key is required for language detection. Trying other providers...")
                model_provider = "anthropic"
                client = initialize_anthropic_client()
                if not client:
                    st.warning("No API keys available for language detection. Using English as default.")
                    return "en", "English (default)"
        
        # Use selected provider to detect language
        if model_provider == "anthropic":
            response = initialize_anthropic_client().messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=10,
                system="You are a language detection tool. Respond with only the ISO language code.",
                messages=[
                    {"role": "user", "content": f"Detect the language of this text and respond with only the ISO language code (e.g., 'en', 'zh-CN', 'zh-TW', 'vi', 'th', 'pt-ES', 'pt-BR'): {text[:200]}..."}
                ]
            )
            lang_code = response.content[0].text.strip().lower()
        
        elif model_provider == "openai":
            response = initialize_openai_client().chat.completions.create(
                model="gpt-4o",
                max_tokens=10,
                messages=[
                    {"role": "system", "content": "You are a language detection tool. Respond with only the ISO language code."},
                    {"role": "user", "content": f"Detect the language of this text and respond with only the ISO language code (e.g., 'en', 'zh-CN', 'zh-TW', 'vi', 'th', 'pt-ES', 'pt-BR'): {text[:200]}..."}
                ]
            )
            lang_code = response.choices[0].message.content.strip().lower()
        
        # Map language codes to full names
        lang_map = {
            "en": "English",
            "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "vi": "Vietnamese",
            "th": "Thai",
            "pt-es": "Portuguese (Spain)",
            "pt-br": "Portuguese (Brazil)",
            "es": "Spanish"
        }
        
        return lang_code, lang_map.get(lang_code, "Unknown")
    
    except Exception as e:
        st.warning(f"Could not detect language automatically: {str(e)}")
        return "en", "English (default)"

# Function to load the prompt template from a markdown file
def load_prompt_template(file_path="QA_prompt.md"):
    """
    Load the QA prompt template from a markdown file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        st.error(f"Prompt template file '{file_path}' not found. This file is required.")
        st.stop()

# Function to load evaluation rules from JSON file
@st.cache_data
def load_evaluation_rules(file_path="evaluation_rules.json", scoring_path="scoring_system.json"):
    try:
        # Load main evaluation rules
        with open(file_path, "r") as f:
            rules = json.load(f)
        
        # Load scoring system if available
        try:
            with open(scoring_path, "r") as f:
                scoring = json.load(f)
                # Merge scoring system into rules
                rules["scoring_system"] = scoring.get("scoring_system", {})
                
                # Check and standardize scoring scale
                if "score_scale" in rules["scoring_system"]:
                    scale = rules["scoring_system"]["score_scale"]
                    
                    # If scale is 0-10, normalize quality levels to match
                    if scale["max"] == 10:
                        with st.sidebar.expander("Debug - Score Scale", expanded=False):
                            st.info(f"Detected 0-10 scale in scoring system")
                            
                        # No need to modify - we'll handle the conversion during analysis
                        pass
                
        except FileNotFoundError:
            # Require scoring system file
            st.error(f"Scoring system file '{scoring_path}' not found.")
            st.stop()
            
        return rules
    except FileNotFoundError:
        # Require evaluation rules file
        st.error(f"Evaluation rules file '{file_path}' not found.")
        st.stop()

def convert_score(score, from_scale, to_scale):
    """
    Convert a score from one scale to another
    
    Args:
        score (float): The score to convert
        from_scale (tuple): The source scale as (min, max)
        to_scale (tuple): The target scale as (min, max)
    
    Returns:
        float: The converted score
    """
    from_min, from_max = from_scale
    to_min, to_max = to_scale
    
    # Ensure score is within source scale bounds
    score = max(from_min, min(score, from_max))
    
    # Convert to percentage within source scale
    percentage = (score - from_min) / (from_max - from_min)
    
    # Apply percentage to target scale
    converted = to_min + percentage * (to_max - to_min)
    
    return converted

# Check for required files
def check_required_files(files_dict):
    missing_files = []
    for file_path, file_desc in files_dict.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_desc} ({file_path})")
    
    if missing_files:
        st.error("Required files not found:")
        for missing_file in missing_files:
            st.error(f"- {missing_file}")
        st.info("Please ensure all required files are available in the application directory.")
        st.stop()
    return True

# Add or update custom CSS for styling
def setup_custom_css():
    st.markdown("""
    <style>
        .score-excellent { color: #e0ffe0; } 
        .score-good { color: #cee6ff; }
        .score-needs-improvement { color: #ffd39d; }
        .score-poor { color: #ffc6c6; }
        
        .score-box {
            padding: 20px;
            border-radius: 10px;
            color: 333333;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .score-box-excellent {
            background-color: #e0ffe0; /* Light green */
            border-left: 5px solid #006400;
        }
        
        .score-box-good {
            background-color: #cee6ff; /* Light blue */
            border-left: 5px solid #0000CD;
        }
        
        .score-box-needs-improvement {
            background-color: #ffd39d; /* Light orange */
            border-left: 5px solid #FF8C00;
        }
        
        .score-box-poor {
            background-color: #ffc6c6; /* Light red */
            border-left: 5px solid #8B0000;
        }
        
        /* Progress stages styling */
        .progress-stage {
            padding: 8px 12px;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #f0f0f0;
            font-size: 14px;
        }
        
        .progress-stage-active {
            background-color: #e6f3ff;
            border-left: 3px solid #2196F3;
        }
        
        .progress-stage-completed {
            background-color: #e8f5e9;
            border-left: 3px solid #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)

# Check API connections and set provider
def check_api_connections():
    anthropic_client = None
    openai_client = None
    status_expander = st.sidebar.expander("Status Messages", expanded=False)
    
    # Test Anthropic connection
    anthropic_api_key = get_api_key("anthropic")
    if anthropic_api_key:
        try:
            anthropic_client = initialize_anthropic_client()
            if anthropic_client:
                # Display success in sidebar
                with status_expander:
                    st.success("Successfully connected to Anthropic API")
        except Exception as e:
            with status_expander:
                st.error(f"Error connecting to Anthropic API: {str(e)}")
    
    # Test OpenAI connection
    openai_api_key = get_api_key("openai")
    if openai_api_key:
        try:
            openai_client = initialize_openai_client()
            if openai_client:
                # Display success in sidebar
                with status_expander:
                    st.success("Successfully connected to OpenAI API")
        except Exception as e:
            with status_expander:
                st.error(f"Error connecting to OpenAI API: {str(e)}")
    
    # Get provider choice from session state
    provider_internal_name = None
    model_name = None
    model_provider = None
    
    if "model_provider" in st.session_state:
        if st.session_state.model_provider == "OpenAI GPT":
            provider_internal_name = "openai"
            model_name = "gpt-4o"
            model_provider = "OpenAI GPT"
        else:
            provider_internal_name = "anthropic"
            model_name = "claude-3-7-sonnet-20250219"
            model_provider = "Anthropic Claude"
    else:
        # Default to available providers in order of preference
        if anthropic_client:
            provider_internal_name = "anthropic"
            model_name = "claude-3-7-sonnet-20250219"
            model_provider = "Anthropic Claude"
        elif openai_client:
            provider_internal_name = "openai"
            model_name = "gpt-4o"
            model_provider = "OpenAI GPT"
    
    # Check if any API key is available
    if not anthropic_client and not openai_client:
        st.warning("⚠️ Please provide at least one valid API key to use this application.")
        st.info("Enter your API key in the sidebar.")
        return None, None, None
    
    # If the selected provider's API key is not available, try to switch to an available provider
    if provider_internal_name == "anthropic" and not anthropic_client:
        if openai_client:
            with st.sidebar:
                st.warning("Anthropic API key not available, switching to OpenAI")
            provider_internal_name = "openai"
            model_name = "gpt-4o"
            model_provider = "OpenAI GPT"
    elif provider_internal_name == "openai" and not openai_client:
        if anthropic_client:
            with st.sidebar:
                st.warning("OpenAI API key not available, switching to Anthropic")
            provider_internal_name = "anthropic"
            model_name = "claude-3-7-sonnet-20250219"
            model_provider = "Anthropic Claude"
    
    return provider_internal_name, model_name, model_provider

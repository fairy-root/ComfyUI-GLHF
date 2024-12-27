import subprocess
import sys
from jax import config
import openai
import json
import os
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re
import random

# Get the directory of the current script (glhf.py)
script_directory = os.path.dirname(os.path.abspath(__file__))
custom_instructions_directory = os.path.join(script_directory, 'custom_instructions')

# Construct the full path to config.json
config_path = os.path.join(script_directory, 'config.json')

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

install_and_import("googlesearch")
install_and_import("requests")
install_and_import("bs4")

def configuration():
    api_key = None
    base_url = None
    models = {}
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            api_key = config_data.get('api_key')
            base_url = config_data.get('baseurl')
            models = config_data.get('models', {})
            models = {key: value for key, value in models.items()}
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")

    return api_key, base_url, models

def load_custom_instructions():
    instructions = ["None"]  # Add "None" option
    if not os.path.exists(custom_instructions_directory):
        os.makedirs(custom_instructions_directory)
    for filename in os.listdir(custom_instructions_directory):
        if filename.endswith(".txt"):
            instructions.append(filename[:-4])  # Remove .txt extension
    return instructions

api_key, base_url, models_dict = configuration()
models_list = list(models_dict.keys())
default_model = models_list[0] if models_list else None
custom_instructions_list = load_custom_instructions()

def fetch_and_extract_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract main content - you might need to adjust selectors based on common website structures
        paragraphs = soup.find_all('p')
        text_content = "\n".join([p.text for p in paragraphs])
        return text_content.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing content from {url}: {e}")
        return None

class GlhfChat:
    chat_history = []  # Class-level chat history

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Enter your prompt here...", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "model": (models_list, {"default": default_model}),
                "Console_log": ("BOOLEAN", {"default": True}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "num_search_results": ("INT", {"default": 5, "min": 1, "max": 10}),
                "keep_context": ("BOOLEAN", {"default": True}),
                "custom_instruction": (custom_instructions_list, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "GLHF"

    def execute(self, prompt, seed, model=None, Console_log=False, enable_web_search=False, num_search_results=3, keep_context=True, custom_instruction="None"):
        selected_model_value = models_dict.get(model)
        if not selected_model_value:
            print(f"Error: Model '{model}' not found in config.json. Using default.")
            selected_model_value = list(models_dict.values())[0] if models_dict else None
            if not selected_model_value:
                return ("Error: No valid model found.",)

        if Console_log:
            print(f"GLHF Chat Request: Model='{model}', Prompt='{prompt}', Seed='{seed}', Web Search Enabled={enable_web_search}, Keep Context={keep_context}, Custom Instruction='{custom_instruction}'")

        augmented_prompt = ""  # Initialize as empty
        if enable_web_search:
            # Regex to find URLs in the prompt
            urls_in_prompt = re.findall(r'(https?://\S+)', prompt)

            try:
                search_results_content = []
                searched_urls = set() # To avoid processing same URLs multiple times

                # Process URLs found in the prompt first
                for url in urls_in_prompt:
                    if url not in searched_urls:
                        print(f"Fetching content from URL in prompt: {url}")
                        content = fetch_and_extract_content(url)
                        if content:
                            search_results_content.append(f"Source (from prompt): {url}\nContent:\n{content}\n---\n")
                        else:
                            search_results_content.append(f"Source (from prompt): {url}\nCould not retrieve content.\n---\n")
                        searched_urls.add(url)

                # Process URLs from Google Search
                for i, url in enumerate(search(prompt, num_results=num_search_results)):
                    if url not in searched_urls:
                        print(f"Fetching content from search result {i+1}: {url}")
                        content = fetch_and_extract_content(url)
                        if content:
                            search_results_content.append(f"Search Result {i+1}:\nSource: {url}\nContent:\n{content}\n---\n")
                        else:
                            search_results_content.append(f"Search Result {i+1}:\nSource: {url}\nCould not retrieve content.\n---\n")
                        searched_urls.add(url)

                if search_results_content:
                    augmented_prompt = f"Original query: {prompt}\n\nWeb search results and linked content:\n\n{''.join(search_results_content)}\n\nBased on this information, please provide a response."
                else:
                    augmented_prompt = f"Original query: {prompt}\n\nNo relevant web search results or linked content found. Please proceed with the original query."
                    if Console_log:
                        print("No relevant web search results or linked content found.")

            except Exception as e:
                print(f"Web Search Error: {e}")
                augmented_prompt = f"Original query: {prompt}\n\nAn error occurred during web search or fetching linked content. Please proceed with the original query."

        else:
            augmented_prompt = prompt # If web search is not enabled, use the original prompt

        response = self._glhf_interaction(base_url, api_key, selected_model_value, augmented_prompt, Console_log, keep_context, custom_instruction)
        return response

    def _glhf_interaction(self, base_url, api_key, model_value, prompt, Console_log, keep_context, custom_instruction):
        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

            messages = []
            if custom_instruction != "None":
                instruction_path = os.path.join(custom_instructions_directory, f"{custom_instruction}.txt")
                try:
                    with open(instruction_path, 'r') as f:
                        system_instruction = f.read()
                        messages.append({"role": "system", "content": system_instruction})
                except FileNotFoundError:
                    print(f"Error: Custom instruction file '{instruction_path}' not found.")
                except Exception as e:
                    print(f"Error reading custom instruction file: {e}")

            if keep_context and self.chat_history:
                messages.extend(self.chat_history)
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=model_value,
                messages=messages,
                stream=False
            )
            output_text = completion.choices[0].message.content

            if Console_log:
                print(f"GLHF Chat Response: {output_text}")

            # Update chat history
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": output_text})

            return (output_text,)

        except openai.AuthenticationError as e:
            print(f"GLHF Chat Error: Authentication failed - {e}")
            return (f"Error: Authentication failed - {e}",)
        except openai.APIConnectionError as e:
            print(f"GLHF Chat Error: Could not connect to GLHF API - {e}")
            return (f"Error: Could not connect to GLHF API - {e}",)
        except openai.RateLimitError as e:
            print(f"GLHF Chat Error: API request exceeded rate limit - {e}")
            return (f"Error: API request exceeded rate limit - {e}",)
        except openai.APIStatusError as e:
            print(f"GLHF Chat Error: GLHF API returned an error - {e}")
            return (f"Error: GLHF API returned an error - {e}",)
        except Exception as e:
            print(f"GLHF Chat Error: An unexpected error occurred - {e}")
            return (f"Error: An unexpected error occurred - {e}",)

# Node export details
NODE_CLASS_MAPPINGS = {
    "glhf_chat": GlhfChat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "glhf_chat": "GLHF Chat with Advanced Web and Link Search"
}
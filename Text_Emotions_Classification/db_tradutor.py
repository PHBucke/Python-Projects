from deep_translator import GoogleTranslator
import os  # Import the os module for file system operations

# Define source and target languages
source_lang = 'english'
target_lang = 'spanish'

# Define the base path (replace with your actual base path)
base_path = "C:/Users/pedro/OneDrive/Software Development/Portfólio/Python Projects/Text_Emotions_Classification/"

# Get a list of all files in the base path with the .txt extension
filenames = [f for f in os.listdir(base_path) if f.endswith(".txt")]

# Create a GoogleTranslator object
translator = GoogleTranslator(source=source_lang, target=target_lang)

# Loop through each filename
for filename in filenames:
  # Construct the full path
  full_path = os.path.join(base_path, filename)  # Use os.path.join for better path handling

  # Translate the file
  try:
    translated_text = translator.translate_file(full_path)
    print(f"Successfully translated {filename}.")
  except Exception as e:
    print(f"Error translating {filename}: {e}")

  # You can optionally write the translated text to a new file here

print("Finished translating all files.")

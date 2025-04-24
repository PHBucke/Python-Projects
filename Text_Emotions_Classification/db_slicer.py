# Define the number of lines per chunk and file path variable
chunk_size = 40
file_path = ("C:/Users/pedro/OneDrive/"
             "Software Development/Portfólio/"
             "Python Projects/"
             "Text_Emotions_Classification/Train.txt")  # Replace with your actual file path

# Open the original file in read mode
with open(file_path, "r") as f:
    # Read the data into a list (modify for your data format)
    data = f.readlines()

# Calculate the total number of lines in the original file
total_lines = len(data)

# Create an empty list to store the data chunks
data_chunks = []

# Iterate over the data in chunks
for i in range(0, total_lines, chunk_size):
    # Get the current chunk of data
    chunk = data[i:i + chunk_size]

    # Append the chunk to the list
    data_chunks.append(chunk)

# Define a counter for the file names
file_counter = 0

# Write each chunk to a separate file
for chunk in data_chunks:
    # Create the filename
    filename = f"train_dataset_{file_counter}.txt"

    # Open the file in write mode
    with open(filename, "w") as f:
        # Write the data to the file (modify for your data format)
        for line in chunk:
            f.write(line)

    # Increment the counter
    file_counter += 1

print(f"Successfully sliced the data into {file_counter} files.")

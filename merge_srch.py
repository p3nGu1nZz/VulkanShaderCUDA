import os

# Define the output file name
output_file = "mergedforllmhfiles.txt"

# Get all .h files in the /h/ directory
h_files = [os.path.join("src", file) for file in os.listdir("src") if file.endswith(".h")]

if not h_files:
    print("No .h files found in the /h/ directory.")
    exit()

print(f"Found {len(h_files)} .h files. Merging...")

try:
    # Open the output file for writing
    with open(output_file, "w") as outfile:
        for file in h_files:
            print(f"Adding contents of {file}...")
            outfile.write(f"// {file}\n")  # Vulkan-style filename comment
            with open(file, "r") as infile:
                outfile.write(infile.read())
                outfile.write("\n")  # Ensure separation between files

    print(f"All files merged into {output_file}.")
except Exception as e:
    print(f"An error occurred: {e}")
import os

# Define the output file name
output_file = "mergedforllmcomps.txt"

# Get all .comp files in the /comp/ directory
comp_files = [os.path.join("comp", file) for file in os.listdir("comp") if file.endswith(".comp")]

if not comp_files:
    print("No .comp files found in the /comp/ directory.")
    exit()

print(f"Found {len(comp_files)} .comp files. Merging...")

try:
    # Open the output file for writing
    with open(output_file, "w") as outfile:
        for file in comp_files:
            print(f"Adding contents of {file}...")
            outfile.write(f"// {file}\n")  # Vulkan-style filename comment
            with open(file, "r") as infile:
                outfile.write(infile.read())
                outfile.write("\n")  # Ensure separation between files

    print(f"All files merged into {output_file}.")
except Exception as e:
    print(f"An error occurred: {e}")
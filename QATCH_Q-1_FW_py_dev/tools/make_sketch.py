import os
import shutil

# THIS TOOL CREATES AN ARDUINO SKETCH FOLDER FROM THE PLATFORMIO FOLDER STRUCTURE
# Created by Alexander J Ross, Copyright 2023 QATCH Technologies LLC

files_to_move = []
folders_to_check = []

current_dir = os.getcwd()
path_parts = os.path.split(current_dir)
if path_parts[1] == "tools":
    project_directory = path_parts[0]
else:
    project_directory = current_dir
print("project dir: ", project_directory)

sketch_name = os.path.split(project_directory)[1]
sketch_name = sketch_name[0:sketch_name.rindex('_')]

for item in os.listdir(project_directory):
    #print (f'item is: {item}')
    if os.path.isdir(os.path.join(project_directory, item)):
        #print (f'folder is: {item}')
        if item.startswith('.') or item in ["build", "sketch", "test", "tools"]:
            continue # ignore these folders, not part of sketch
        folders_to_check.append(os.path.join(project_directory, item))

for search_folder in folders_to_check:
    #print (f'Checking folder \"{search_folder}\"...')
    files_to_move += [os.path.join(dp, f) for dp, dn, fn in os.walk(search_folder) for f in fn]

search_str = "#define CODE_VERSION"
fw_version = "dev"
for file in files_to_move:
    if not (file.endswith(".h") or file.endswith(".cpp")):
        continue # only process .h and .cpp files
    # print("Procesing: ", file)
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find(search_str) >= 0:
                fw_version = line[len(search_str):].strip().replace('"', '')
                print("FW Version: ", fw_version)
sketch_name += f"_{fw_version}"

print(f'files to move: {len(files_to_move)}\n - ' + '\n - '.join(files_to_move)) 
print()
print("sketch name:", sketch_name)
print()
input("PRESS 'ENTER' TO MAKE SKETCH...")
print()

sketch_dir = os.path.join(project_directory, "sketch", sketch_name)
os.makedirs(sketch_dir, exist_ok=True)
sketch_name += ".ino"

for file in files_to_move:
    path_parts = os.path.split(file)
    if path_parts[1] != "README": # skip 'README' files (not including 'README.md')
        print (f'Creating file \"{path_parts[1]}\"...')
        shutil.copy(file, sketch_dir)
    else:
        print( f'Skipping file \"{path_parts[1]}\"...')
    if path_parts[0].endswith('src') and path_parts[1].startswith('QATCH'): # rename project file
        print (f'Renaming file \"{path_parts[1]}\" to \"{sketch_name}\"...')
        os.replace(os.path.join(sketch_dir, path_parts[1]), os.path.join(sketch_dir, sketch_name))

print()
print ("Finished creating sketch. Open Arduino sketch here:")
print (os.path.join(sketch_dir, sketch_name))
print()
input("Press any key to close...")

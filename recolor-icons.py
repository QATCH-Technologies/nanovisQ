import os
import re

DIRECTORY_PATH = "QATCH/icons"
TARGET_COLOR = "#333333"


def process_color_match(match):
    attribute = match.group(1)
    current_value = match.group(2)

    if current_value.lower() == "none":
        return match.group(0)

    return f'{attribute}="{TARGET_COLOR}"'


def recolor_svgs(directory, target_color):
    pattern = re.compile(r'(fill|stroke)="([^"]+)"', re.IGNORECASE)

    processed_count = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith(".svg"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                svg_content = file.read()
            new_content = pattern.sub(process_color_match, svg_content)
            style_pattern = re.compile(r'(fill|stroke)\s*:\s*([^;"]+)', re.IGNORECASE)

            def style_replace(match):
                attr = match.group(1)
                val = match.group(2).strip()
                if val.lower() == "none":
                    return match.group(0)
                return f"{attr}:{target_color}"

            new_content = style_pattern.sub(style_replace, new_content)
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(new_content)

            processed_count += 1
            print(f"Recolored: {filename}")

    print(f"\n--- Done! Successfully recolored {processed_count} icons to {target_color}. ---")


if __name__ == "__main__":
    if os.path.exists(DIRECTORY_PATH):
        recolor_svgs(DIRECTORY_PATH, TARGET_COLOR)
    else:
        print(f"Error: The directory '{DIRECTORY_PATH}' was not found.")

import os
from PIL import Image #pip install Pillow

input_folder = 're_in'
output_folder = 're_out'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    
    input_img = Image.open(input_path)
    
    # 128x output
    output_img = input_img.resize((255, 255), Image.NEAREST)
    
    # 256x output
    # output_img = input_image.resize((256, 256), Image.NEAREST)
    
    # 512x output
    # output_img = input_image.resize((512, 512), Image.NEAREST)
    
    # 1024x output
    # output_img = input_image.resize((1024, 1024), Image.NEAREST)
    
    output_path = os.path.join(output_folder, filename)
    output_img.save(output_path)

print("and It's done bro!")
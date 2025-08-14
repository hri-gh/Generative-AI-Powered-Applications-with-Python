# v1:Simple Script: Introduction to Generate image captions with the BLIP model (Load a local image and generate a caption using the BLIP model.

import requests
from PIL import Image # Python Imaging Library (PIL, is used to open the image file and convert it into an RGB format)
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Load your image, DON'T FORGET TO WRITE YOUR IMAGE NAME
img_path = "./img3.jpeg"
# convert it into an RGB format
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "the image of"

# The pre-processed image is passed through the processor to generate inputs in the required format.
inputs = processor(images=image, text=text, return_tensors="pt") #The return_tensors argument is set to "pt" to return PyTorch tensors

# print (inputs) # To see the format (inputs dictionary)

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

# print(output) # To check the complete output

# Decode the generated tokens to human-readable text
caption = processor.decode(outputs[0], skip_special_tokens=True) #skip_special_tokens argument is set to True to ignore special tokens in the output text.

# Print the caption
print(caption)

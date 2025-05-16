

import torch
import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
import torchvision.models as models
from PIL import Image
import base64

current_dir = os.path.dirname(os.path.realpath(__file__))

# Define the model architecture 
NUM_FEATURES = 22
model = models.resnet18(pretrained=False)  # Set pretrained=False to start fresh
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=NUM_FEATURES)  

# Load the saved weights into the model
model.load_state_dict(torch.load(os.path.join(current_dir, 'model\\model.pth'), map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

IMAGES_PATH = os.path.join(current_dir, 'data', 'images')
print("Path to dataset files:", IMAGES_PATH)

transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 for ResNet18
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model.to("cpu")
dataset = ImageFolder(IMAGES_PATH, transform=transform)

def get_prediction(model, device, class_to_idx, base64Data):
    # Add padding if necessary
    padding = len(base64Data) % 4
    if padding != 0:
        base64Data += "=" * (4 - padding)

    try:
        # Decode the base64 data and write to a file
        with open('output_image.jpg', 'wb') as fh:
            fh.write(base64.b64decode(base64Data))  # Decode and write to file
    except base64.binascii.Error as e:
        print(f"Error decoding base64 data: {e}")
    try:
        img = Image.open("output_image.jpg")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)

            idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
            predicted_class = idx_to_class[predicted_idx.item()]

        print("result:", predicted_class)
        return f"{str(predicted_class)}"


    except Exception as e:
        print(f"Error processing {img}: {e}")


def run(base64data):
    return get_prediction(model, "cpu", dataset.class_to_idx, base64data)

from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):

    #For testing server connection
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        message = "POST data processed successfully"
        self.wfile.write(bytes("HELLO!!", "utf8"))


    def do_POST(self):
        # Get the length of the content
        content_length = int(self.headers['Content-Length'])
        # Read the POST data
        post_data = self.rfile.read(content_length)
        print(f"succesfully recieved an image! analyzing...")
        #print(f"post data: {post_data}") (Debug)

        message = run(post_data.decode())
        if message is None:
            print("Message is None! Setting it to default message.")
            message = "Error: Image invalid"
            self.send_response(400)
        else:
            self.send_response(200)

        # Respond back to the client
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(message.encode("utf8"))

def run_server(port):
    # Run the server
    with HTTPServer(('', port), Handler) as server:
        print(f"Server running on port {port}...")
        server.serve_forever()


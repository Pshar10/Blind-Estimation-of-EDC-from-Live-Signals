import torch
from torch.utils.tensorboard import SummaryWriter
from model_flex import R2Dnet  # Make sure this import works

"""A script to vivualize the model"""

# def main():
#     # Define the model
#     model = R2Dnet()

#     # Set up the device (GPU or CPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Move the model to the selected device
#     model.to(device)

#     # Create a dummy input tensor appropriate for your model and move to the device
#     # Adjust the size of the dummy input to match your model's input size
#     dummy_input = torch.randn(1, 16384).to(device)

#     # Set up TensorBoard
#     writer = SummaryWriter("runs/model_visualization")

#     # Add the model graph to TensorBoard
#     writer.add_graph(model, dummy_input)

#     # Close the writer
#     writer.close()

#     print("Model graph written to TensorBoard. You can now view it by running 'tensorboard --logdir=runs'")

# if __name__ == "__main__":
#     main()


import torch
import torch.onnx
from model_flex import R2Dnet  # Ensure this is your model class
import webbrowser


# Instantiate your model
model = R2Dnet()

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor appropriate for your model
dummy_input = torch.randn(1, 16384)

# Export the model
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11, input_names=['input'], output_names=['output'])

url = 'https://netron.app/'

# Open URL in a new tab, if a browser window is already open.
webbrowser.open_new_tab(url)


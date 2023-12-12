import argparse
import sys
import torch
import torch.onnx
import torch.nn as nn
import coremltools
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms
from PIL import Image


from segmentation_models_pytorch import DeepLabV3

class SegmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("Segmentation Model Converter")

        self.label = tk.Label(master, text="Choose a PyTorch checkpoint:")
        self.label.pack()

        self.choose_button = tk.Button(master, text="Choose File", command=self.choose_checkpoint)
        self.choose_button.pack()

        self.convert_button = tk.Button(master, text="Convert to Core ML", command=self.convert_to_coreml)
        self.convert_button.pack()

    def choose_checkpoint(self):
        self.checkpoint_path = filedialog.askopenfilename(filetypes=[("PyTorch Checkpoint", "*.pth")])
        print(f"Selected checkpoint: {self.checkpoint_path}")

    def convert_to_coreml(self):
        if not hasattr(self, 'checkpoint_path') or not self.checkpoint_path:
            print("Please choose a PyTorch checkpoint first.")
            return

        
        model = DeepLabV3('resnet50', in_channels=3, classes=21)
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()

       
        dummy_input = torch.randn(1, 3, 224, 224)

        
        traced_model = torch.jit.trace(model, dummy_input)

   
        onnx_path = "segmentation_model.onnx"
        torch.onnx.export(traced_model, dummy_input, onnx_path, verbose=True)

      
        mlmodel = coremltools.converters.onnx.convert(
            model=onnx_path,
            minimum_ios_deployment_target='13'
        )

        # Save the Core ML model
        mlmodel.save('segmentation_model.mlmodel')

        print("Conversion to Core ML completed successfully.")

def main():
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()

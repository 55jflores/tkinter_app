import tkinter as tk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 

import cv2
from PIL import Image

test_transform = transforms.Compose([
    transforms.Resize(100),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
])

device = torch.device("cpu")
loaded_model = torch.jit.load('malaria_torchScript_model.pt').to(device)

class_names = ['Parasitized','Uninfected']

def predict(image):
    # Passing image through model
    pred = loaded_model(image)
    # Grabbing max index to grab correct item from class_names
    myindex = torch.argmax(pred,dim=1).item()
    # Converting to probabilities
    prob_pred = torch.exp(pred)
    answer.config(text=f" {class_names[myindex]}: {100*round(prob_pred[0][0].item(),2) if prob_pred[0][0].item()>prob_pred[0][1].item() else 100*round(prob_pred[0][1].item(),2)}%")

def send_pic():
    # Read in image using opencv. Convert to RGB format. numpy n dimensional array
    image = cv2.imread(entry.get())
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to PIL format. Transform image and reshape for model
    image = Image.fromarray(image)
    image = test_transform(image)
    image = image.view(1,3,100,100)

    predict(image)

def clear_text():
    entry.delete(0,'end')

def create_gui():
    master = tk.Tk()
    button_frame = tk.Frame(master)
    button_frame.pack(side=tk.BOTTOM)

    greeting = tk.Label(text="Malaria or Not")
    greeting.pack()

    text_entry = tk.Label(master, text="Enter image name")
    text_entry.pack()

    global entry 
    entry = tk.Entry()
    entry.pack()

    global answer
    answer = tk.Label(master,text='')
    answer.pack()

    quit_button = tk.Button(button_frame,text='Quit',command=master.quit)
    quit_button.columnconfigure(0)

    clear_button = tk.Button(button_frame,text='Clear Text',command=clear_text)
    clear_button.columnconfigure(0)
    
    show_text = tk.Button(button_frame,text='Detect',command=send_pic)
    show_text.columnconfigure(0)

    quit_button.grid(row=0,column=0)
    clear_button.grid(row=0,column=1)
    show_text.grid(row=0,column=3)

    tk.mainloop()

create_gui()
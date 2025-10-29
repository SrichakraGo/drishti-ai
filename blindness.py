# Simple GUI for Diabetic Retinopathy Detection (no login)

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import main  # imports the main() from your model.py

print('GUI SYSTEM STARTED...')

def OpenFile():
    try:
        # Ask user to select an image
        file_path = askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            messagebox.showinfo("No File Selected", "Please select an image to proceed.")
            return

        print(f"Selected file: {file_path}")

        # Get prediction from model
        value, classes = main(file_path)
        messagebox.showinfo("Prediction Result", f"Predicted Label: {value}\nPredicted Class: {classes}")

        # Display image using matplotlib
        image = Image.open(file_path).convert('RGB')
        plt.imshow(np.array(image))
        plt.title(f'Predicted: {classes} (Label: {value})')
        plt.axis('off')
        plt.show()

        print("Prediction completed successfully!")

    except Exception as error:
        print(f"Error during prediction: {error}")
        messagebox.showerror("Error", f"Something went wrong!\n{error}")

# --- GUI setup ---
root = Tk()
root.geometry('600x300')
root.title("Blindness Detection System")
root.configure(bg='black')

label1 = Label(root, text="Blindness Detection Demo", font=('Arial', 24))
label1.pack(pady=30)

button_upload = Button(root, text="Upload Image & Predict", command=OpenFile)
button_upload.pack(pady=20)

root.mainloop()

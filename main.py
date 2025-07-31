import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from keras.models import load_model
import io

# Load trained model
model = load_model("model.keras")

# GUI window
window = tk.Tk()
window.title("Handwritten Digit Recognizer")
window.geometry("300x300")

# Canvas for drawing
canvas_width = 250
canvas_height = 250
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# PIL image to draw on (for processing)
image1 = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image1)

# Draw on canvas
def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# Clear canvas
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)

# Predict digit
def predict_digit():
    # Resize to 28x28 and invert colors (white on black)
    img = image1.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img).astype("float32")

    # Normalize to [-1, 1]
    img_array = (img_array - 127.5) / 127.5

    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    result_label.config(text=f"Prediction: {digit}")

# Buttons
button_frame = tk.Frame(window)
button_frame.pack()

predict_btn = tk.Button(button_frame, text="Predict", command=predict_digit)
predict_btn.pack(side=tk.LEFT, padx=10)

clear_btn = tk.Button(button_frame, text="Clear", command=clear)
clear_btn.pack(side=tk.LEFT)

# Label to show prediction
result_label = tk.Label(window, text="Draw a digit and click Predict", font=("Helvetica", 16))
result_label.pack()

# Start GUI loop
window.mainloop()

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model("lung_cancer_cnn_model.keras")
class_labels = ['Non-Cancerous', 'Cancerous']


def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    if prediction.shape[-1] == 1:
        prob = float(prediction[0][0])
        if prob >= 0.5:
            return "Non-Cancerous", prob
        else:
            return "Cancerous", 1 - prob
    elif prediction.shape[-1] == 2:
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        return class_labels[class_idx], confidence

    return "Unknown", 0.0


def submit_form():
    name = entry_name.get()
    age = entry_age.get()
    contact = entry_contact.get()

    if not name or not age or not contact or not image_path.get():
        result_label.config(text="\u26a0\ufe0f Please fill all fields and upload image.", fg="red")
        return

    result, conf = predict(image_path.get())
    print(f"DEBUG: Prediction = {result}, Confidence = {conf:.2f}")

    result_frame.pack(pady=10)

    label_patient_name.config(text=f"\ud83d\udc64 Patient Name: {name}")
    label_patient_age.config(text=f"\ud83c\udf82 Age: {age}")
    label_patient_contact.config(text=f"\ud83d\udcde Contact No: {contact}")
    label_result.config(
        text=f"\ud83e\uddec Prediction: {result}",
        fg="green" if result == "Non-Cancerous" else "red",
        font=("Arial", 14, "bold")
    )
    label_confidence.config(
        text=f"\ud83d\udcca Confidence: {conf * 100:.2f}%",
        fg="blue",
        font=("Arial", 12)
    )

    img = Image.open(image_path.get()).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    label_uploaded_image.config(image=img_tk)
    label_uploaded_image.image = img_tk

    entry_name.delete(0, tk.END)
    entry_age.delete(0, tk.END)
    entry_contact.delete(0, tk.END)
    image_path.set("")
    upload_label.config(text="")
    result_label.config(text="\u2705 Diagnosis complete. See report below.", fg="green")

# Image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    image_path.set(file_path)
    upload_label.config(text="\u2705 Image uploaded successfully!", fg="green")

# GUI setup
root = tk.Tk()
root.title("\ud83e\ude7b Lung Cancer CT Scan Detector")
root.geometry("620x800")
root.configure(bg="#f2f2f2")
root.resizable(False, False)

# Title
title = tk.Label(root, text="Lung Cancer CT Scan Detection", font=("Helvetica", 18, "bold"), bg="#f2f2f2", fg="#1f3b4d")
title.pack(pady=10)

# Patient Info Frame
form_frame = tk.LabelFrame(root, text="\ud83d\udc64 Patient Information", padx=15, pady=15, bg="#ffffff", fg="#003366", font=("Arial", 12, "bold"))
form_frame.pack(padx=20, pady=10, fill="x")

form_inner = tk.Frame(form_frame, bg="#ffffff")
form_inner.pack()

# Form fields
tk.Label(form_inner, text="Name:", bg="#ffffff").grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_name = tk.Entry(form_inner, width=30)
entry_name.grid(row=0, column=1, padx=5, pady=5)


tk.Label(form_inner, text="Age:", bg="#ffffff").grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_age = tk.Entry(form_inner, width=30)
entry_age.grid(row=1, column=1, padx=5, pady=5)


tk.Label(form_inner, text="Contact No:", bg="#ffffff").grid(row=2, column=0, sticky="e", padx=5, pady=5)
entry_contact = tk.Entry(form_inner, width=30)
entry_contact.grid(row=2, column=1, padx=5, pady=5)

image_path = tk.StringVar()
btn_upload = tk.Button(form_inner, text="\ud83d\udcc1 Upload CT Scan Image", command=upload_image, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
btn_upload.grid(row=3, columnspan=2, pady=10)

upload_label = tk.Label(form_inner, text="", bg="#ffffff", font=("Arial", 10))
upload_label.grid(row=4, columnspan=2)

btn_submit = tk.Button(form_inner, text="\ud83d\udcdd Submit", command=submit_form, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=20)
btn_submit.grid(row=5, columnspan=2, pady=15)

result_label = tk.Label(root, text="", font=("Arial", 12), bg="#f2f2f2")
result_label.pack()

# Result Frame
result_frame = tk.LabelFrame(root, text="\ud83d\udccb Diagnosis Report", padx=10, pady=10, font=("Arial", 12, "bold"), bg="#e8f4fc", fg="#003366")

label_patient_name = tk.Label(result_frame, text="", font=("Arial", 12), bg="#e8f4fc")
label_patient_name.pack(anchor="center")

label_patient_age = tk.Label(result_frame, text="", font=("Arial", 12), bg="#e8f4fc")
label_patient_age.pack(anchor="center")

label_patient_contact = tk.Label(result_frame, text="", font=("Arial", 12), bg="#e8f4fc")
label_patient_contact.pack(anchor="center")

label_uploaded_image = tk.Label(result_frame, bg="#e8f4fc")
label_uploaded_image.pack(pady=10)

label_result = tk.Label(result_frame, text="", font=("Arial", 14), bg="#e8f4fc")
label_result.pack()

label_confidence = tk.Label(result_frame, text="", font=("Arial", 12), bg="#e8f4fc")
label_confidence.pack()

root.mainloop()

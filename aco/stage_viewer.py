# stage_viewer.py - Tkinter application for showing the algorithm's progress
import pickle
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np

class StageViewer(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Stage viewer")
        self.pack()
        self.create_start_screen()

    def create_start_screen(self):
        self.file_selection = tk.Button(text="Click to select the image pickle",
                                        command=self.sel_callback).pack(fill=tk.X)

    def sel_callback(self):
        file = filedialog.askopenfile(mode="rb")
        images = pickle.load(file)
        self.image_history = images
        self.create_stage_viewer()

    def create_stage_viewer(self):
        self.canvas = tk.Canvas(master=self.master, width=300, height=300)
        self.canvas.pack()
        self.img = ImageTk.PhotoImage(Image.fromarray(255 - (self.image_history[0]*255).astype(np.uint8)))
        self.canvas.create_image(20, 20, anchor=tk.NW, image=self.img)
        self.img_idx = tk.IntVar()
        self.slider = tk.Scale(master=self.master, from_=0, to=len(self.image_history)-1,
                               variable=self.img_idx, command=lambda x: self.refresh_image(x), orient=tk.HORIZONTAL)
        self.slider.pack()
        self.label = tk.Label(master=self.master)
        self.label.pack()

    def refresh_image(self, img_index):
        print(f"Callback value: {img_index}")
        self.img = ImageTk.PhotoImage(Image.fromarray(255 - (self.image_history[int(img_index)]*255).astype(np.uint8)))
        self.label.config(text=f"Current step: {self.img_idx.get()}")
        self.canvas.create_image(20, 20, anchor=tk.NW, image=self.img)


if __name__ == "__main__":
    root = tk.Tk()
    app = StageViewer(master=root)
    app.mainloop()
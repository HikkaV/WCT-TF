from run import *
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

out_f = None


class Gui():
    def __init__(self):
        self.root = Tk()
        self.root.winfo_toplevel().title('Style-transfer')
        self.root.geometry("800x520+300+300")
        self.root.resizable(width=True, height=True)
        menu = Menu(self.root)
        self.root.config(menu=menu)

        filemenu = Menu(menu)
        menu.add_cascade(label="Commands", menu=filemenu)
        filemenu.add_command(label="Load content image", command=self.open_image_content)
        filemenu.add_command(label="Load style image", command=self.open_image_style)
        filemenu.add_command(label="Run programm", command=self.run)
        filemenu.add_separator()
        filemenu.add_command(label="Change learning rate", command=self.change_eta)
        filemenu.add_command(label="Change used layers", command=self.change_layers)
        filemenu.add_command(label="Change epochs", command=self.change_epochs)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=lambda: sys.exit(1))

        self.root.mainloop()

    def define_epochs(self):
        args.passes = int(self.e2.get())

    def change_epochs(self):
        Label(self.root, text="Epochs :").grid(row=1 ,column=2)

        self.e2 = Entry(self.root)
        self.e2.insert(index=20, string='1')
        self.e2.grid(row=1, column=3)
        Button(self.root, text='Define epochs', command=self.define_epochs).grid(row=3, column=2, sticky=W, pady=4)

    def define_layers(self):
        args.relu_targets = []
        args.checkpoints = []

        if self.var5.get() == 1:
            args.relu_targets.append("relu5_1")
            args.checkpoints.append('models/relu5_1')
        if self.var4.get() == 1:
            args.relu_targets.append("relu4_1")
            args.checkpoints.append('models/relu4_1')
        if self.var3.get() == 1:
            args.relu_targets.append("relu3_1")
            args.checkpoints.append('models/relu3_1')
        if self.var2.get() == 1:
            args.relu_targets.append("relu2_1")
            args.checkpoints.append('models/relu2_1')
        if self.var1.get() == 1:
            args.relu_targets.append("relu1_1")
            args.checkpoints.append('models/relu1_1')

    def change_layers(self):
        Label(self.root, text="Used layers :").grid(row=4, sticky=W)
        self.var1 = IntVar()
        Checkbutton(self.root, text="relu1_1", variable=self.var1).grid(row=5, sticky=W)
        self.var2 = IntVar()
        Checkbutton(self.root, text="relu2_1", variable=self.var2).grid(row=6, sticky=W)
        self.var3 = IntVar()
        Checkbutton(self.root, text="relu3_1", variable=self.var3).grid(row=7, sticky=W)
        self.var4 = IntVar()
        Checkbutton(self.root, text="relu4_1", variable=self.var4).grid(row=8, sticky=W)
        self.var5 = IntVar()
        Checkbutton(self.root, text="relu5_1", variable=self.var5).grid(row=9, sticky=W)
        Button(self.root, text='Define layers', command=self.define_layers).grid(row=10, column=1, sticky=W, pady=4)

    def define_eta(self):
        args.alpha = float(self.e1.get())

    def change_eta(self):
        Label(self.root, text="Learning rate :").grid(row=1)

        self.e1 = Entry(self.root)
        self.e1.insert(index=20, string='0.9')
        self.e1.grid(row=1, column=1)
        Button(self.root, text='Define eta', command=self.define_eta).grid(row=3, column=1, sticky=W, pady=4)

    def open_content(self):
        filename = filedialog.askopenfilename(title='open content image',
                                              filetypes=(("jpg files", "*.jpg"),
                                                         ('png files', '*.png'), ('jpeg files', '*.jpeg'),
                                                         ('JPG files', '*.JPG')))
        args.content_path = filename
        return filename

    def open_style(self):
        filename = filedialog.askopenfilename(title='open style image', filetypes=(("jpg files", "*.jpg"),
                                                                                   ('png files', '*.png'),
                                                                                   ('jpeg files', '*.jpeg'),
                                                                                   ('JPG files', '*.JPG')))
        args.style_path = filename
        return filename

    def choose_dir(self):
        filename = filedialog.askdirectory(title='choose directory to save output image')
        args.out_path = filename

    def open_image_content(self):

        x = self.open_content()
        img = Image.open(x)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self.root, image=img, text='Content image', compound=TOP)

        panel.image = img
        panel.grid(row=0, column=0)
        # panel.pack()

    def open_image_style(self):
        x = self.open_style()
        img = Image.open(x)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(image=img, text='Style image', compound=TOP)
        panel.image = img
        panel.grid(row=0, column=1)
        # panel.pack()
        self.choose_dir()

    def run(self):
        global out_f
        main()
        out_f = return_out()
        if out_f is not None:
            img = Image.open(out_f)
            img = img.resize((250, 250), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            panel = Label(image=img, text='Output image', compound=TOP)
            panel.image = img
            panel.grid(row=0, column=2)


if __name__ == '__main__':
    gui = Gui()

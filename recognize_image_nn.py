import tensorflow as tf
from tensorflow import keras
import numpy as np
from tkinter import *
import matplotlib
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from PIL import Image
import pickle

def main ():

    def create_model ():
        data = keras.datasets.fashion_mnist
        (train_images , train_labels) , (test_images , test_labels)  = data.load_data()
        train_images = train_images/255.0
        test_images = test_images/255.0

        model = keras.Sequential([
            keras.layers.Flatten(input_shape = (28, 28)) ,
            keras.layers.Dense (128 , activation = "relu") ,
            keras.layers.Dense (128 , activation= "relu"),
            keras.layers.Dense (10 , activation = "softmax")
        ])

        model.compile (optimizer = "adam" , loss = "sparse_categorical_crossentropy" , metrics=["accuracy"])
        model.fit (train_images , train_labels , epochs = 5)


        return model

    main_ = Tk()
    main_.configure(background = "white")
    main_.geometry("500x500")
    main_.title ("recognize what costume")

    def recognize_image (file_name):
        img = Image.open(file_name).convert ("L")
        img = img.resize ((28 , 28) , Image.ANTIALIAS)
        imgarr = np.asarray(img)
        imgarr = imgarr / 255.0

        model = create_model()

        class_names = ['T-shirt/top', 'Pents', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        arrmg  = []
        arrmg.append(imgarr)
        arrmg = np.asarray (arrmg)
        prediction = model.predict(arrmg)
        print (imgarr[0][0])
        plt.grid(False)
        plt.imshow (img)
        ind = int (np.argmax(prediction))
        plt.title ("prediction : " +  class_names [ind])
        plt.show()
        print (prediction)

    def button_pushed():
        file_name = askopenfilename()
        recognize_image(file_name)

    but = Button (main_, width = 25 , height = 7 , command = button_pushed , text = "load an image" , bg = "grey")
    but.place (x = 100 , y = 100)
    logo = PhotoImage (file = "C:\\Users\\Public\\machine_learning\\Data\\logo_2.png")
    logo_label = Label (main_ , image = logo)
    logo_label.place (x = 100 , y = 300)

    main_.mainloop()

if __name__ == '__main__':
    main()




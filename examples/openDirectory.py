import Tkinter
import tkFileDialog

root = Tkinter.Tk()
dirname = tkFileDialog.askdirectory(parent=root,initialdir="./",title='Please select a directory')

print dirname

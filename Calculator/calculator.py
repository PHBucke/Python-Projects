from tkinter import *

calculator = Tk()
calculator.geometry("354x460")
calculator.title("Calculator")

calculator_label = Label(calculator, text = "Calculator", bg = 'White', font = ("Times", 30, 'bold'))
calculator_label.pack(side=TOP)
calculator_label.config(background='Dark gray')

text_in = StringVar()
operator = ""

def click_buttom(number):
    global operator
    operator = operator + str(number)
    text_in.set(operator)

def equl_buttom():
    global operator
    addition = str(eval(operator))
    text_in.set(addition)
    operator = ''

def equl_buttom():
    global operator
    subtraction = str(eval(operator))
    text_in.set(subtraction)
    operator = ''
    
def equl_buttom():
    global operator
    multiplication = str(eval(operator))
    text_in.set(multiplication)
    operator = ''
    
def clear_buttom():
    text_in.set('')

calculator_text = Entry(calculator, font = ("Courier New",12,'bold'), textvar = text_in, width=25, bd=5, bg='powder blue')
calculator_text.pack()

buttom_1 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(1), text = "1", font = ("Courier New", 16, 'bold'))
buttom_1.place(x = 10, y = 100)

buttom_2 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(2), text = "2", font = ("Courier New", 16, 'bold'))
buttom_2.place(x = 75, y = 100)

buttom_3 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(3), text = "3", font = ("Courier New", 16, 'bold'))
buttom_3.place(x = 140, y = 100)

buttom_4 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(4), text = "4", font = ("Courier New", 16, 'bold'))
buttom_4.place(x = 10, y = 170)

buttom_5 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(5), text = "5", font = ("Courier New", 16, 'bold'))
buttom_5.place(x = 75, y = 170)

buttom_6 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(6), text = "6", font = ("Courier New", 16, 'bold'))
buttom_6.place(x = 140, y = 170)

buttom_7 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(7), text = "7", font = ("Courier New", 16, 'bold'))
buttom_7.place(x = 10, y = 240)

buttom_8 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(8), text = "8", font = ("Courier New", 16, 'bold'))
buttom_8.place(x = 75, y = 240)

buttom_9 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(9), text = "9", font = ("Courier New", 16, 'bold'))
buttom_9.place(x = 140, y = 240)

buttom_0 = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom(0), text = "0", font = ("Courier New", 16, 'bold'))
buttom_0.place(x = 10, y = 310)

buttom_dot = Button(calculator, padx = 47, pady = 14, bd = 4, bg = 'powder blue', command = lambda:click_buttom("."), text = ".", font = ("Courier New", 16, 'bold'))
buttom_dot.place(x = 75, y = 310)

buttom_addiction = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', text = "+", command = lambda:click_buttom("+"), font = ("Courier New", 16, 'bold'))
buttom_addiction.place(x = 205, y = 100)

buttom_subtraction = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', text = "-", command = lambda:click_buttom("-"), font = ("Courier New", 16, 'bold'))
buttom_subtraction.place(x = 205, y = 170)

buttom_multiplication = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', text = "*", command = lambda:click_buttom("*"), font = ("Courier New", 16, 'bold'))
buttom_multiplication.place(x = 205, y = 240)

buttom_division = Button(calculator, padx = 14, pady = 14, bd = 4, bg = 'powder blue', text = "%", command = lambda:click_buttom("/"), font = ("Courier New", 16, 'bold'))
buttom_division.place(x = 205, y = 310)

buttom_clear = Button(calculator, padx = 14, pady = 119, bd = 4, bg = 'powder blue', text = "CE", command = clear_buttom, font = ("Courier New", 16, 'bold'))
buttom_clear.place(x = 270, y = 100)

buttom_equal = Button(calculator, padx = 150, pady = 14, bd = 4, bg = 'powder blue', text = "=", command = equl_buttom, font = ("Courier New", 16, 'bold'))
buttom_equal.place(x = 10, y = 380)

calculator.mainloop()
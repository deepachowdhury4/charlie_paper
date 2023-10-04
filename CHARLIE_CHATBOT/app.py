from tkinter import *

from torch import cudnn_grid_sampler
from chat import get_response, bot_name
import csv
from colorama import Fore, Style

BG_GRAY="#6b6d78" #color of send box #e9ecf2
BG_COLOR="#d8dae3"  #color of text box
TEXT_COLOR="#0a0a0a" #color of text

FONT="Helvetica 16"
FONT_BOLD="Helvetica 18 bold"

class ChatApplication:
    def __init__(self):
        self.window=Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("CHARLIE")
        self.window.resizable(width=True,height=True)
        self.window.configure(width=470, height=550,bg=BG_COLOR)

        #head label 
        head_label=Label(self.window, bg=BG_COLOR, fg="#171e8a",
                            text="Welcome!\nTo learn about CHARLIE, type HELP INTRO.", font="Helvetica 14 bold", pady=10)
        head_label.place(relwidth=1)

        #tiny divider
        line=Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.7, relheight=0.012)

        #text widget stored as instance variable
        self.text_widget = Text(self.window, width=20,height=2,bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, spacing1=6, spacing2=6, padx=5, pady=5)
                                #check pady

        self.text_widget.place(relheight=0.745, relwidth=0.95, rely=0.1)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        #scroll bar
        scrollbar=Scrollbar(self.window)
        scrollbar.place(relheight=1, relx=0.95)
        scrollbar.configure(command=self.text_widget.yview)

        #bottom label
        bottom_label=Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        #message entry box
        self.msg_entry=Entry(bottom_label, bg="#ffffff", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        #send button
        send_button = Button(bottom_label, text="Send", font="Helvetica 16 bold", width=20, bg='#ffffff',
                                command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg,"YOU")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)

        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(cursor="arrow", state=NORMAL, wrap=WORD, spacing3=2)
        self.text_widget.tag_configure('blue',foreground="#508ae6", font='Helvetica 16 bold')
        self.text_widget.insert(END, "YOU: ",'blue')
        self.text_widget.insert(END, msg+'\n')
        self.text_widget.configure(cursor="arrow", state=DISABLED, wrap=WORD)

        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(cursor="arrow", state=NORMAL, wrap=WORD, spacing3=2)
        self.text_widget.tag_configure('blue',foreground="#508ae6", font='Helvetica 16 bold')
        self.text_widget.insert(END, "CHARLIE: ",'blue')
        self.text_widget.insert(END, get_response(msg)+'\n')
        self.text_widget.configure(cursor="arrow", state=DISABLED, wrap=WORD)

        with open('conversation_data.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile)
            if msg == 'quit' or msg == 'bye':
                csvwriter.writerow(['end of conversation'])
            else:
                csvwriter.writerow([msg, get_response(msg)])
            

        self.text_widget.see(END)



if __name__== "__main__":
    app = ChatApplication()
    app.run()


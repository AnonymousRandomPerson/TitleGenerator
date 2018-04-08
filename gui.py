import os

from Tkinter import *
from tkinter.filedialog import *

import title_generator

class Screen:

    def __init__(self):
        self.root = Tk()
        self.root.wm_title('Sci-Ti')

        self.select_frame = Frame(self.root)
        self.select_frame.grid(row=0, column=0)

        self.select_button = Button(self.select_frame, text='Select Input', command=self.open_select)
        self.select_button.grid(row=0, column=0)
        self.input_name = Label(self.select_frame, text='Please select a text file.', fg='red')
        self.input_name.grid(row=1, column=0)

        self.generate_frame = Frame(self.root)
        self.generate_frame.grid(row=1, column=0)

        self.generate_button = Button(self.generate_frame, text='Generate Titles', command=self.generate, state=DISABLED)
        self.generate_button.grid(row=0, column=0)

        self.titles_frame = Frame(self.root)
        self.titles_frame.grid(row=2, column=0)

        self.file_name = ''

        self.title_labels = []

        self.root.mainloop()

    def open_select(self):
        self.file_name = askopenfilename(filetypes=[('Plain text files', '*.txt')])

        if self.file_name:
            self.input_name['text'] = os.path.basename(self.file_name)
            self.input_name['fg'] = 'black'

            self.generate_button['state'] = NORMAL

            self.generate()

    def generate(self):
        titles_ranked = title_generator.generate_titles(self.file_name)

        if not self.title_labels:
            for i in xrange(len(titles_ranked)):
                title_label = Label(self.titles_frame)
                title_label.grid(row = i, column=0, sticky=W)

                score_label = Label(self.titles_frame)
                score_label.grid(row = i, column=1, sticky=W)

                self.title_labels.append(title_label)
                self.title_labels.append(score_label)

        for i, title_ranked in enumerate(titles_ranked):
            title, score = title_ranked
            self.title_labels[i * 2]['text'] = title
            self.title_labels[i * 2 + 1]['text'] = score

if __name__ == '__main__':
    app = Screen()
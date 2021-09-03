from tkinter.ttk import Combobox
from PIL import Image as im
from PIL import ImageTk
import numpy as np
import copy
import random
from matplotlib import colors as col
from matplotlib import pyplot as plt
import webcolors
import math
import pandas as pd
from tkinter import *


img = im.new('RGB', (150, 150))
img_changed = im.new('RGB', (150, 150))
color_list = list(col._colors_full_map.values())


class Board:
    def __init__(self, panel):
        self.canvas = panel
        self.color = None
        self.id = None
        self.grain_state = None
        self.grain_energy = None
        self.density = None
        self.first_grains = []
        self.colors = {}
        self.number = None
        self.img = None
        self.zm_bool = False
        self.ilosc = 0
        self.to_change = None
        self.neighb = None
        self.conditions = None
        self.par_k = None
        self.img1 = None
        self.img_mc = None
        self.img_energy = None
        self.img_drx = None
        self.img_dys = None


    def reset(self, w, h):
        self.img = im.new('RGB', (w, h), (255, 255, 255))
        self.color = np.array(self.img)
        self.id = np.ones((w, h))
        self.id *= -1
        self.grain_state = np.zeros((w, h))
        self.grain_energy = np.zeros((w, h))
        self.density = np.zeros((w, h))
        self.number = None
        self.first_grains = []
        self.colors = {}
        self.ilosc = 0
        self.zm_bool = False
        self.to_change = np.zeros(self.img.size)


    def typ_neigh(self, exType, i, h, j, w):
        if exType == 'moore':
            return True
        if exType == 'von Neumann':
            if (i == h - 1 and j == w - 1) or (i == h + 1 and j == w + 1) or (i == h + 1 and j == w - 1) or (
                    i == h - 1 and j == w + 1):
                return False
            else:
                return True
        if exType == 'pentagonalne lewe':
            if i + 1 == h and j + 1 == w:
                return True
            elif i == h and j + 1 == w:
                return True
            elif i - 1 == h and j + 1 == w:
                return True
            elif i + 1 == h and j == w:
                return True
            elif i - 1 == h and j == w:
                return True
            elif i == h and j == w:
                return True
            else:
                return False
        if exType == 'pentagonalne prawe':
            if i + 1 == h and j - 1 == w:
                return True
            elif i == h and j - 1 == w:
                return True
            elif i - 1 == h and j - 1 == w:
                return True
            elif i + 1 == h and j == w:
                return True
            elif i - 1 == h and j == w:
                return True
            else:
                return False
        if exType == 'heksagonalne lewe':
            if i + 1 == h and j + 1 == w:
                return False
            elif i - 1 == h and j - 1 == w:
                return False
            else:
                return True
        if exType == 'heksagonalne prawe':
            if i + 1 == h and j - 1 == w:
                return False
            elif i - 1 == h and j + 1 == w:
                return False
            else:
                return True
        return False

    def step(self, bc, typ, z, r, q,):
        self.conditions = bc
        self.neighb = typ
        self.to_change = np.ones(self.img.size)
        self.number = q
        if z == 'losowe':
            for i in range(self.number):
                self.colors[i] = webcolors.hex_to_rgb(color_list[i])
                x = random.randint(0, len(self.id) - 1)
                y = random.randint(0, len(self.id[0]) - 1)
                self.first_grains.append((x, y))
        elif z == 'rownomierne':
            w, h = self.img.size
            py = int(math.sqrt(h * q/w))
            px = int(q/py)
            width = int(w/px)
            height = int(h/py)
            #????
        elif z == 'z promieniem':
            for i in range(self.number):
                self.colors[i] = webcolors.hex_to_rgb(color_list[i])
                while True:
                    x = random.randint(0, len(self.id) - 1)
                    y = random.randint(0, len(self.id[0]) - 1)
                    if i == 0:
                        self.first_grains.append((x, y))
                        break
                    k = 0
                    for m, n in self.first_grains:
                        odl = math.sqrt((x - m) ** 2 + (y - n) ** 2)
                        if odl > r:
                            k += 1
                    if k == len(self.first_grains):
                        self.first_grains.append((x, y))
                        break
        a = 0
        for i, j in self.first_grains:
            self.id[i][j] = a
            a += 1
            self.ilosc += 1
        while -1 in self.id:
            for i in range(len(self.id)):
                for j in range(len(self.id[0])):
                    if self.id[i][j] != -1:
                        self.color[i][j] = self.colors[int(self.id[i][j])]
            new_ar = copy.copy(self.id)
            for i in range(len(self.id)):
                for j in range(len(self.id[0])):
                    if self.id[i][j] != -1:
                        for k in range(i - 1, i + 2):
                            for l in range(j - 1, j + 2):
                                if self.typ_neigh(self.neighb, k, i, l, j):
                                    if self.conditions == 'periodyczne':
                                        if k < 0:
                                            k = len(self.color) - 1
                                        if k > len(self.color) - 1:
                                            k = 0
                                        if l < 0:
                                            l = len(self.color[0]) - 1
                                        if l > len(self.color[0]) - 1:
                                            l = 0
                                    if self.conditions == 'absorbujace':
                                        if k < 0:
                                            k = 0
                                        if k > len(self.color) - 1:
                                            k = len(self.color) - 1
                                        if l < 0:
                                            l = 0
                                        if l > len(self.color[0]) - 1:
                                            l = len(self.color[0]) - 1
                                    if self.id[k][l] == -1 and self.to_change[k][l] == 1:
                                        new_ar[k][l] = self.id[i][j]
                                        self.ilosc += 1
                                        self.to_change[k][l] = 0
            self.id = copy.copy(new_ar)
            self.to_change = np.ones(self.img.size)
            self.update_panell(self.color)
        self.img = self.change_picture()

    def update_panell(self, array):
        img_changed = im.fromarray(array)
        image = ImageTk.PhotoImage(img_changed, )
        image_id = self.canvas.create_image(0, 0, anchor=NW, image=image)
        self.canvas.itemconfig(image_id, image=image)
        self.canvas.after(10)
        self.canvas.update()
        self.img1 = img_changed

    def change_picture(self):
        for x in range(len(self.id)):
            for y in range(len(self.id[0])):
                if self.id[x][y] != -1:
                    self.color[x][y] = self.colors[self.id[x][y]]
                else:
                    self.color[x][y] = (0, 0, 0)
        img = im.fromarray(self.color)
        return img


    def mc(self, par_k, iter):
        self.par_k = par_k
        for j in range(iter):
            while 1 in self.to_change and (self.conditions is not None) and (self.neighb is not None):
                x = random.randint(0, self.img.size[0] - 1)
                y = random.randint(0, self.img.size[1] - 1)
                if self.to_change[x][y] == 1:
                    self.to_change[x][y] = 0
                    if self.line(x, y) == False:
                        continue
                    E_before = self.calc(x, y)
                    if E_before == 0:
                        continue
                    neibours_list = self.neigbours(x, y)
                    last_value = self.id[x][y]
                    self.id[x][y] = random.choice(neibours_list)
                    E_after = self.calc(x, y)
                    delta_E = E_after - E_before
                    if delta_E < 0:
                        continue
                    else:
                        p = math.exp(-(delta_E / self.par_k))
                        l = ['ok', 'nie']
                        temp = random.choices(l, [p, 1 - p])
                        if temp[0] == 'nie':
                            self.id[x][y] = last_value
                            self.grain_energy[x][y] = E_before
                        else:
                            self.grain_energy[x][y] = E_after
            self.to_change = np.ones(self.img.size)
        self.img_mc = self.change_picture()


    def line(self, x, y):
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if self.typ_neigh(self.neighb, x, i, y, j):
                    if self.conditions == 'periodyczne':
                        if i < 0:
                            i = len(self.color) - 1
                        if i > len(self.color) - 1:
                            i = 0
                        if j < 0:
                            j = len(self.color[0]) - 1
                        if j > len(self.color[0]) - 1:
                            j = 0
                    if self.conditions == 'absorbujace':
                        if i < 0:
                            i = 0
                        if i > len(self.color) - 1:
                            i = len(self.color) - 1
                        if j < 0:
                            j = 0
                        if j > len(self.color[0]) - 1:
                            j = len(self.color[0]) - 1
                    if self.id[x][y] != self.id[i][j]:
                        return True
        return False

    def calc(self, x, y):
        pom = 0
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if self.typ_neigh(self.neighb, x, i, y, j):
                    if self.conditions == 'periodyczne':
                        if i < 0:
                            i = len(self.color) - 1
                        if i > len(self.color) - 1:
                            i = 0
                        if j < 0:
                            j = len(self.color[0]) - 1
                        if j > len(self.color[0]) - 1:
                            j = 0
                    if self.conditions == 'absorbujace':
                        if i < 0:
                            i = 0
                        if i > len(self.color) - 1:
                            i = len(self.color) - 1
                        if j < 0:
                            j = 0
                        if j > len(self.color[0]) - 1:
                            j = len(self.color[0]) - 1
                    if self.id[x][y] != self.id[i][j]:
                        pom += 1
        return pom

    def neigbours(self, x, y):
        temp = []
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if self.typ_neigh(self.neighb, x, i, y, j):
                    if self.conditions == 'periodyczne':
                        if i < 0:
                            i = len(self.color) - 1
                        if i > len(self.color) - 1:
                            i = 0
                        if j < 0:
                            j = len(self.color[0]) - 1
                        if j > len(self.color[0]) - 1:
                            j = 0
                    if self.conditions == 'absorbujace':
                        if i < 0:
                            i = 0
                        if i > len(self.color) - 1:
                            i = len(self.color) - 1
                        if j < 0:
                            j = 0
                        if j > len(self.color[0]) - 1:
                            j = len(self.color[0]) - 1
                    if self.id[x][y] != self.id[i][j]:
                        if not self.id[x][y] in temp:
                            temp.append(self.id[i][j])
                            self.to_change[i][j] = 0
        return temp


    def show_energy(self):
        if (self.conditions is not None) and (self.neighb is not None):
            self.img_energy = im.new('RGB', self.img.size, (255, 255, 255))
            for x in range(len(self.grain_energy)):
                for y in range(len(self.grain_energy[0])):
                    if self.line(x, y):
                        self.img_energy.putpixel((y, x), (0, 0, 0))
                    else:
                        self.img_energy.putpixel((y, x), (255, 255, 255))

    def step1(self, rozm):
        A = 8.67E+13
        B = 9.41E+00
        t = 0.001
        critical_value = 4.21584E+12 / self.id.shape[0] * self.id.shape[1]
        list_pom = []
        while t <= 0.2:
            ro1 = A / B + (1 - A / B) * math.exp(-B * t)
            ro2 = A / B + (1 - A / B) * math.exp(-B * (t + 0.001))
            list_pom.append((round(t, 5), ro1))
            delta = ro2 - ro1
            wart_sr = delta / self.id.shape[0] * self.id.shape[1]
            self.density = self.density + wart_sr * 0.2
            pack_val = wart_sr * 0.8
            p_line = rozm * 0.8
            p_inside = rozm - p_line
            num_rec = -2
            lines_index = np.transpose((self.grain_energy > 0).nonzero())
            recry_index = np.transpose((self.grain_state > 0).nonzero())
            for i in range(int(p_line)):
                j = np.random.randint(lines_index.shape[0])
                x = lines_index[j][0]
                y = lines_index[j][1]
                self.to_change[x][y] = 0
                self.density[x][y] += pack_val
                if self.density[x][y] > critical_value:
                    self.grain_state[x][y] = 1
                    self.density[x][y] = 0
                    while True:
                        u = random.randint(0, 255)
                        color = (u, u, u)
                        if color not in self.colors.values():
                            self.colors[num_rec] = color
                            self.id[x][y] = num_rec
                            num_rec -= 1
                            break

            for i in range(int(p_inside)):
                x = np.random.randint(self.grain_energy.shape[0])
                y = np.random.randint(self.grain_energy.shape[1])
                self.density[x][y] += pack_val
                if self.density[x][y] > critical_value:
                    self.density[x][y] = 0
                    self.grain_state[x][y] = 1
                    while True:
                        u = random.randint(0, 255)
                        color = (u, u, u)
                        if not color in self.colors.values():
                            self.colors[num_rec] = color
                            self.id[x][y] = num_rec
                            num_rec -= 1
                            break
            self.to_change = np.ones(self.img.size)
            for x in range(recry_index.shape[0]):
                i = recry_index[x][0]
                j = recry_index[x][1]
                self.density[i][j] = 0
                tabx = self.density.shape[0]
                taby = self.density.shape[1]
                c_id = self.id[i][j]
                self.roz(i % tabx, (j + 1) % taby, c_id)
                self.roz(i % tabx, (j - 1) % taby, c_id)
                self.roz((i + 1) % tabx, j % taby, c_id)
                self.roz((i - 1) % tabx, j % taby, c_id)
            t += 0.001
        self.img_drx = self.change_picture()
        self.show_dys()
        df = pd.DataFrame(list_pom, columns=['time', 'ro'])
        df.to_csv("values.csv")

    def roz(self, i, j, idnow):
        neib = np.zeros(4)
        tabx = self.density.shape[0]
        taby = self.density.shape[1]
        if self.conditions == 'periodyczny':
            if i < 0:
                i = len(self.color) - 1
            if i > len(self.color) - 1:
                i = 0
            if j < 0:
                j = len(self.color[0]) - 1
            if j > len(self.color[0]) - 1:
                j = 0
        if self.conditions == 'absorbujacy':
            if i < 0:
                i = 0
            if i > len(self.color) - 1:
                i = len(self.color) - 1
            if j < 0:
                j = 0
            if j > len(self.color[0]) - 1:
                j = len(self.color[0]) - 1
        neib[0] = self.density[i % tabx][(j - 1) % taby]
        neib[3] = self.density[i % tabx][(j + 1) % taby]
        neib[1] = self.density[(i - 1) % tabx][j % taby]
        neib[2] = self.density[(i + 1) % tabx][j % taby]
        temp = np.transpose((neib >= self.density[i % tabx][j % taby]).nonzero())
        if temp.shape[0] >= 1:
            self.grain_state[i][j] = 0
        else:
            self.grain_state[i][j] = 1
            self.id[i][j] = idnow


    def show_dys(self):
        if (self.conditions is not None) and (self.neighb is not None):
            self.img_dys = im.new('RGB', self.img.size)
            for x in range(len(self.grain_state)):
                for y in range(len(self.grain_state[0])):
                    if self.grain_state[x][y] != 0:
                        self.img_dys.putpixel((y, x), (0, 255, 0))
                    else:
                        self.img_dys.putpixel((y, x), (150, 75, 0))


def generate_board(w, h):
    board.reset(w, h)
    show(board.img)


def show(img):
    global img_changed
    img_changed = img
    image = ImageTk.PhotoImage(img_changed)
    image_id = panell.create_image(0, 0, anchor=NW, image=image)
    panell.itemconfig(image_id, image=image)
    panell.image = image


def CA(bc, type):
    r = int(radius.get())
    q = int(quantity.get())
    z = location.get()
    board.step(bc, type, z, r, q)
    show(board.img)


def MC(par_k, i):
    board.mc(par_k, i)
    board.show_energy()
    show(board.img_mc)


def MC_energy():
    show(board.img_energy)


def DRX(len):
    board.step1(len)
    show(board.img_drx)

def DYS():
    show(board.img_dys)


window = Tk()
window.title("Modelowanie wieloskalowe")

left_frame = Frame(window)
right_frame = Frame(window)
left_up_frame = Frame(left_frame)

#tablica
label_1 = Label(left_up_frame, text="szerokosc")
label_1.pack(side=TOP)
width = Entry(left_up_frame)
width.insert(0, "150")
width.pack(side=TOP)
label_2 = Label(left_up_frame, text="wysokosc")
label_2.pack(side=TOP)
height = Entry(left_up_frame)
height.insert(0, "150")
height.pack(side=TOP)
start = Button(left_up_frame, text="generate board", width=15, command=lambda: generate_board((int(width.get())), (int(height.get()))))
start.pack(side=RIGHT)

boundary = StringVar()
conditions = Combobox(right_frame, textvariable=boundary)
conditions['values'] = ('periodyczne', 'absorbujace')
conditions.current(0)
label_3 = Label(right_frame, text="warunki brzegowe")
label_3.grid(row=0, column=0)
conditions.grid(row=0, column=1)

val = StringVar()
neigh = Combobox(right_frame, textvariable=val)
neigh['values'] = ('moore', 'von Neumann', 'pentagonalne lewe', 'pentagonalne prawe', 'heksagonalne lewe', 'heksagonalne prawe')
neigh.insert(0, 'moore')
neigh.current()
label_4 = Label(right_frame, text="sasiedztwo")
label_4.grid(row=1, column=0)
neigh.grid(row=1, column=1)

zar_value = StringVar()
location = Combobox(right_frame, textvariable=zar_value)
location['values'] = ('losowe', 'rownomierne', 'z promieniem', 'wyklikanie')
location.insert(0, 'losowe')
location.current()
label_5 = Label(right_frame, text="zarodkowanie")
label_5.grid(row=2, column=0)
location.grid(row=2, column=1)

numb = IntVar(value=5)
quantity = Spinbox(right_frame, textvariable=numb)
label_6 = Label(right_frame, text="liczba ziaren")
label_6.grid(row=3, column=0)
quantity.grid(row=3, column=1)

radius_val = IntVar(value=5)
radius = Spinbox(right_frame, textvariable=radius_val)
label_7 = Label(right_frame, text="promien")
label_7.grid(row=4, column=0)
radius.grid(row=4, column=1)

generate = Button(right_frame, text="CA", command=lambda: CA(conditions.get(), neigh.get()))
generate.grid(row=5, columnspan=2)

label_8 = Label(right_frame, text="liczba iteracji")
label_8.grid(row=6, column=0)
iteration = Entry(right_frame)
iteration.insert(0, '1')
iteration.grid(row=6, column=1)

label_9 = Label(right_frame, text="parametr k")
label_9.grid(row=7, column=0)
par_k = Entry(right_frame)
par_k.insert(0, '0.1')
par_k.grid(row=7, column=1)

generate_mc = Button(right_frame, text="MC", command=lambda: MC(float(par_k.get()), int(iteration.get())))
generate_mc.grid(row=8, columnspan=1)

generate_energy = Button(right_frame, text='ENERGY', command=lambda: MC_energy())
generate_energy.grid(row=8, columnspan=2)


label_10 = Label(right_frame, text="wielkosc")
label_10.grid(row=9, column=0)
pl_entry = Entry(right_frame)
pl_entry.insert(0, '100')
pl_entry.grid(row=9, column=1)

generate_drx = Button(right_frame, text="DRX", command=lambda: DRX(int(pl_entry.get())))
generate_drx.grid(row=10, columnspan=1)

gestosc_dys = Button(right_frame, text="DYSLOKACJA", command=lambda: DYS())
gestosc_dys.grid(row=10, columnspan=2)

panell = Canvas(left_frame, width=img.width, height=img.height)
panell.pack(side=TOP)
board = Board(panel=panell)


right_frame.pack(side=RIGHT)
left_frame.pack(side=BOTTOM)
left_up_frame.pack(side=TOP)


window.mainloop()

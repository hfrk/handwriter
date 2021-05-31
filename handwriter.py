# handwriter.py
import numpy as np
import cv2
import datetime
import random

class Page:
    """
    Creates a drawable blank Page object
    :Parameters:
        width, height: int
            Specifies the canvas size in pixels
        margin: tuple of int
            Specifies the margin of the page
            Format is (top,left,right,bottom)
    """
    def __init__(self, width, height, margin=(254,254,254,254)):
        self._width = width
        self._height = height
        self._canvas = np.ones([height,width,3], dtype='uint8') * 255
        self._margin = margin
        self._xcursor = margin[1]
        self._ycursor = margin[0]
        CHARSET =   {
            'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': [], 'H': [], 'I': [], 'J': [], 'K': [], 'L': [], 'M': [],
            'N': [], 'O': [], 'P': [], 'Q': [], 'R': [], 'S': [], 'T': [], 'U': [], 'V': [], 'W': [], 'X': [], 'Y': [], 'Z': [],
            'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [], 'g': [], 'h': [], 'i': [], 'j': [], 'k': [], 'l': [], 'm': [],
            'n': [], 'o': [], 'p': [], 'q': [], 'r': [], 's': [], 't': [], 'u': [], 'v': [], 'w': [], 'x': [], 'y': [], 'z': [],
            '0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
            '!': [], '?': [], ':': [], ';': [], '#': [], '%': [], '&': [], '^': [], '*': [], "'": [], '_': [], ',': [], '.': [],
            '+': [], '-': [], '=': [], '/': [], '|': [], '(': [], ')': [], '[': [], ']': [], '{': [], '}': [], '<': [], '>': []
                    }
        self._CHARSET = CHARSET
        self._keylist = [key for key in CHARSET.keys()]
        self._normalized = False
        self._prev_charset = {}
        self._prev_font_size = None

    def extract(self, filename, threshold1 = 0.01, threshold2 = 0.01, tries = 0):
        """
        Cut sample characters into charset
        """
        def _getPoints(arr, length, thres=0.01):
            """
            Returns the cropping points for a character
            """
            res = []
            last = -1
            for i in range(length-1):
                if arr[i].all() and not arr[i+1].all():
                    last = i+1
                elif arr[i+1].all() and not arr[i].all() and i+1-last > thres*length and last != -1:
                    res.append([last, i+1])
                    last = -1
            return res

        def _cleanse(arr):
            """
            Remove whites
            """
            isWhite = (arr==255)
            h, w = arr.shape[0:2]
            t, b, l, r = 0, h, 0, w

            if isWhite[t].all():
                for i in range(h-1):
                    if not isWhite[i+1].all() and isWhite[i].all():
                        t = i+1
                        break
            if isWhite[b-1].all():
                for i in range(h-1,-1,-1):
                    if not isWhite[i-1].all() and isWhite[i].all():
                        b = i
                        break
            isWhite = isWhite.transpose()
            if isWhite[l].all():
                for i in range(w-1):
                    if not isWhite[i+1].all() and isWhite[i].all():
                        l = i+1
                        break
            if isWhite[r-1].all():
                for i in range(w-1,-1,-1):
                    if not isWhite[i-1].all() and isWhite[i].all():
                        r = i
                        break
            return arr[t:b,l:r]
        
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        checker = [13,13,13,13,10,13,13]
        p = _getPoints((img == 128), img.shape[0], threshold1)
        while len(p) != len(checker):
            threshold1 *= (1-0.1*(len(checker)-len(p)))
            # print(f"BAD IMAGE: {len(checker)-len(p)} row(s) is missing")
            tries += 1
            p = _getPoints((img == 255), img.shape[0], threshold1)
            if tries >= 100:
                raise Exception("ERROR: IMAGE UNRECOGNIZED")
        counter = 0
        for i in range(len(p)):
            q = img[p[i][0]:p[i][1],:]
            isWhite = (q.transpose() == 255)
            point = _getPoints(isWhite, isWhite.shape[0], threshold2)
            while len(point) != checker[i]:
                threshold2 *= (1-0.1*(checker[i]-len(point)))
                # print(f"BAD IMAGE on row {i}: {checker[i]-len(point)} col(s) is missing")
                tries += 1
                point = _getPoints(isWhite, isWhite.shape[0], threshold2)
                if tries >= 100:
                    raise Exception("ERROR: IMAGE UNRECOGNIZED")
            for j in range(len(point)):
                r = q[:,point[j][0]:point[j][1]]
                r = _cleanse(r)
                self._CHARSET[self._keylist[counter]].append(r)
                counter += 1
        self._normalized = False

    def draw(self, text, font_size, color=np.uint8([0,0,0])):
        """
        Draws string of text into the canvas
        :Parameters:
            text: str
                Text to be drawn
            font_size: int
                Size of the drawn charset
            color: tuple of uint8, RGB format
                Color of text
        """
        def _normalize(font):
            if type(font) != type({}):
                return
            height_UPPER09 = max([x.shape[0] for x in font['A']])
            height_lowsmol = max([x.shape[0] for x in font['a']])
            height_lowtall = 1.2*max([x.shape[0] for x in font['f']])
            height_symsmol = 2*max([x.shape[0] for x in font['=']])
            height_symmini = 0.3*max([x.shape[0] for x in font[',']])
            for key in font.keys():
                if key in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?()[]{}/|#%&':
                    height = height_UPPER09
                elif key in 'aceimnorsuvwxz':
                    height = height_lowsmol
                elif key in 'bdfghjklpqty':
                    height = height_lowtall
                elif key in '+=<>:;^*':
                    height = height_symsmol
                elif key in "-'_,.":
                    height = height_symmini
                for i in range(len(font[key])):
                    h, w = font[key][i].shape[0:2]
                    font[key][i] = cv2.resize(font[key][i], (int(height/h*w), int(height)), interpolation = cv2.INTER_LINEAR)
            return font

        def _edit_charset(charset, size, color):
            res_charset = {}
            for key in charset.keys():
                res_charset[key] = []
                for i in range(len(charset[key])):
                    res = np.ndarray([charset[key][i].shape[0],charset[key][i].shape[1],3], dtype = 'uint8')
                    res[:,:,0] = (charset[key][i] <= 64) * (charset[key][i] / 255 * (255-color[2]) + color[2]) + (charset[key][i] > 64) * 255
                    res[:,:,1] = (charset[key][i] <= 64) * (charset[key][i] / 255 * (255-color[1]) + color[1]) + (charset[key][i] > 64) * 255
                    res[:,:,2] = (charset[key][i] <= 64) * (charset[key][i] / 255 * (255-color[0]) + color[0]) + (charset[key][i] > 64) * 255
                    res = np.uint8(res)
                    h, w = res.shape[0:2]
                    res_charset[key].append( cv2.resize(res, (int(w/300*size),int(h/300*size)), interpolation = cv2.INTER_AREA) )
            return res_charset

        def _put(char, canvas, x, y, offset):
            # canvas[y+1,x:x+char.shape[1],:] = np.zeros([1,char.shape[1],3], dtype='uint8')
            y += int(offset)
            canvas[y-char.shape[0]:y,x:x+char.shape[1],:] &= char[:,:,:]
            return canvas
        
        color = np.uint8(color)
        font_size = int(4/3 * font_size)
        charset = self._CHARSET if self._normalized else _normalize(self._CHARSET)
        self._normalized = True
        if self._prev_font_size != font_size:
            charset = _edit_charset(self._CHARSET, font_size, color)
            self._prev_charset = charset
        else:
            charset = self._prev_charset
        self._prev_font_size = font_size
        linebreak = int(1.5*font_size)
        whitespace = int(0.5*font_size)
        startx = self._xcursor
        starty = self._ycursor
        for char in text:
            if char in self._keylist:
                char_offset = 0 # vertical offset
                if char in 'fgjpqy':
                    char_offset = 0.4
                elif char in '+-=<>:;*':
                    char_offset = -0.2
                elif char in "^'":
                    char_offset = -0.6
                startx += int(random.uniform(-1,1)*0.15*font_size)
                starty += int(random.normalvariate(random.uniform(-0.2,0.2),0.35)*0.1*font_size)
                c = charset[char][random.randint(0,len(charset[char])-1)]
                
                if startx+c.shape[1] >= self._canvas.shape[1]-self._margin[2]:
                    startx = self._margin[1] + int(random.uniform(-1,1)*0.1*font_size)
                    starty += linebreak
                if starty >= self._canvas.shape[0]-self._margin[3]:
                    continue
                self._canvas = _put(c, self._canvas, startx, starty+font_size, char_offset*font_size)
                startx += c.shape[1]
            elif char == '\n':
                startx = self._margin[1] + int(random.uniform(-1,1)*0.1*font_size)
                starty += linebreak
            elif char == ' ':
                startx += whitespace + int(random.uniform(0,1)*0.1*font_size)
            elif char == '\t':
                startx += 4 * (whitespace + int(random.uniform(0,1)*0.1*font_size))
            else:
                startx += whitespace + int(random.uniform(0,1)*0.1*font_size)
                print(f"character '{char}' not yet provided")
        self._canvas = np.uint8(self._canvas)
        self._xcursor = self._margin[1]
        self._ycursor = starty
        self._prev_font_size = font_size

    def show(self):
        """
        Shows the preview of the canvas
        """
        cv2.imshow('canvas', self._canvas)
        cv2.waitKey(0)
        cv2.destroyWindow('canvas')

    def save(self, filename='canvas.jpg'):
        """
        Save the canvas as an image file
        """
        cv2.imwrite(filename, self._canvas)

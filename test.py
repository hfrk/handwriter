# test.py
import handwriter
import datetime

page = handwriter.Page(2100,2970,[150, 150, 150, 150])
page.extract('alphabet/alphabet2_1.jpg')
page.extract('alphabet/alphabet2_2.jpg')
with open('lorem ipsum.txt', 'r') as f:
    for foo in f:
        page.draw(foo, 45, color=(0,50,120))
page.save('result/canvas.jpg')

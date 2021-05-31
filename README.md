# Handwriter
A lazy implementation of "computer-generated" handwriting for y'all lazy peeps!

## How to use
Too lazy to handwrite tens of pages of lab reports? Open your Python interpreter and write the following:
```
...
import handwriter

page = Page(WIDTH, HEIGHT, PAGE_MARGIN)
page.extract('YOUR-SAMPLE-HANDWRITING.jpg')
...
with open('YOUR-TXT-FILE.txt') as f:
    page.draw(f.read(), FONT_SIZE, COLOR)
page.save()
```
...and voila! You just wrote yourself a page of your lab report!

## Features
(more features are still under development)

import os

picnum = 0
strimage = ''
strlabel = ''

def eachFile(filepath, filenum):

    global picnum
    global strimage
    global strlabel

    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        filename = allDir.split('.')
        strpng = filename[0] + '.jpg'
        picnum = picnum + 1
        picstr = str(picnum)
        strimage = strimage + picstr + ' ' + filenum + '/' + strpng + "\n"
        strlabel = strlabel + picstr + ' ' + filenum + "\n"


if __name__ == '__main__':

    for i in range(1, 27):
        filenum = str(i)
        filepath = './data/TJAID/' + filenum + '/'
        eachFile(filepath, filenum)

    facc = open('./data/TJAID/images', "w")
    facc.write(strimage)
    facc.close()

    facc = open('./data/TJAID/image_class_labels', "w")
    facc.write(strlabel)
    facc.close()

    print("finish write images and image_class_labels")
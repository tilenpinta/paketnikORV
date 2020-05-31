def lbp(image):
    height = image.shape[0]
    width = image.shape[1]
    imgLBP = np.zeros(shape=(height, width), dtype="uint8")

    maskSize = 3
    for i in range(0, height - maskSize):
        for j in range(0, width - maskSize):
            img = image[i:i+maskSize, j:j+maskSize]

            result = (img >= img[1, 1])*1
            result2 = np.zeros(8)

            result2[0] = result[0, 0]
            result2[1] = result[0, 1]
            result2[2] = result[0, 2]
            result2[3] = result[1, 2]
            result2[4] = result[2, 2]
            result2[5] = result[2, 1]
            result2[6] = result[2, 0]
            result2[7] = result[1, 0]

            vector = np.where(result2)[0]

            num = np.sum(2**vector)

            imgLBP[i+1, j+1] = num
    return imgLBP

def LBPd(d):
    LPBd1

    for y in range(0, height - 1):
        for x in range(0, width - 1):
            byte = []
            upL = grayImg[y-1, x-1]
            upR = grayImg[y - 1, x + 1]
            left = grayImg[y, x - 1]
            right = grayImg[y, x + 1]
            downL = grayImg[y + 1, x - 1]
            down = grayImg[y + 1, x]
            downR = grayImg[y + 1, x + 1]
            arr = [upL, up, upR, left, right, downL, down, downR]

            num = 0

            for i in arr:
                primerjaj = num+d

                if primerjaj > 7:
                    primerjaj = (primerjaj - 7) - 1

                if i >= arr[primerjaj]:
                    byte.append(1)
                else:
                    byte.append(0)
                num =+ 1

            rezultat = 0

            for i in range(0, 8):
                rezultat += utezi[i] * byte[i]

            LBPd1[y, x] = rezultat
    return LBPd1

def angle(x, y):
    if x == 0:
        x=1
    kot = np.round(np.arctan2(y, x) * (180/np.pi))

    return kot

def gradientX(G, h, w):
    Gradient = G.copy()

    for x in range(h):
        for y in range(w):
            if (x == h-1 or y == w-1):
                Gradient[x][y] = 0
            else:
                Gradient[x][y] = abs(int(G[x][y -1]*(-1)) + int(G[x][y+1])) #[-1,0,1] pomnozis pa sestejes

    return Gradient

def gradientY(G, h, w):
    Gradient = G.copy()

    for x in range(h):
        for y in range(w):
            if (x == h-1 or y == w-1):
                Gradient[x][y] = 0
            else:
                Gradient[x][y] = abs(int(G[x-1][y] * (-1)) + int(G[x+1][y]))  # [[0,-1,0],[0,0,0],[0,1,0]] pomnozis pa sestejes

    return Gradient

def HOG(bins, cellsize, regionsize):
    grayImg
    widthR = width
    heightR = height

    while(True): #sirina slike deljiva z cellsize
        if widthR % cellsize == 0:
            break
        else:
            widthR+=1

    while (True):
        if heightR % cellsize == 0:
            break
        else:
            heightR += 1

    imgResized = cv2.resize(grayImg, (widthR, heightR))

    Gx = np.asarray(imgResized).copy()
    Gy = np.asarray(imgResized).copy()
    Gx = gradientX(Gx, heightR, widthR)
    Gy = gradientY(Gy, heightR, widthR)
    Gradient = np.round(np.sqrt(np.multiply(Gx, Gx) + np.multiply(Gy, Gy)))
    A = angle(Gx, Gy)

    spaceBin = 180/bins
    num = 0


    for regY in range(0, heightR-cellsize, cellsize):
        for regX in range(0, widthR-cellsize, cellsize):
            num+=1
            Hist = [[0.0] * bins for i in range(regionsize * regionsize)]
            plusHisto = 0

            startRegX = regX
            startRegY = regY

            konecRegijeX = regX + (cellsize * regionsize)
            konecRegijeY = regY + (cellsize * regionsize)

            for celY in range(startRegY, konecRegijeY, cellsize):
                for celX in range(startRegX, konecRegijeX, cellsize):

                    startCelX = celX
                    startCelY = celY
                    endCelX = celX + cellsize
                    endCelY = celY + cellsize

                    for Y in range(startCelY, endCelY):
                        for X in range(startCelX, endCelX):
                            kot = A[Y, X]
                            gradient = Gradient[Y, X]

                            if (kot > spaceBin):
                                index = np.ceil((kot / spaceBin))
                                ostanek1 = index - (kot / spaceBin)
                                trenutni = (1 - ostanek1) * gradient
                                prejsni = ostanek1 * gradient
                            else:
                                index = np.ceil((kot / spaceBin))
                                ostanek1 = (kot / spaceBin)
                                prejsni = (1 - ostanek1) * gradient
                                trenutni = ostanek1 * gradient

                            Hist[plusHisto][int(index)] += trenutni
                            Hist[plusHisto][int(index - 1)] += prejsni

                    koncneVrenosti = []
                    plusHisto += 1
                    for histogram in range(0, regionsize * regionsize):
                        for steviloHistgrama in range(0, bins):
                            koncneVrenosti.append(Hist[histogram][steviloHistgrama])

                    print(koncneVrenosti)
                    print(len(koncneVrenosti))
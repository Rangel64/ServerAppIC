import cv2
import numpy as np
import scipy.fftpack as fp
import keras
import pandas as pd

def colorMedia(matrizVerde, matrizVermelho, matrizAzul):
    (width, height) = np.shape(matrizVerde)
    i = 0
    mediaVerde = 0
    qtdVerde = 0
    mediaVermelho = 0
    qtdVermelho = 0
    mediaAzul = 0
    qtdAzul = 0
    while i < width:
        j = 0
        sumRowVerde = 0
        qtdRowVerde = 0
        sumRowVermelho = 0
        qtdRowVermelho = 0
        sumRowAzul = 0
        qtdRowAzul = 0
        while j < height:
            sumRowVerde += matrizVerde[i][j]
            qtdRowVerde += 1
            sumRowVermelho += matrizVermelho[i][j]
            qtdRowVermelho += 1
            sumRowAzul += matrizAzul[i][j]
            qtdRowAzul += 1
            j += 1
        if qtdRowVerde > 0:
            mediaVerde += sumRowVerde/qtdRowVerde
            qtdVerde += 1
        if qtdRowVermelho > 0:
            mediaVermelho += sumRowVermelho/qtdRowVermelho
            qtdVermelho += 1
        if qtdRowAzul > 0:
            mediaAzul += sumRowAzul/qtdRowAzul
            qtdAzul += 1
        i += 1
    return (mediaVerde/qtdVerde, mediaVermelho/qtdVermelho, mediaAzul/qtdAzul )

def calculoIndicesInfraRed(red, green, blue):
    # gli = (2*green-red-blue)/(2*green+red+blue)
    savi = 1.5*((green-red)/(green+red+0.5))
    mpri = (green-red)/(green+red+0.5)

    return savi, mpri


def calculoIndicesRNA(imageBytes):
    
    
    print('Carregando...')
    image = cv2.imdecode(np.frombuffer(imageBytes,np.uint8),1)
    
    
    (imageBlue,imageGreen,imageRed) = cv2.split(image) 
    
    redFFT = fp.fft2(imageRed)
    greenFFT = fp.fft2(imageGreen)
    blueFFT = fp.fft2(imageBlue)

    redFFTm = np.absolute(redFFT)
    greenFFTm = np.absolute(greenFFT)
    blueFFTm = np.absolute(blueFFT)

    redFFTm = fp.fftshift(redFFTm)
    greenFFTm = fp.fftshift(greenFFTm)
    blueFFTm = fp.fftshift(blueFFTm)
    
    (Green,Red,Blue) = colorMedia(greenFFTm, redFFTm, blueFFTm)
    
    nRed = Red/(Red+Green+Blue)
    nGreen = Green/(Red+Green+Blue)
    nBlue = Blue/(Red+Green+Blue)
    
    print('Calculando os indices.')
    (savi,mpri) = calculoIndicesInfraRed(nRed,nGreen,nBlue)
    print(nRed,nGreen,nBlue,savi,mpri)
    
    return (nRed,nGreen,nBlue,savi,mpri)

def rede (nRed,nGreen,nBlue,savi,mpri) :
    
    print('Rede neural analisando a imagem')
    
    entrada = []
    entrada.append(nRed)
    entrada.append(nGreen)
    entrada.append(nBlue)
    # entrada.append(gli)
    entrada.append(savi)
    entrada.append(mpri)
    
    
    df = pd.DataFrame(entrada)
    
    array = np.array(df)
    array = np.transpose(array)
    
    df2 = pd.DataFrame(array)
    
    
    print(df2)
    #csv = df.to_csv('D:/IFTM/Engenharia da Computacao/Iniciacao cientifica/Aplicacoes_de_RNA_sao_calculo_de_lotacao_de_pastagens_Rangel_Neves_Souza/Entradas/aaaaaaaaa.csv', index=False)
    
    model = keras.models.load_model('models/my_model_teste_34.h5')
    #model = keras.models.load_model('C:/Users/range/Desktop/Aplicacoes_de_RNA_sao_calculo_de_lotacao_de_pastagens_Rangel_Neves_Souza/models/my_model_dinamarca.h5')
    
    return  model.predict(df2)



#(nRed,nGreen,nBlue,savi,mpri) = calculoIndicesRNA('C:/Users/range/Desktop/Aplicacoes_de_RNA_sao_calculo_de_lotacao_de_pastagens_Rangel_Neves_Souza/biomass_data/train/images/biomass_image_train_0026.jpg')

def result(imageBytes):
    (nRed,nGreen,nBlue,savi,mpri) = calculoIndicesRNA(imageBytes)

    result = rede(nRed,nGreen,nBlue,savi,mpri)
    print('Massa estimada: ' + str(result[0][0]))
    
    return result[0][0]






                                                                                        














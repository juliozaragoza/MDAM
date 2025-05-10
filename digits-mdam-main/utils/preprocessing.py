import cv2

def to_white(image, threshold):
    # Cargar la imagen en escala de grises
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar el umbral para convertir los píxeles oscuros en blanco
    _, thresholded = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded

def to_black(image, threshold):
    # Cargar la imagen en escala de grises
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar el umbral para convertir los píxeles claros en negro
    _, thresholded = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    return thresholded

def to_black_and_white(image, threshold1, threshold2):
    # Cargar la imagen en escala de grises
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar el umbral para convertir los píxeles oscuros en blanco y los claros en negro
    _, thresholded1 = cv2.threshold(img, threshold1, 255, cv2.THRESH_BINARY_INV)
    _, thresholded2 = cv2.threshold(thresholded1, threshold2, 255, cv2.THRESH_BINARY)
    
    return thresholded2

import cv2

def resize_image(img, length=64, board=False):

    (h, w) = img.shape[0:2]

    # 获取长的一边缩放比 
    scale = min(length/w, length/h)

    width = int(w * scale)
    height = int(h * scale)

    # 如果是放大
    if scale > 1:
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    else: # 如果是缩小
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    if board:
        fill_color = [0,0,0]
        if width > height:
            fill_board = (length - height) // 2
            img = cv2.copyMakeBorder(img, fill_board, fill_board, 0, 0, cv2.BORDER_CONSTANT, value=fill_color)
        else:
            fill_board = (length - width) // 2
            img = cv2.copyMakeBorder(img, 0, 0, fill_board, fill_board, cv2.BORDER_CONSTANT, value=fill_color)

    return img

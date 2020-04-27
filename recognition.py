import cv2 
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image,ImageDraw,ImageFont



def color_change(image):
    '''
    将图片转化到HSV空间，并按位取反,突出车牌区域
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)     #转化为HSV颜色空间
    lower_blue = np.array([100, 43, 46])      #蓝色阈值
    upper_blue = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)     #构建掩模
    res = cv2.bitwise_and(image, image, mask=mask)  #按位运算
    return res


def binaryzation(image):
    '''
    把图象进行二值化，并进行开闭运算，最后找到可能存在的区域,并返回区域信息
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转化为灰度图
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)  #图像二值化

    canny = cv2.Canny(thresh, 100, 200)  #边缘检测
    kernel = np.ones((12,40), np.uint8)  
    img_edge = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)         # 闭运算
    img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_OPEN, kernel)  # 开运算

    contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def cut_out(contours, image):
    '''
    通过长宽比判断车牌的位置，并截取
    '''                                
    for contour in contours:
        rect = cv2.minAreaRect(contour)   #找出最小外接矩形 中心点、宽和高、角度
        if rect[1][1] > rect[1][0]:
            k = rect[1][1] / rect[1][0]
        else:
            k = rect[1][0]/rect[1][1]
        if (k > 2.5) & (k < 5):    #判断车牌的轮廓
        
            points = cv2.boxPoints(rect)     #获取外接矩形的四个点
            box = np.int0(points)
            image_with_box = cv2.drawContours(image, [box], -1, (0, 255, 0), 3)  #找出车牌的位置     
            x = []
            y = []
            for i in range(4):
                x.append(box[i][1])
                y.append(box[i][0])
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            max_y = max(y)
            cut = image[min_x:max_x, min_y:max_y] 
            return cut

def find_cut_pos(histogram, compare_range, cut_gap):
    '''
    histogram 是图像某个方向统计的白色像素数量的直方图
    compare_range 是要进行波谷判断时左右两边的范围大小
    cut_gap 是判断两个切割位置是否太靠近的阈值
    '''
    
    cut_pos = []
    hist_len = len(histogram)
    for index, hist in enumerate(histogram):
        if index < compare_range:
            left = histogram[ :index ]
            right = histogram[index+1 : index+compare_range]
        elif index + compare_range < hist_len:
            left = histogram[ index-compare_range:index ]
            right = histogram[index+1 : index+compare_range]
        else:
            left = histogram[ index-compare_range:index ]
            right = histogram[index+1:]
        
        left_min = 0
        right_min = 0
        if len(left) > 0:
            left_min = left.min()
        if len(right) > 0:
            right_min = right.min()
            
        # 跟左右两边的区域的数据比较，判断是否处于波谷
        if (hist < left_min and hist < right_min) or (hist < left_min and hist == right_min) or (hist == left_min and hist < right_min):
            cut_pos.append(index)
    
    # 找一下哪些是距离比较近的一些候选切割位置
    remove_index = []
    for i in range(len(cut_pos) - 1):
        if cut_pos[i] + cut_gap >= cut_pos[i+1]:
            remove_index.append(i)
    # 将距离比较近的一些候选切割位置去除
    final_cut_pos = []
    for i in range( len(cut_pos) ):
        if i in remove_index:
            continue
        final_cut_pos.append( cut_pos[i] )
    
    return final_cut_pos
    

def car_binaryzation_cut(image):
    '''
    截取车牌的二值化并切割
    '''
    # RGB转GARY
    change_size= cv2.resize(image, (440, 140))

    gray_img = cv2.cvtColor(change_size, cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)       #图像二值化

    #获取行方向的字符像素分布图
    histogram = []
    for i in range(140):  
        n = 0
        for j in range(440):
            if thresh[i, j] == 255:
                n += 1
        histogram.append(n)
    histogram = np.array(histogram)
    # 根据波谷找出垂直方向分割的位置
    cut_pos_x = find_cut_pos(histogram, 60, 5)

    # 将车牌区域的上下非字符部分设置为黑色
    if len(cut_pos_x) == 2:
        thresh[:cut_pos_x[0]] = 0
        thresh[cut_pos_x[1]:] = 0

    # 获取列方向的字符像素分布图
    histogram = []
    for i in range(440):
        n = 0
        for j in range(140):
            if thresh[j, i] == 255:
                n += 1
        histogram.append(n)
    histogram = np.array(histogram)
    # 根据波谷找出水平方向分割的位置
    cut_pos_y = find_cut_pos(histogram, 30, 5)
    
    if len(cut_pos_x) == 2:
        x_up = cut_pos_x[0]
        x_down = cut_pos_x[1]
    else:
        x_up = 20
        x_down = 120

    if len(cut_pos_y) == 14: 
        p1 = thresh[x_up:x_down, cut_pos_y[0]:cut_pos_y[1]]
        p2 = thresh[x_up:x_down, cut_pos_y[2]:cut_pos_y[3]]
        p3 = thresh[x_up:x_down, cut_pos_y[4]:cut_pos_y[5]]
        p4 = thresh[x_up:x_down, cut_pos_y[6]:cut_pos_y[7]]
        p5 = thresh[x_up:x_down, cut_pos_y[8]:cut_pos_y[9]]
        p6 = thresh[x_up:x_down, cut_pos_y[10]:cut_pos_y[11]]
        p7 = thresh[x_up:x_down, cut_pos_y[12]:cut_pos_y[13]]
    else:
        p1 = thresh[x_up:x_down, 12:63]
        p2 = thresh[x_up:x_down, 65:117]
        p3 = thresh[x_up:x_down, 142:197]
        p4 = thresh[x_up:x_down, 200:260]
        p5 = thresh[x_up:x_down, 260:317]
        p6 = thresh[x_up:x_down, 320:365]
        p7 = thresh[x_up:x_down, 370:425]

    img_list = [p1, p2, p3, p4, p5, p6, p7]
    return img_list

def char_reconition(img_list):
    char_imgs = [
        '0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', 
        'A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png', 'G.png', 'H.png', 'J.png', 
        'K.png', 'L.png', 'M.png', 'N.png', 'P.png', 'Q.png', 'R.png', 'S.png', 'T.png', 
        'U.png', 'V.png', 'W.png', 'X.png', 'Y.png', 'Z.png', 
        '云.png', '京.png', '冀.png', '吉.png', '宁.png', '川.png', '晋.png', '桂.png', '沪.png', 
        '津.png', '浙.png', '湘.png', '皖.png', '粤.png', '苏.png', '蒙.png', '豫.png', '赣.png', 
        '辽.png', '闽.png', '陕.png', '鲁.png', '黑.png',   
    ]
    char_list = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 
        'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z', 
        '云', '京', '冀', '吉', '宁', '川', '晋', '桂', '沪', 
        '津', '浙', '湘', '皖', '粤', '苏', '蒙', '豫', '赣', 
        '辽', '闽', '陕', '鲁', '黑',   
    ]
    
    ss = []
    for i in range(7):    #模板匹配
        s = []
        p_h, p_w = img_list[i].shape
        
        for j in range(len(char_imgs)):
            if i == 0 and j < 34:
                continue
            
            #读入图像，针对中文的方法
            image = cv2.imdecode(np.fromfile('./tmp/' + char_imgs[j], dtype=np.uint8), -1)

            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except:
                pass
            # 放到一样大小进行match
            image = cv2.resize(image, (p_w, p_h) )

            __, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            # result = cv2.matchTemplate(img_list[i], thresh, cv2.TM_CCOEFF_NORMED)
            result = cv2.matchTemplate(img_list[i], thresh, cv2.TM_CCOEFF_NORMED)

            s.append(result.item())
        # 找出最匹配的一个结果
        max_index = np.argmax(s)

        if i == 0:
            ss.append(char_list[ max_index + 34 ])
        else:
            ss.append(char_list[ max_index ])

    car = ss[0] + ss[1] + '.' + ss[2] + ss[3] + ss[4] + ss[5] + ss[6]
    return car


if __name__ == '__main__':                                    
    # 读入图片，由于有的图片名字是中文，所以得用这种方法读取
    image = cv2.imdecode(np.fromfile('./test/吉AGH827.jpg', dtype=np.uint8), -1)
    # image = cv2.imdecode(np.fromfile('./test/粤A6ZC93.jpg', dtype=np.uint8), -1)
    # image = cv2.imdecode(np.fromfile('./test/蒙AN6715.jpg', dtype=np.uint8), -1)
    # image = cv2.imdecode(np.fromfile('./test/苏M32991.jfif', dtype=np.uint8), -1)
    # image = cv2.imdecode(np.fromfile('./test/辽LU2345.jfif', dtype=np.uint8), -1)
    image = cv2.imdecode(np.fromfile('./test/陕A866W5.jfif', dtype=np.uint8), -1)

    
    image = cv2.resize(image, (570, 430))
    
    import time
    t1 = time.time()

    res = color_change(image)        #转化到HSV，并大致取出车牌位置
    contours = binaryzation(res)                  #对颜色识别过的区域进行二值化，并进行开闭运算识别轮廓
    lp_image = cut_out(contours, image)                  #找到符合条件的区域，并进行切割
    cut_img_list = car_binaryzation_cut(lp_image)
    result = char_reconition(cut_img_list)

    t2 = time.time()
    print('运行所需时间:  '+ str(t2-t1) )

    print(result)
    # 按 esc 退出
    # while True:
    #     k=cv2.waitKey(5)&0xFF
    #     if k==27:
    #         cv2.destroyAllWindows()
    #         break

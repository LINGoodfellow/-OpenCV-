import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# 在 mac 实验过程中，出现过字体缺失，如果没有黑体，老师可以在这里下载 https://www.fontpalace.com/font-details/SimHei/

char_imgs = [
    'cmp.png', '云.png', '0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', 
    'A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png', 'G.png', 'H.png', 'J.png', 
    'K.png', 'L.png', 'M.png', 'N.png', 'P.png', 'Q.png', 'R.png', 'S.png', 'T.png', 
    'U.png', 'V.png', 'W.png', 'X.png', 'Y.png', 'Z.png', 
     '京.png', '冀.png', '吉.png', '宁.png', '川.png', '晋.png', '桂.png', '沪.png', 
    '津.png', '浙.png', '湘.png', '皖.png', '粤.png', '苏.png', '蒙.png', '豫.png', '赣.png', 
    '辽.png', '闽.png', '陕.png', '鲁.png', '黑.png',   
]

char_list = [
    'cmp', '云', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 
    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 
     '京', '冀', '吉', '宁', '川', '晋', '桂', '沪', 
    '津', '浙', '湘', '皖', '粤', '苏', '蒙', '豫', '赣', 
    '辽', '闽', '陕', '鲁', '黑',   
]


colors = [ '#054E9F' for i in range(len(char_imgs)-1) ]     # 全部直方图柱颜色设置为蓝色,cmp 与 cmp的比较不需要
colors[0] = 'r'     # 第一条直方柱颜色设置成红色

target_index = 0
target_img = cv2.imdecode(np.fromfile('./tmp/' + char_imgs[target_index], dtype = np.uint8), -1)
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

# 归一化常用算法
approach = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]    # 为matchTemplate的参数中的参数，依次为均方差算法、相关性算法和相关性系数算法
approach_name = ['均方差', '相关性', '相关性系数']

for approach_index in range(  len(approach) ):
    r = []
    for j in range( len(char_list) ):
        #读入图像，针对中文的方法
        image = cv2.imdecode(np.fromfile('./tmp/' + char_imgs[j], dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        __, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        result = cv2.matchTemplate(target_img, thresh, approach[approach_index] )
        r.append(result.item())

    r = np.array(r)
    # 计算和其他模板匹配的分数的误差
    print( approach_name[approach_index], np.mean( abs((r[2:] - r[1])) ) )

    plt.subplot(1, 3, approach_index+1)
    plt.bar( range(len(r)-1), r[1:], color=colors)
    plt.xticks([])  # 禁用横坐标刻度
    plt.title('%s' % (approach_name[approach_index]), size=20)

plt.show()
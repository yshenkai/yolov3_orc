import numpy as np
from PIL import Image
def resizeNormalize(img,imgH=32):
    scale=img.size[1]*1.0/imgH
    w=int(img.size[0]/scale)
    img=img.resize((w,imgH),Image.BILINEAR)
    w,h=img.size
    img=(np.array(img)/255.-0.5)/0.5#normalize -1到1
    return img

def strLableConverter(res,alpha):#res 网络返回的结果 label， alpha是一个列表，里面有key与汉字的映射 map
    N=len(res)
    raw=[]
    for i in range(N):
        if res[i]!=0 and (not (i>0 and res[i-1]==res[i])):
            raw.append(alpha[res[i]-1])
    return ''.join(raw)

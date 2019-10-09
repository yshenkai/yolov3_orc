from functools import reduce
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb
def compose(*funcs):
    if funcs:
        return reduce(lambda f,g:lambda *a,**kw:g(f(*a,**kw)),funcs)
    else:
        raise ValueError("compose of empty sequece")
def letterbox_image(image,size):
    '''
    resize image with unchange aspect ratio using padding
    :param image:
    :param size:
    :return:
    '''
    curr_w,curr_h=image.size
    w,h=size
    scale=min(w/curr_w,h/curr_h)
    nw=int(curr_w*scale)
    nh=int(curr_h*scale)

    image=image.resize((nw,nh),Image.BILINEAR)
    new_image=Image.new(mode="RGB",size=size,color=(128,128,128))
    new_image.paste(image,((w-nw)/2,(h-nh)/2))
    return new_image
def rand(a=0,b=1):
    return np.random.rand()*(b-a)+a

def get_random_data(annotation_line,input_shape,random=True,max_boxes=20,jitter=0.3,hue=0.1,sat=1.5,val=1.5,proc_img=True):
    line=annotation_line.split()
    image=Image.open(line[0])
    iw,ih=image.size
    h,w=input_shape
    box=np.array([np.array(list(map(int,box.split(",")))) for box in line[1:]])
    if not random:
        scale=min(w/iw,h/ih)
        nw=int(iw*scale)
        nh=int(ih*scale)
        dx=(w-nw)//2
        dy=(h-nh)//2
        image_data=0
        if proc_img:
            image=image.resize((nw,nh),Image.BILINEAR)
            new_image=Image.new(mode="RGB",size=(w,h),color=(128,128,128))
            new_image.paste(image,box=(dx,dy))
            image_data=np.array(new_image)/255.
        box_data=np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if(len(box)>max_boxes): box=box[:max_boxes]
            box[:,[0,2]]=box[:,[0,2]]*scale+dx
            box[:,[1,3]]=box[:,[1,3]]*scale+dy
            box_data[:len(box)]=box
        return image_data,box_data
    new_ar=w/h+rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale=rand(0.25,2)
    if new_ar<1:
        nh=int(scale*h)
        nw=int(nh*new_ar)
    else:
        nw=int(scale*w)
        nh=int(nw/new_ar)
    image=image.resize((nw,nh),Image.BILINEAR)

    dx=int(rand(0,w-nw))
    dy=int(rand(0,h-nh))
    new_image=Image.new(mode="RGB",size=(w,h),color=(128,128,128))
    new_image.paste(image,box=(dx,dy))
    image=new_image
    filp=rand()<0.5
    if filp:
        image=image.transpose(Image.FLIP_LEFT_RIGHT)
    hue=rand(-hue,hue)
    sat=rand(1,sat) if rand()<0.5 else 1/rand(1,sat)
    val=rand(1,val) if rand()<0.5 else 1/rand(1,val)
    x=rgb_to_hsv(np.array(image)/255.0)
    x[...,0]+=hue
    x[...,0][x[...,0]>1]-=1
    x[...,0][x[...,0]<0]+=1
    x[...,1]*=sat
    x[...,2]*=val
    x[x>1]=1
    x[x<0]=0
    image_data=hsv_to_rgb(x)
    box_data=np.zeros(shape=(max_boxes,5))
    if(len(box)>0):
        np.random.shuffle(box)
        box[:,[0,2]]=box[:,[0,2]]*nw/iw+dx
        box[:,[1,3]]=box[:,[1,3]]*nh/ih+dy
        if filp:
            box[:,[0,2]]=w-box[:,[2,0]]
            box[:,0:2][box[:,0:2]<0]=0
            box[:,2][box[:,2]>w]=w
            box[:,3][box[:,3]>h]=h
            box_w=box[:,2]-box[:,0]
            box_h=box[:,3]-box[:,1]
            box=box[np.logical_and(box_w>1,box_h>1)]
            if len(box)>max_boxes:box=box[:max_boxes]
            box_data[:len(box)]=box
    return image_data,box_data



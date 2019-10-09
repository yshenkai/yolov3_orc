from functools import wraps
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D,Add,ZeroPadding2D,UpSampling2D,Concatenate,MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.regularizers import l2
from yolov3.util import compose
import numpy as np
@wraps(Conv2D)
def DarknetConv2D(*args,**kwargs):
    darknet_conv_kwargs={"kernel_regularizer":l2(5e-4)}
    darknet_conv_kwargs["padding"]='valid' if kwargs.get("strides")==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args,**darknet_conv_kwargs)
def DarknetConv2D_BN_Leaky(*args,**kwargs):
    no_bias_kwargs={"use_bias":False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args,**no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
    )
def resblock_body(x,num_filters,num_blocks):
    x=ZeroPadding2D(padding=((1,0),(1,0)))(x)
    x=DarknetConv2D_BN_Leaky(num_filters,(3,3),strides=(2,2))(x)
    for i in range(num_blocks):
        y=compose(
            DarknetConv2D_BN_Leaky(num_filters//2,(1,1)),
            DarknetConv2D_BN_Leaky(num_filters,(3,3))
        )(x)
        x=Add()([x,y])
    return x

def darknet_body(x):
    x=DarknetConv2D_BN_Leaky(32,(3,3))(x)
    x=resblock_body(x,64,1)
    x=resblock_body(x,128,2)
    x=resblock_body(x,256,8)
    x=resblock_body(x,512,8)
    x=resblock_body(x,1024,4)
    return x

def make_last_layer(x,num_filters,out_filters):
    x=compose(
        DarknetConv2D_BN_Leaky(num_filters,(1,1)),
        DarknetConv2D_BN_Leaky(num_filters*2,(3,3)),
        DarknetConv2D_BN_Leaky(num_filters,(1,1)),
        DarknetConv2D_BN_Leaky(num_filters*2,(3,3)),
        DarknetConv2D_BN_Leaky(num_filters,(1,1))
    )(x)
    y=compose(
        DarknetConv2D_BN_Leaky(num_filters*2,(3,3)),
        DarknetConv2D(out_filters,(1,1))
    )(x)
    return x,y

def yolo_body(inputs,num_anchors,num_classes):
    darknet=Model(inputs,darknet_body(inputs))
    x,y1=make_last_layer(darknet.output,512,num_anchors*(num_classes+5))
    x=compose(
        DarknetConv2D_BN_Leaky(256,(1,1)),
        UpSampling2D(2)
    )(x)
    x=Concatenate()([x,darknet.layers[152].output])
    x,y2=make_last_layer(x,256,num_anchors*(num_classes+5))
    x=compose(
        DarknetConv2D_BN_Leaky(256,(1,1)),
        UpSampling2D(2)
    )(x)
    x=Concatenate()([x,darknet.layers[92].output])
    x,y3=make_last_layer(x,128,num_anchors*(num_classes+5))
    return Model(inputs,[y1,y2,y3])

def yolo_head(feats,anchors,num_classes,input_shape,calc_loss=False):
    '''

    :param feats: bacth_size w,h,num_anchors 85
    :param anchors: (9,2)
    :param num_classes:
    :param input_shape:
    :param calc_loss:
    :return:
    '''
    num_anchors=len(anchors)
    anchors_tensor=K.reshape(anchors,shape=(1,1,1,num_anchors,2))
    grid_shape=K.shape(feats)[1:3]#w,h
    grid_y=K.tile(K.reshape(K.arange(0,stop=grid_shape[0]),[-1,1,1,1]),[1,grid_shape[1],1,1])
    grid_x=K.tile(K.reshape(K.arange(0,stop=grid_shape[1]),[1,-1,1,1]),[grid_shape[0],1,1,1])
    grid=K.constant([grid_x,grid_y])
    grid=K.cast(grid,K.dtype(feats))
    feats=K.reshape(feats,[-1,grid_shape[0],grid_shape[1],num_anchors,num_classes+5])
    box_xy=(K.sigmoid(feats[...,:2])+grid)/K.cast(grid_shape[::-1],K.dtype(feats))
    box_wh=K.exp(feats[...,2:4])*anchors_tensor/K.cast(input_shape[::-1],K.dtype(feats))
    box_confidence=K.sigmoid(feats[...,4:5])
    box_class_probs=K.sigmoid(feats[...,5:])
    if calc_loss==True:
        return grid,feats,box_xy,box_wh
    return box_xy,box_xy,box_confidence,box_class_probs


def yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape):
    '''get corrected boxes'''
    box_yx=box_xy[...,::-1]
    box_hw=box_wh[...,::-1]
    input_shape=K.cast(input_shape,K.dtype(box_yx))
    image_shape=K.cast(image_shape,K.dtype(box_yx))
    new_shape=K.round(image_shape*K.min(input_shape/image_shape))
    offset=(input_shape-new_shape)/2./input_shape
    scale=input_shape/new_shape
    box_yx=(box_yx-offset)*scale
    box_hw*=scale
    box_mins=box_yx-(box_hw/2.)
    box_maxes=box_yx+(box_hw/2.)
    boxes=K.concatenate([
        box_mins[...,0:1],
        box_mins[...,1:2],
        box_maxes[...,0:1],
        box_maxes[...,1:2]
    ])
    boxes*=K.concatenate([image_shape,image_shape])
    return boxes
def yolo_boxes_and_scores(feats,anchors,num_classes,input_shape,image_shape):
    box_xy,box_wh,box_confidence,box_class_prob=yolo_head(feats,anchors,num_classes,input_shape)
    boxes=yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape)
    boxes=K.reshape(boxes,[-1,4])
    box_scores=box_confidence*box_class_prob
    box_scores=K.reshape(box_scores,[-1,num_classes])
    return boxes,box_scores

def yolo_eval(yolo_outputs,anchors,num_classes,image_shape,max_boxes=20,score_threshold=0.6,iou_threshold=0.5):
    num_layers=len(yolo_outputs)
    anchor_mask=[[6,7,8],[3,4,5],[0,1,2]] if num_layers==3 else [[3,4,5],[1,2,3]]
    input_shape=K.shape(yolo_outputs[0])[1:3]*32
    boxes=[]
    box_scores=[]
    for l in range(num_layers):
        _boxes,_box_score=yolo_boxes_and_scores(yolo_outputs[l],anchors[anchor_mask[l]],num_classes,input_shape,image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_score)
    boxes=K.concatenate(boxes,axis=0)
    boxes=K.concatenate(boxes,axis=0)
    mask=box_scores>=score_threshold
    max_boxes_tensor=K.constant(max_boxes,dtype="int32")
    boxes_=[]
    scores_=[]
    classes_=[]
    for c in range(num_classes):
        class_boxes=K.tf.boolean_mask(boxes,mask[:,c])
        class_box_scores=K.tf.boolean_mask(box_scores[:,c],mask[:,c])
        nms_index=K.tf.image.non_max_suppression(class_boxes,class_box_scores,iou_threshold=iou_threshold)
        class_boxes=K.gather(class_boxes,nms_index)
        class_box_scores=K.gather(class_box_scores,nms_index)
        classes=K.ones_like(class_box_scores,"int32")*c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_=K.concatenate(boxes_,axis=0)
    scores_=K.concatenate(scores_,axis=0)
    classes_=K.concatenate(classes_,axis=0)
    return boxes_,scores_,classes_


def preprocess_true_boxes(true_boxes,input_shape,anchors,num_classes):
    '''
    preprocess true boxes
    :param true_boxes: (m,T,5) m bacth_size,w*h*num_anchors,5 5=>x_min,y_min,x_max,y_max,class_id
    :param input_shape:
    :param anchors:(N,2)
    :param num_classes:int
    :return:
    '''
    assert (true_boxes[...,4]<num_classes).all()

    num_layers=len(anchors)/3
    anchor_mask=[[6,7,8],[3,4,5],[0,1,2]] if num_layers==3 else [[3,4,5],[1,2,3]]

    true_boxes=np.array(true_boxes,dtype="float32")
    input_shape=np.array(input_shape,dtype="int32")
    boxes_xy=(true_boxes[...,0:2]+true_boxes[...,2:4])//2
    boxes_wh=true_boxes[...,2:4]-true_boxes[...,0:2]
    true_boxes[...,0:2]=boxes_xy/input_shape[::-1]
    true_boxes[...,2:4]=boxes_wh/input_shape[::-1]

    m=true_boxes.shape[0]

    grid_shape=[input_shape//{0:32,1:16,2:8}[l] for l in range(num_layers)]
    y_true=[np.zeros((m,grid_shape[l][0],grid_shape[l][1],len(anchor_mask[l]),5+num_classes),"float32") for l in range(num_layers)]

    anchors=np.expand_dims(anchors,0)
    anchor_maxes=anchors/2.
    anchors_mines=-anchor_maxes
    valid_mask=boxes_wh[...,0]>0
    for b in range(m):
        wh=boxes_wh[b,valid_mask[b]]
        if len(wh)==0:continue
        wh=np.expand_dims(wh,-2)
        box_maxes=wh/2.
        box_min=-box_maxes

        intersect_mins=np.maximum(box_min,anchors_mines)
        intersect_maxes=np.minimum(box_maxes,anchor_maxes)
        intersect_wh=np.maximum(0.,intersect_maxes-intersect_mins)
        intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
        box_area=wh[...,0]*wh[...,1]
        anchor_area=anchors[...,0]*anchors[...,1]
        iou=intersect_area/(box_area+anchor_area-intersect_area)
        best_anchor=np.argmax(iou,axis=-1)

        for t,n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i=np.floor(true_boxes[b,t,0]*grid_shape[l][1]).astype("int32")
                    j=np.floor(true_boxes[b,t,1]*grid_shape[l][0]).astype("int32")
                    k=anchor_mask[l].index(n)
                    c=true_boxes[b,t,4].astype("int32")
                    y_true[l][b,j,i,k,0:4]=true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4]=1
                    y_true[l][b,j,i,k,5+c]=1

    return y_true

def box_iou(b1,b2):
    '''

    :param b1:(i1,i2,i3,...,4) xywh
    :param b2:(j,4) xywh
    :return:
    '''


def yolo_loss(args,anchors,num_classes,ignore_threshold=0.5,print_loss=False):
    '''

    :param args: yolo_outputs,y_true
    :param anchors:(N,2)
    :param num_classes: int
    :param ignore_threshold: float
    :param print_loss: boolean
    :return: loss_
    '''
    num_layers=len(anchors)//3
    yolo_outputs=args[:num_layers]
    y_true=args[num_layers:]
    anchor_mask=[[6,7,8],[3,4,5],[0,1,2]] if num_layers==3 else [[3,4,5],[1,2,3]]
    input_shape=K.cast(K.shape(yolo_outputs[0])[1:3]*32,K.dtype(y_true[0]))
    grid_shapes=[K.cast(K.shape(yolo_outputs[l])[1:3],K.dtype(y_true[0])) for l in range(num_layers)]
    loss=0
    m=K.shape(yolo_outputs[0])[0]
    mf=K.cast(m,K.dtype(yolo_outputs[0]))
    for l in range(num_layers):
        object_mask=y_true[l][...,4:5]
        true_class_probs=y_true[l][...,5:]
        grid,raw_pred,pred_xy,pred_wh=yolo_head(yolo_outputs[l],anchors[anchor_mask],num_classes,input_shape,calc_loss=True)
        pred_box=K.concatenate([pred_xy,pred_wh])
        raw_true_xy=y_true[l][...,0:2]*grid_shapes[l][::-1]-grid
        raw_true_wh=K.log(y_true[l][...,2:4]/anchors[anchor_mask[l]]*input_shape[::-1])
        raw_true_wh=K.switch(object_mask,raw_true_wh,K.zeros_like(raw_true_wh))
        box_loss_scale=2-y_true[l][...,2:3]*y_true[l][...,3:4]
        ignore_mask=K.tf.TensorArray(K.dtype(y_true[0]),size=1,dynamic_size=True)
        object_mask_bool=K.cast(object_mask,'bool')
        def loop_body(b,ignore_mask):
            true_box=K.tf.boolean_mask(y_true[l][b,...,0:4],object_mask_bool[b,...,0])
            iou=




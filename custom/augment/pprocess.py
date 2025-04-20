import numpy as np
import cv2
import sys
import random




#添加椒盐噪声
def sp_noise(image,boost=False):
      output = np.zeros(image.shape,np.uint8)
      prob=random.random()*0.0005+0.0001  #随机噪声比例
      thres = 1 - prob/5
      if boost:
          prob = random.random() * 0.005 + 0.001
          thres = 1 - prob
      map_sp=np.random.rand(image.shape[0],image.shape[1])
      image[map_sp< 4*prob]=0
      image[map_sp> thres]=255
    #   print("use sp_noise")
      return image


def gasuss_noise(image, mean=0.0, var=0.08):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var, image.shape)
    out = image + noise
    out[out<0]=0.0
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    return out
#旋转模糊
def apply_planar_rotation_blur(image, num_steps=10, max_angle=10):
    """
    模拟图像在图像平面中旋转造成的模糊（平面内旋转模糊）
    :param image: 原始图像
    :param num_steps: 模拟的旋转帧数（越多越平滑）
    :param max_angle: 最大旋转角度（度）
    :return: 模糊图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    acc = np.zeros_like(image, dtype=np.float32)

    for i in range(num_steps):
        angle = -max_angle / 2 + i * (max_angle / (num_steps - 1))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc += rotated.astype(np.float32)

    # 平均融合
    acc /= num_steps
    return np.uint8(np.clip(acc, 0, 255))

prob_crop=0.95

prob_stay=0.1
prob_noise=0.6
prob_blur=0.25
prob_spnoise=0.05

prob_noise_stay=0.05
prob_noise_sp=0.15
prob_noise_gasuss=0.8
#旋转模糊
prob_rotation_blur=0.7


#主函数，根据概率选择对图像进行不同的处理
def process(data):
    img=data['img'].copy()
    img_flt=img.astype(np.double)/255
    img_flt=(img_flt**(0.8+1.7*random.random()))*255
    img_flt=img_flt.clip(0,255)
    img=img_flt.astype(np.uint8)
    r=random.random() 
    
    if r<prob_stay:
        img_n=img
    elif r<(prob_stay+prob_blur):
        r1=random.random()
        if r1<0.35:
            img_n=cv2.blur(img,(7,7))
        else:
            img_n=cv2.blur(img,(3,3))
        #加旋转模糊
        img_n=apply_planar_rotation_blur(img, num_steps=10, max_angle=10)
    elif r<(prob_stay+prob_blur+prob_noise):
        #img_n=np.zeros(img.shape)
        for i in range(0,3):
            r1=random.random()
            if r1<prob_noise_stay:
                img[...,i]=img[...,i]
                #cv2.imshow('stay',img_n[i])
            elif r1<(prob_noise_stay+prob_noise_sp):
                img[...,i]=sp_noise(img[...,i])
                #cv2.imshow('sp', img_n[i])
            else:
                img[...,i]=gasuss_noise(img[...,i])
                #cv2.imshow('gasuss', img_n[i])
        img_n=img
    else:
        img_n=sp_noise(img,boost=True)
    data['img']=img_n
    return data

#与 process 类似，用于分类任务
def process_cls(img):
    #print('==============================')
    #img=img.copy()
    img_flt=img.astype(np.double)
    img_flt=img_flt**(0.85+0.3*random.random())
    img_flt=img_flt.clip(0,255)
    img=img_flt.astype(np.uint8)
    r=random.random() 
    if r<prob_stay:
        img_n=img
    elif r<(prob_stay+prob_blur):
        r1=random.random()
        if r1<0.35:
            img_n=cv2.blur(img,(7,7))
        else:
            img_n=cv2.blur(img,(3,3))
        img_n=apply_planar_rotation_blur(img, num_steps=10, max_angle=10)
    elif r<(prob_stay+prob_blur+prob_noise):
        #img_n=np.zeros(img.shape)
        for i in range(0,3):
            r1=random.random()
            if r1<prob_noise_stay:
                img[...,i]=img[...,i]
                #cv2.imshow('stay',img_n[i])
            elif r1<(prob_noise_stay+prob_noise_sp):
                img[...,i]=sp_noise(img[...,i])
                #cv2.imshow('sp', img_n[i])
            else:
                img[...,i]=gasuss_noise(img[...,i])
                #cv2.imshow('gasuss', img_n[i])
        img_n=img
    else:
        img_n=sp_noise(img,boost=True)
    
    return img_n



prob_light=0.5
num_cut_min=2
num_cut_max=6
area_ratio=0.08
protect_lt=130
def high_light_cut(img):
    p=random.random()
    num_cut=random.randint(num_cut_min,num_cut_max)
    min_len=min(img.shape[0:1])
    r=int(min_len*area_ratio)//2
    if p<prob_light:
        xy_rec=[]
        img_max=np.max(img,axis=2)
        #_,img_max_thresh=cv2.threshold(img_max,254,255,cv2.THRESH_BINARY)
        img_max_thresh=np.zeros(img_max.shape)
        img_max_thresh[img_max==255]=255
        img_max_thresh=img_max_thresh[r:img.shape[0]-r,r:img.shape[1]-r]
        for i in range(num_cut):
            #img_max=np.max(img,axis=2)
            #_,img_max_thresh=cv2.threshold(img_max,254,255,cv2.THRESH_BINARY)
            #cv2.imshow('immax',img_max_thresh)
            #img_max_thresh=img_max_thresh[r:img.shape[0]-r,r:img.shape[0]-r]
            xy=np.stack(np.where(img_max_thresh==255),axis=1)
            for rec in xy_rec:
                x_ls=np.abs(xy[:,0]-rec[0])
                y_ls=np.abs(xy[:,1]-rec[1])
                xy=xy[(x_ls>2*r)*(y_ls>2*r),:]
            if xy.shape[0]<1:
                break
            index=random.randint(0, xy.shape[0]-1)
            x,y=xy[index,0]+r,xy[index,1]+r
            roi=img[x-r:x+r,y-r:y+r]
            x_low=x-2*r if x>2*r else 0
            y_low=y-2*r if y>2*r else 0
            x_max=x if x<img_max_thresh.shape[0] else img_max_thresh.shape[0]
            y_max=x if y<img_max_thresh.shape[0] else img_max_thresh.shape[1]
            img_max_thresh[x_low:x_max,y_low:y_max]=0
            #roi[roi<cut_off]=cut_off
            #roi-=cut_off
            cv2.threshold(roi,protect_lt,255,cv2.THRESH_TRUNC,roi)
            xy_rec.append([x,y])
    else:
        pass
    return img

num_log=0






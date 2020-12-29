import cv2 
import numpy as np

weights = r'/Users/upasanathakuria/Desktop/People-Counting-in-Real-Time/Detectx-Yolo-V3/yolov3.weights'
config1 = r'/Users/upasanathakuria/Desktop/People-Counting-in-Real-Time/Detectx-Yolo-V3/cfg/yolov3.cfg'
class_labels = r'/Users/upasanathakuria/Desktop/People-Counting-in-Real-Time/Detectx-Yolo-V3/data/coco.names'

iou_thresh = 0.4

with open(class_labels, 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]


def draw_boxes(img,boxes,classesIds,class_labels,confidences,idex):
    bboxs = []
    if idex is not None:
        for i in idex.flatten():
            x,y,w,h =boxes[i].astype("int")
            bboxs.append((x, y, w, h))

            label=class_labels[classesIds[i]]
            confidence = confidences[i]
            cv2.rectangle(img,(int(x-w/2),int(y-w/2)),(int(x+w/2),int(y+w/2)),(0,255,0),3)
            cv2.putText(img, str(label) + ':' + "{0:.2f}".format(confidence) , (int(x-w/2), int(y-w/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 4)
    return img, bboxs
            
def out_transformation(out,width, height,class_labels,label="person"):
    boxes=[]
    confidences=[]
    classesIds=[]
    for i in out:
        for k in i:
            
            scores=k[5:]
            classes=np.argmax(scores)
            confidence=scores[classes]
            if (confidence>0.4) and (class_labels[classes] == label)  :
                confidences.append(float(confidence))
                box=k[0:4]* np.array([width,height,width,height],dtype=int)
                
                boxes.append(box)                    
                classesIds.append(classes)
    return boxes,confidences,classesIds

def infer_image(net,layer_names,img,class_labels,width,height,iou_thresh):
    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True)
    net.setInput(blob)
    
    out=net.forward(layer_names)

    boxes, confidences, classesIds = out_transformation(
        out, width, height, class_labels)
    
    idex=cv2.dnn.NMSBoxes(boxes,confidences,0.5,iou_thresh)
    idex = np.array(idex)

    img, bboxs=draw_boxes(img,boxes,classesIds,class_labels,confidences,idex)
    return img, bboxs

net=cv2.dnn.readNet(weights, config1)

layer_names = net.getLayerNames()
layer_names=[layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if __name__ == "__main__":

    cam=cv2.VideoCapture(0)
    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    width=int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writter=cv2.VideoWriter('output.avi',fourcc,30,(width,height),True)



    while cam.isOpened():
        _,frame=cam.read()
        frame , bboxs =infer_image(net,layer_names,frame,class_labels,width,height,iou_thresh)
        writter.write(frame)
        cv2.imshow('output',frame)
        if cv2.waitKey(10) & 0xFF==27:
            break
    cam.release()
    writter.release()
    cv2.destroyAllWindows()
            

import cv2

def detect_live(model, device=0, score_filter=0.8, image = None):
    frame = image

    scale_percent = 2
    # calculate the 50 percent of original dimensions
    src_dims=frame.shape
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    frame=cv2.resize(frame,(width,height),interpolation=cv2.INTER_LANCZOS4)
    labels, boxes, scores = model.predict(frame)
    #print('sssssss')
    # Plot each box with its label and score
    for i in range(boxes.shape[0]):

        if scores[i] < score_filter:
            continue

        box = boxes[i]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
        if labels:
            cv2.putText(frame, '{}: {}'.format(labels[i], round(scores[i].item(), 2)), (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    frame = cv2.resize(frame, (src_dims[1], src_dims[0]),interpolation=cv2.INTER_LANCZOS4)

    return frame


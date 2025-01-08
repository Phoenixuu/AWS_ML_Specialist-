import cv2
from scrfd import SCRFD

if __name__ == "__main__":
    model_file = 'path_to_model.onnx'
    image_file = 'path_to_image.jpg'

    # Create SCRFD instance
    scrfd = SCRFD(model_file=model_file)

    # Load image
    img = cv2.imread(image_file)

    # Perform detection
    det, kps = scrfd.detect(img)

    for img_path in img_paths:
        img = cv2.imread(img_path)

        ta = time.time()
        bboxes, kpss = detector.detect(img, 0.4, input_size=(640, 640))
        tb = time.time()
        print('all cost:', tb - ta)
        print(img_path, bboxes.shape)
        if kpss is not None:
            print(kpss.shape)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1, y1, x2, y2, score = bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(int)
                    cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
        filename = img_path.split('/')[-1]
        print('output:', filename)
        cv2.imwrite(f'../data/outputs/{filename}', img)

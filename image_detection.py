import cv2


def create_test_data(test, model, image):
    negative = []
    for i in test:
        for j in i[1]:
            if j < 0:
                negative.append(i)
    test_data = []
    for j in test:
        if j not in negative:
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

            img = img[j[1][1]:j[1][1] + j[1][3], j[1][0]:j[1][0] + j[1][2]]
            new_img = cv2.resize(img, (50, 50))
            new_img = new_img.reshape(-1, 50, 50, 1)
            predict = model.predict(new_img)
            test_data.append([j, predict])

    return test_data



import streamlit as st
from PIL import Image
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from image_detection import create_test_data

st.set_page_config(page_title='Face Mask Detector', page_icon='🤯', layout='centered', initial_sidebar_state='expanded')

model = tf.keras.models.load_model('model/')
detector = MTCNN()
options = ['face_no_mask', 'face_with_mask']


def mask_image():
    path_img = './images/out.jpg'

    image = cv2.imread(path_img)
    img = image
    faces = detector.detect_faces(img)

    test = []

    for face in faces:
        bounding_box = face['box']
        test.append([image, bounding_box])

    print(test)

    test_data = create_test_data(test, model, path_img)
    final_pred = []
    k = 0

    for i, j in test_data:
        no_mask, with_mask = j[0][0], j[0][1]
        bounding_box = faces[k]['box']
        color = (0, 255, 0) if with_mask > no_mask else (0, 0, 255)
        cv2.rectangle(image, (bounding_box[0], bounding_box[1]),
                    (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                    color=color, thickness=2)
        k += 1
        final_pred.append('Face with Mask' if with_mask > no_mask else 'Face without Mask')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, final_pred


def mask_detection():
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Please, choose the detection method :)")
    choice = st.sidebar.selectbox(".. from this options:", activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Image Detection</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            im = our_image.save('./images/out.jpg')
            saved_image = st.image(image_file, caption='', use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                image, prediction = mask_image()
                st.text(prediction)
                st.image(image, use_column_width=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Webcam Detection</h2>', unsafe_allow_html=True)
        st.markdown("### Click here ⬇")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            our_image = Image.fromarray(frame, 'RGB')
            im = our_image.save('./images/out.jpg')
            image, prediction = mask_image()
            st.text(prediction)
            st.image(image, use_column_width=True)

        else:
            st.write('Stopped')


mask_detection()

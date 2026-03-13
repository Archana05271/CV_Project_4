import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="AR Filter System", page_icon="🎭")

st.title("🎭 AR Filters & Image Processing")

mode = st.sidebar.selectbox(
    "Select Feature",
    ["AR Face Filters", "Image URL Filter Processor"]
)

# ---------- Overlay Function ----------
def overlay_filter(background, overlay, x, y):

    h, w = overlay.shape[:2]

    if x >= background.shape[1] or y >= background.shape[0]:
        return background

    h = min(h, background.shape[0] - y)
    w = min(w, background.shape[1] - x)

    overlay = overlay[0:h, 0:w]

    if overlay.shape[2] == 4:

        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.dstack((alpha, alpha, alpha))

        background_region = background[y:y+h, x:x+w]

        blended = alpha * overlay_rgb + (1 - alpha) * background_region

        background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background


# =====================================
# FEATURE 1 : AR FACE FILTERS
# =====================================

if mode == "AR Face Filters":

    st.header("🎭 AR Face Filters")

    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg","png","jpeg"])

    filter_option = st.selectbox(
        "Choose Filter",
        ["None","Sunglasses","Hat","Mustache","Crown"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        img = np.array(image)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces) == 0:
            st.warning("No face detected in the image")

        for (x,y,w,h) in faces:

            if filter_option == "Sunglasses":

                overlay = cv2.imread("filters/sunglasses.png", cv2.IMREAD_UNCHANGED)

                if overlay is None:
                    st.error("Sunglasses image not found")
                else:
                    overlay = cv2.resize(overlay,(w,int(h/3)))
                    img = overlay_filter(img,overlay,x,y+int(h/4))

            elif filter_option == "Hat":

                overlay = cv2.imread("filters/hat.png", cv2.IMREAD_UNCHANGED)

                if overlay is None:
                    st.error("Hat image not found")
                else:
                    overlay = cv2.resize(overlay,(w,int(h/2)))
                    img = overlay_filter(img,overlay,x,max(0,y-int(h/2)))

            elif filter_option == "Mustache":

                overlay = cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED)

                if overlay is None:
                    st.error("Mustache image not found")
                else:
                    overlay = cv2.resize(overlay,(int(w/2),int(h/6)))
                    img = overlay_filter(img,overlay,x+int(w/4),y+int(h/2))

            elif filter_option == "Crown":

                overlay = cv2.imread("filters/crown.png", cv2.IMREAD_UNCHANGED)

                if overlay is None:
                    st.error("Crown image not found")
                else:
                    overlay = cv2.resize(overlay,(w,int(h/2)))
                    img = overlay_filter(img,overlay,x,max(0,y-int(h/2)))

        st.image(img, channels="BGR")


# =====================================
# FEATURE 2 : IMAGE URL PROCESSING
# =====================================

elif mode == "Image URL Filter Processor":

    st.header("🖼️ Image Processing Filters")

    image_url = st.text_input("Enter Image URL")

    filter_option = st.selectbox(
        "Choose Filter",
        ["None","Blur","Edge Detection","Contour Detection","Grayscale"]
    )

    if st.button("Process Image"):

        if image_url == "":
            st.warning("Please enter an image URL")

        else:

            try:

                response = requests.get(image_url)

                if response.status_code != 200:
                    st.error("Unable to load image from URL")

                else:

                    img = Image.open(BytesIO(response.content))
                    img = np.array(img)

                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if filter_option == "Blur":
                        img = cv2.GaussianBlur(img,(9,9),0)

                    elif filter_option == "Edge Detection":
                        img = cv2.Canny(img,100,200)

                    elif filter_option == "Contour Detection":

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray,100,200)

                        contours,_ = cv2.findContours(
                            edges,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        img = cv2.drawContours(img,contours,-1,(0,255,0),2)

                    elif filter_option == "Grayscale":
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    st.subheader("Processed Image")

                    if len(img.shape) == 2:
                        st.image(img, clamp=True)
                    else:
                        st.image(img, channels="BGR")

            except:
                st.error("Invalid image URL")
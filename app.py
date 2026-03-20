from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Pequeña mejora visual
st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #6b7280;
            margin-bottom: 1.2rem;
        }
        .stMetric {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 8px;
            background-color: rgba(255,255,255,0.02);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def process_image(model, image_bytes, conf_threshold, iou_threshold, max_det):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil_img)[..., ::-1]  # RGB -> BGR para YOLO

    results = model(
        np_img,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=int(max_det)
    )

    result = results[0]
    boxes = result.boxes
    annotated = result.plot()              # BGR numpy array
    annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB

    return pil_img, result, boxes, annotated_rgb

st.markdown('<div class="main-title">Detección de Objetos en Imágenes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Esta aplicación utiliza YOLOv5 para detectar objetos desde la cámara o desde una imagen subida.</div>',
    unsafe_allow_html=True
)

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("Parámetros")
        st.subheader("Configuración de detección")

        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

        st.markdown("---")
        source_option = st.radio(
            "Fuente de imagen",
            ["Usar cámara", "Subir imagen"]
        )

    image_bytes = None

    if source_option == "Usar cámara":
        picture = st.camera_input("Capturar imagen", key="camera")
        if picture:
            image_bytes = picture.getvalue()

    else:
        uploaded_file = st.file_uploader(
            "Sube una imagen",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file:
            image_bytes = uploaded_file.read()

    if image_bytes:
        with st.spinner("Detectando objetos..."):
            try:
                original_img, result, boxes, annotated_rgb = process_image(
                    model,
                    image_bytes,
                    conf_threshold,
                    iou_threshold,
                    max_det
                )
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        label_names = model.names
        category_count = {}
        category_conf = {}

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cat = int(box.cls.item())
                conf = float(box.conf.item())
                category_count[cat] = category_count.get(cat, 0) + 1
                category_conf.setdefault(cat, []).append(conf)

            data = [
                {
                    "Categoría": label_names[cat],
                    "Cantidad": count,
                    "Confianza promedio": round(float(np.mean(category_conf[cat])), 2)
                }
                for cat, count in category_count.items()
            ]

            df = pd.DataFrame(data).sort_values(by="Cantidad", ascending=False)

            total_detections = int(df["Cantidad"].sum())
            total_categories = int(df["Categoría"].nunique())
            avg_conf_global = round(df["Confianza promedio"].mean(), 2)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Detecciones totales", total_detections)
            with m2:
                st.metric("Categorías detectadas", total_categories)
            with m3:
                st.metric("Confianza promedio", avg_conf_global)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Imagen original")
                st.image(original_img, use_container_width=True)

            with col2:
                st.subheader("Imagen con detecciones")
                st.image(annotated_rgb, use_container_width=True)

            st.subheader("Resumen de objetos detectados")
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Categoría")["Cantidad"])

            annotated_pil = Image.fromarray(annotated_rgb)
            buffer = io.BytesIO()
            annotated_pil.save(buffer, format="PNG")
            st.download_button(
                label="Descargar imagen con detecciones",
                data=buffer.getvalue(),
                file_name="detecciones_yolo.png",
                mime="image/png"
            )

            with st.expander("Ver detalles técnicos"):
                st.write("Clases detectadas:", list(df["Categoría"]))
                st.write("Parámetros usados:")
                st.write({
                    "Confianza mínima": conf_threshold,
                    "Umbral IoU": iou_threshold,
                    "Detecciones máximas": int(max_det)
                })

        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Imagen original")
                st.image(original_img, use_container_width=True)

            with col2:
                st.subheader("Imagen procesada")
                st.image(annotated_rgb, use_container_width=True)

            st.info("No se detectaron objetos con los parámetros actuales.")
            st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")

else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("Acerca de la aplicación: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")

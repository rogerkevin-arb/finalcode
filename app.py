import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from skimage.morphology import skeletonize
import base64
from io import BytesIO
import gc

import tensorflow.keras.backend as K
import matplotlib.cm as cm

import hashlib
import psutil, os

import onnxruntime as ort

# =========================
# CONTROL DE MEMORIA
# =========================

def check_memory(limit_mb=2200):
    import psutil, os, gc
    import streamlit as st
    import tensorflow as tf

    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024**2

    if ram_mb > limit_mb:
        st.warning(f"🚨 Reiniciando app por exceso de memoria...")

        # 1. limpiar cachés de Streamlit
        st.cache_data.clear()
        st.cache_resource.clear()

        # 2. limpiar session state (MUY importante)
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # 3. limpiar TensorFlow
        tf.keras.backend.clear_session()

        # 4. garbage collector agresivo
        gc.collect()
        gc.collect()

        # 5. reinicio Streamlit
        st.rerun()


# ======== mostrar memoria ========
def mostrar_memoria():
    proceso = psutil.Process(os.getpid())
    ram_mb = proceso.memory_info().rss / 1024**2

    st.sidebar.write(f"🧠 RAM usada: {ram_mb:.0f} MB")

    if ram_mb > 1600:
        st.sidebar.error("⚠️ Muy cerca del límite")
    elif ram_mb > 600:
        st.sidebar.warning("Cuidado con la memoria")

    return ram_mb


# ======== Cargar modelos ========

@st.cache_resource
def cargar_detector_murosc():
    session = ort.InferenceSession(
        "best.onnx",
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name

@st.cache_resource
def cargar_clasificador_ladrillo():
    return load_model('model_CL.h5', safe_mode=False)

# ==================================================
# MENÚ PRINCIPAL
# ==================================================

mostrar_memoria()
check_memory(2200)

# ======== Interfaz ========

model_detector_murosc, input_name, output_name = cargar_detector_murosc()
model_clasificador_ladrillo = cargar_clasificador_ladrillo()

def detectar_muros(session, input_name, output_name, image, conf=0.25):

    img = image.astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))

    img = np.expand_dims(img, axis=0)

    outputs = session.run(
        [output_name],
        {input_name: img}
    )[0]

    print(outputs.shape)
    print(outputs[:, :, :5])

    outputs = np.squeeze(outputs)

    boxes = []
    scores = []

    for i in range(outputs.shape[1]):

        x = outputs[0,i]
        y = outputs[1,i]
        w = outputs[2,i]
        h = outputs[3,i]
        score = outputs[4,i]

        if score < conf:
            continue

        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2

        boxes.append([x1,y1,x2,y2])
        scores.append(float(score))

    if len(boxes)==0:
        return [],[]

    boxes_cv=[]

    for b in boxes:

        x1,y1,x2,y2=b

        boxes_cv.append([
            int(x1),
            int(y1),
            int(x2-x1),
            int(y2-y1)
        ])

    idx=cv2.dnn.NMSBoxes(
        boxes_cv,
        scores,
        conf,
        0.45
    )

    final_boxes=[]
    final_scores=[]

    if len(idx)>0:

        for i in idx.flatten():

            final_boxes.append(boxes[i])

            final_scores.append(scores[i])

    return final_boxes,final_scores
    

st.title("Detección, Conteo y Clasificación Automática de Muros Confinados con Unidades Tubulares")

st.markdown("""
### Instrucciones
1. En edificaciones grandes se recomienda el empleo de vehículos aéreos no tripulados (UAVs), para una captura  centrada y de buena resolución.
2. Se recomienda capturar únicamente la edificación de interés, minimizando la presencia de fondo de otras edificaciones en la imagen.
    - Se rellena con píxeles negros hacia ARRIBA o DERECHA hasta formar imagen cuadrada LxL.
    - Se aceptan imágenes en cualquier formato W:H.
    - Se redimensiona (1024x1024) y se procesa con YOLO 11 para detección de muros confinados en sus cuatro lados.
    - Las coordenadas de las cajas delimitadoras se reescalan a la imagen LxL para la extraccion de muros.
    - Se extraen los muros manteniendo la resolucion, se vuelven cuadradas y se redimensionan a 512x512 para su clasificación mediante un modelo CNN.
- Se identifican y se hace un conteo automático de los muros confinados con unidades tubulares (Norma E070).
""")

st.markdown("### Parámetros")
conf_yolo = st.slider("Umbral de confianza del detector YOLO11l",min_value=0.0,max_value=1.0, value=0.80,step=0.01)
umbral_clasificador = st.slider("Umbral del clasificador",min_value=0.0,max_value=1.0,value=0.50,step=0.01)
st.markdown("### Filtro de Muros Redundantes % (Opcional)")
porcentaje_minimo = st.slider("Eliminar detecciones con área menor al (%) del muro más grande",min_value=0,max_value=100,value=25,step=1)


uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","jpeg","png","JPG"])

if uploaded_file is not None:
    
    # detectar si es nueva imagen
    new_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    if "last_image_hash" not in st.session_state:
        st.session_state.last_image_hash = None
    if st.session_state.last_image_hash != new_hash:
        st.session_state.last_image_hash = new_hash
        # BORRAR TODO LO ANTERIOR
        gc.collect()
        plt.close("all")
        # borrar variables de ejecución anterior
        for k in list(st.session_state.keys()):
            if k != "last_image_hash":
                del st.session_state[k]
        # 3. limpiar TensorFlow (CLAVE)
        tf.keras.backend.clear_session()
        # 4. forzar garbage collector
        gc.collect()
        
    # =========================
    # 1. CARGA IMAGEN
    # =========================
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    img = np.array(image)
    h, w = img.shape[:2]

    # =========================
    # 2. PAD A CUADRADO LxL (ARRIBA o DERECHA)
    # =========================
    L = max(h, w)

    padded = np.zeros((L, L, 3), dtype=np.uint8)

    padded[L-h : L, 0 : w] = img

    st.image(padded, caption="Imagen cuadrada LxL", use_container_width=True)

    # =========================
    # 3. YOLO 1024x1024
    # =========================
    img_1024 = cv2.resize(padded, (1024, 1024), interpolation=cv2.INTER_AREA)

    # seguridad anti artefactos
    img_1024 = np.ascontiguousarray(img_1024)

    # =========================
    # 4. DETECCIÓN YOLO
    # =========================

    boxes_1024, conf_yolo_boxes = detectar_muros(
        model_detector_murosc,
        input_name,
        output_name,
        img_1024,
        conf_yolo
    )
    
    if len(boxes_1024) == 0:
    
        st.warning("No se detectaron muros confinados en la imagen.")
        st.image(
            img_1024,
            caption="Imagen procesada (sin detecciones)",
            use_container_width=True
        )
        st.stop()
    
    # =========================
    # 5. REESCALAR A LxL
    # =========================

    scale = L / 1024
    boxes_L = []
    for (x1, y1, x2, y2) in boxes_1024:
        boxes_L.append([
            x1 * scale,
            y1 * scale,
            x2 * scale,
            y2 * scale
        ])

    # =========================
    # FILTRAR POR ÁREA
    # =========================

    if len(boxes_L) > 0:

        areas = []

        for x1, y1, x2, y2 in boxes_L:
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)

        area_max = max(areas)
        umbral_area = area_max * (porcentaje_minimo / 100)

        boxes_L_filtradas = []
        boxes_1024_filtradas = []
        conf_yolo_filtradas = []

        for box_L, box_1024, conf, area in zip(
                boxes_L, boxes_1024, conf_yolo_boxes, areas):

            if area >= umbral_area:

                boxes_L_filtradas.append(box_L)
                boxes_1024_filtradas.append(box_1024)
                conf_yolo_filtradas.append(conf)

        boxes_L = boxes_L_filtradas
        boxes_1024 = boxes_1024_filtradas
        conf_yolo_boxes = conf_yolo_filtradas

    st.info(
        f"Muros detectados : {len(boxes_L)}"
    )

    # =========================
    # 6. EXTRAER PARCHES + CLASIFICAR
    # =========================
    labels = []

    for (x1, y1, x2, y2) in boxes_L:

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        crop = padded[y1:y2, x1:x2]

        if crop.size == 0:
            labels.append(0)
            continue

        # hacer cuadrado rellenando hacia ARRIBA o DERECHA
        h2, w2 = crop.shape[:2]
        Lc = max(h2, w2)

        square = np.zeros((Lc, Lc, 3), dtype=np.uint8)

        # ✔ anclado abajo-izquierda dentro del parche
        square[Lc-h2:Lc, 0:w2] = crop

        patch_512 = cv2.resize(square, (512, 512))

        pred = model_clasificador_ladrillo.predict(
            np.expand_dims(patch_512.astype(np.float32), axis=0),
            verbose=0
        )[0][0]

        labels.append(pred)

    # =========================
    # 7. DIBUJAR RESULTADO
    # =========================

    output = img_1024.copy()

    for i, (box, score, conf_det) in enumerate(zip(boxes_1024, labels, conf_yolo_boxes), start=1):
        
        x1, y1, x2, y2 = map(int, box)

        is_pandereta = score >= umbral_clasificador
        color = (150, 0, 0) if is_pandereta else (0, 150, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        # TEXTO MULTILÍNEA
        clase = "TubularBrick" if is_pandereta else "NoTubularBrick"

        lines = [
            f"Muro {i}",
            "Muro Confinado",
            f"Conf: {conf_det:.2f}",
            clase,
            f"Pred: {score:.2f}"
        ]

        cx = x1 + 10
        cy = y1 + 20

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for j, line in enumerate(lines):

            (text_w, text_h), baseline = cv2.getTextSize(
                line, font, font_scale, thickness
            )

            y_text = cy + j * (text_h + 6)

            # color del fondo = mismo color de la caja
            box_color = color

            # coordenadas del rectángulo de fondo
            cv2.rectangle(
                output,
                (cx - 2, y_text - text_h - 2),
                (cx + text_w + 2, y_text + baseline + 2),
                box_color,
                -1
            )

            # texto  encima
            cv2.putText(
                output,
                line,
                (cx, y_text),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
            
    st.image(output, caption="Resultado YOLO + Clasificación", use_container_width=True)

    st.markdown(f"**Resolución recibida:** {w} x {h}")
    st.markdown(f"**Resolución procesada:** {L} x {L}")

    # =========================
    # 8. LEYENDA
    # =========================

    total = len(labels)
    pandereta = sum([1 for s in labels if s >= umbral_clasificador])
    no_pandereta = total - pandereta

    relaciones_LH = []

    for (x1, y1, x2, y2) in boxes_L:

        ancho = x2 - x1
        alto = y2 - y1

        if alto > 0:
            relaciones_LH.append(ancho / alto)
        else:
            relaciones_LH.append(0)

    st.markdown("---")
    st.markdown("#### Resumen de Muros Confinados Detectados")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total de muros confinados", total)
    with col_b:
        st.metric("Muros con unidades Tubulares", pandereta)
    with col_c:
        st.metric("Muros sin unidades Tubulares", no_pandereta)

    st.markdown("---")

    # =========================
    # GRÁFICO CIRCULAR (CENTRADO)
    # =========================
    st.markdown("#### Diagrama Circular : Muros Detectados (%)")
    
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        fig_pie, ax_pie = plt.subplots(figsize=(7,5))

        ax_pie.pie(
            [pandereta, no_pandereta],
            labels=["Con U. Tubular", "Sin U. Tubular"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#FFA500", "#1f77b4"]
        )

        ax_pie.set_title("Diagrama Circular - Muros Detectados (%)")
        ax_pie.legend(labels=["Tubular", "Sin Tubular"], loc="upper right", bbox_to_anchor=(1.25, 1))
        st.pyplot(fig_pie)

    st.markdown("---")

    # =========================
    # GRÁFICO BARRAS (CENTRADO)
    # =========================
    st.markdown("#### Gráfico de barras : Relación Longitud - Altura (L/A)")
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        fig_bar, ax_bar = plt.subplots(figsize=(8,5))

        numeros_muro = np.arange(1, len(relaciones_LH)+1)

        ax_bar.bar(
            numeros_muro,
            relaciones_LH,
            color="#2ecc71"
        )

        ax_bar.set_xlabel("Número de Muro")
        ax_bar.set_ylabel("L/A")
        ax_bar.set_title("Relación Longitud/Altura por paño de muro")

        ax_bar.set_xticks(numeros_muro)

        for x, y in zip(numeros_muro, relaciones_LH):
            ax_bar.text(
                x,
                y,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        
        st.pyplot(fig_bar)

    # =========================
    # 9. GUIAS
    # =========================

    st.markdown("---")

    with st.expander("📚 Restricciones del uso de Unidades de Albañileria en Muros Confinados según Zona Sísmica."):
      st.image("tabla3.png", caption="Restricciones del uso de Unidades de Albañileria en Muros Confinados según Zona Sísmica",use_container_width=True)

    # =========================
    # 9. VISUALIZACIÓN DESLIZABLE DE MUROS
    # =========================
    st.markdown("## Detalle de Muros Confinados Detectados")

    with st.container():
        for i, (box, score, conf) in enumerate(zip(boxes_L, labels, conf_yolo_boxes), start=1):
        
            x1, y1, x2, y2 = map(int, box)
            crop = padded[y1:y2, x1:x2]
        
            if crop.size == 0:
                continue
        
            # Relación L/A
            ancho = x2 - x1
            alto = y2 - y1
            relacion = (ancho / alto) if alto > 0 else 0
        
            # Clasificación
            es_tubular = score >= umbral_clasificador
        
            clase_texto = (
                "Muro confinado con unidades tubulares (Pandereta)"
                if es_tubular
                else "Muro confinado sin unidades tubulares"
            )
        
            st.markdown(
                f"""
                ### Muro {i}
        
                **Tipo:** {clase_texto}  
                **Score clasificación:** `{score:.3f}`  
                **Confianza YOLO:** `{conf:.3f}`  
                **Relación L/A:** `{relacion:.2f}`
                """
            )
        
            st.image(crop, width=800)


    st.markdown("---")
    st.markdown("#### Referencias usadas para la tabla y YOLO11l :")
    st.markdown("""
    1. C. y S. Ministerio de Vivienda, “Norma Técnica E.070 Albañilería,” Lima, Perú, 2020. 
    2. Ultralytics YOLO11 | Ultralytics Docs,” Ultralytics Docs. 
    """)

    image.close()   
    del image
    del img
    del padded
    del img_1024
    del output    
    del boxes_1024
    del boxes_L
    del conf_yolo_boxes 
    del labels
    del relaciones_LH
    gc.collect()
        
        

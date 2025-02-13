import streamlit as st
import joblib
import numpy as np

# Cargar el modelo y el escalador
scaler = joblib.load("escalador.bin")
model = joblib.load("modelo.km.bin")

# Configurar la página
st.title("Asistente Cardíaco")
st.write("**Autor: Alfredo Diaz**")
st.write("Esta aplicación permite predecir si una persona tiene problemas cardíacos en base a su edad y nivel de colesterol.")

# Crear pestañas
input_tab, output_tab = st.tabs(["Ingresar Datos", "Resultado"])

with input_tab:
    st.header("Ingrese los datos del paciente")
    edad = st.slider("Edad", min_value=18, max_value=80, value=30, step=1)
    colesterol = st.slider("Colesterol", min_value=100, max_value=600, value=200, step=1)
    
    # Botón para predecir
    if st.button("Predecir", key="predict_button"):
        st.session_state["predict"] = True
        st.session_state["input_data"] = np.array([[edad, colesterol]])

with output_tab:
    st.header("Resultado del Diagnóstico")
    if st.session_state.get("predict", False):
        input_data = st.session_state["input_data"]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        if prediction == 1:
            st.error("El paciente tiene problemas cardíacos.")
            st.image("https://www.clikisalud.net/wp-content/uploads/2018/09/problemas-cardiacos-jovenes.jpg", caption="Precaución: Consulte a un especialista")
        else:
            st.success("El paciente no tiene problemas cardíacos.")
            st.image("https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg", caption="Siga cuidando su salud")

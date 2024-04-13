import streamlit as st
from clase.estructura_logica import LogicaProposicional
from clase.perceptron_simple import Perceptron


# Función para mostrar resultados en Streamlit
def mostrar_resultados(entradas, perceptron, operacion):
    st.write(f"### Resultados para la operación {operacion}")
    for entrada in entradas:
        prediccion = perceptron.predecir(entrada)
        resultado = "✔" if prediccion == 1 else "✘"
        color = "green" if prediccion == 1 else "red"
        # Utilizando HTML directamente para controlar el estilo.
        st.markdown(
            f"Entrada: {entrada} -> {prediccion} <span style='color: {color};'>{resultado}</span>",
            unsafe_allow_html=True,
        )


# Crear instancias de Perceptrón y cargar datos
perceptron_and = Perceptron()
perceptron_or = Perceptron()

entrada_and = LogicaProposicional(x1=[[0, 0], [0, 1], [1, 0], [1, 1]], x2=[0, 0, 0, 1])
entrada_or = LogicaProposicional(x1=[[0, 0], [0, 1], [1, 0], [1, 1]], x2=[0, 1, 1, 1])

# Entrenamiento de los modelos
perceptron_and.entrenar(entrada_and.x1, entrada_and.x2)
perceptron_or.entrenar(entrada_or.x1, entrada_or.x2)

# Configuración de la interfaz de Streamlit
st.title("Demostración del Perceptrón: Operaciones Lógicas")

st.write("## Entrenamiento para AND")
mostrar_resultados(entrada_and.x1, perceptron_and, "AND")

st.write("## Entrenamiento para OR")
mostrar_resultados(entrada_or.x1, perceptron_or, "OR")

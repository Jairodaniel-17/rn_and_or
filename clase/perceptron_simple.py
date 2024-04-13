from random import random


class Perceptron:
    def __init__(self, tasa_aprendizaje=0.05, n_entradas=2):
        self.pesos = [random() for _ in range(n_entradas)]
        self.umbral = random()
        self.tasa_aprendizaje = tasa_aprendizaje

    def funcion_activacion(self, suma_ponderada):
        return 1 if suma_ponderada >= 0 else 0

    def predecir(self, entradas):
        suma_ponderada = (
            sum(peso * entrada for peso, entrada in zip(self.pesos, entradas))
            - self.umbral
        )
        return self.funcion_activacion(suma_ponderada)

    def entrenar(self, datos_entrenamiento, etiquetas):
        for _ in range(100):  # Número de épocas
            for entrada, etiqueta in zip(datos_entrenamiento, etiquetas):
                prediccion = self.predecir(entrada)
                error = etiqueta - prediccion
                for i in range(len(self.pesos)):
                    self.pesos[i] += self.tasa_aprendizaje * error * entrada[i]
                self.umbral -= self.tasa_aprendizaje * error

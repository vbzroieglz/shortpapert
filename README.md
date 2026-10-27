# XGBoost, LightGBM y CatBoost con SHAP y LIME para diabetes tipo 2 (DiaHealth)

Código y resultados del short paper *"Comparación de XGBoost, LightGBM y CatBoost con SHAP y LIME para la Clasificación de Diabetes Tipo 2 bajo un Protocolo sin Fuga de Información: Estudio Empírico sobre DiaHealth"* (Grupo 6, E.P. de Ingeniería de Sistemas, Universidad Peruana Unión).

## Datos
DiaHealth (Prama et al., Mendeley Data, 2024, doi:[10.17632/7m7555vgrn.1](https://doi.org/10.17632/7m7555vgrn.1), licencia CC BY 4.0). El CSV **no se incluye** en este repositorio: descárgalo de Mendeley (`Diabetes_Final_Data_V2.csv`) y colócalo junto al cuaderno.

## Contenido
- `Pipeline_v2_DiaHealth.ipynb`: cuaderno ejecutado (Google Colab).
- `resultados/`: tablas T0–T9 y `resumen_paper.json` generados por el cuaderno. `T3_calculada_desde_T2a.csv` contiene las pruebas de Friedman y Wilcoxon-Holm calculadas a partir de los resultados por pliegue (T2a).
- `figuras/`: curvas ROC/PR, matrices de confusión, SHAP de los tres modelos y LIME de CatBoost.

## Protocolo (resumen)
Reglas de calidad R1–R4 → partición estratificada 80/20 (semilla 42) → *pipeline* imputación por mediana + SMOTE-NC + modelo, ajustado solo con entrenamiento → búsqueda aleatoria (25 intentos, CV de 5 pliegues, AUC-ROC) → CV estratificada de 10 pliegues → umbral de tamizaje (sensibilidad ≥ 0.80 en predicciones fuera de pliegue, CV de 5 pliegues) → evaluación en la prueba intacta (IC 95% por bootstrap, DeLong + Holm) → SHAP (TreeSHAP) y LIME.

## Reproducir
```bash
pip install -r requirements.txt
```
Abrir el cuaderno y ejecutar todas las celdas (semilla 42). Versiones usadas: Python 3.13, ver `requirements.txt`.

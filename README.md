# Actividad 2 - Clasificacion de activos financieros

Aplicacion en Streamlit para una actividad de master sobre aprendizaje supervisado aplicado a mercados financieros.

## Que hace

- Descarga datos historicos con `yfinance`
- Construye una variable objetivo binaria:
  - `1` si la rentabilidad del siguiente periodo es positiva
  - `0` si la rentabilidad del siguiente periodo es negativa o cero
- Genera variables explicativas tecnicas y de mercado
- Entrena dos clasificadores:
  - Regresion Logistica
  - Random Forest
- Evalua:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Matriz de confusion
- Muestra interpretacion automatica de resultados

## Como ejecutarlo en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue sin servidor propio con Streamlit Community Cloud

1. Crea una cuenta en GitHub.
2. Crea un repositorio nuevo y sube estos tres archivos:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Crea una cuenta en Streamlit Community Cloud.
4. Conecta tu cuenta de GitHub.
5. Selecciona el repositorio y despliega la app.
6. Comparte la URL publica con tu profesor.

## Tickers de ejemplo

- Banco Santander: `SAN.MC`
- Inditex: `ITX.MC`
- Iberdrola: `IBE.MC`
- BBVA: `BBVA.MC`
- IBEX 35: `^IBEX`
- S&P 500: `^GSPC`

## Consejo academico

Para que el trabajo quede mas solido, ejecuta varios activos, compara resultados y escoge uno para el informe final. No presentes el modelo como una maquina de ganar dinero, sino como una herramienta de apoyo a la decision.

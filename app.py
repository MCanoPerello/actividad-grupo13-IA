import warnings
warnings.filterwarnings('ignore')

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title='Clasificacion de activos financieros', layout='wide')


@st.cache_data(show_spinner=False)
def descargar_datos(ticker: str, benchmark: str, inicio: str, fin: str):
    activo = yf.download(ticker, start=inicio, end=fin, auto_adjust=True, progress=False)
    indice = yf.download(benchmark, start=inicio, end=fin, auto_adjust=True, progress=False)
    return activo, indice


def rsi(series: pd.Series, periodo: int = 14) -> pd.Series:
    delta = series.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    media_ganancia = ganancia.ewm(alpha=1 / periodo, adjust=False).mean()
    media_perdida = perdida.ewm(alpha=1 / periodo, adjust=False).mean()
    rs = media_ganancia / media_perdida.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def preparar_dataset(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    bench = benchmark_df.copy()

    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    bench.columns = [c[0] if isinstance(c, tuple) else c for c in bench.columns]

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    bench = bench[['Close']].rename(columns={'Close': 'Bench_Close'})

    data['ret_1'] = data['Close'].pct_change(1)
    data['ret_5'] = data['Close'].pct_change(5)
    data['ret_10'] = data['Close'].pct_change(10)
    data['ret_20'] = data['Close'].pct_change(20)
    data['vol_chg_1'] = data['Volume'].pct_change(1)
    data['vol_chg_5'] = data['Volume'].pct_change(5)

    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['sma_ratio_5_20'] = data['sma_5'] / data['sma_20'] - 1

    data['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']

    data['rsi_14'] = rsi(data['Close'], 14)
    data['volatility_10'] = data['ret_1'].rolling(10).std()
    data['volatility_20'] = data['ret_1'].rolling(20).std()
    data['range_intraday'] = (data['High'] - data['Low']) / data['Close']

    bench['bench_ret_1'] = bench['Bench_Close'].pct_change(1)
    bench['bench_ret_5'] = bench['Bench_Close'].pct_change(5)
    bench['bench_ret_20'] = bench['Bench_Close'].pct_change(20)

    data = data.join(bench[['bench_ret_1', 'bench_ret_5', 'bench_ret_20']], how='left')

    data['target'] = (data['Close'].shift(-1) / data['Close'] - 1 > 0).astype(int)
    data['next_return'] = data['Close'].shift(-1) / data['Close'] - 1

    data = data.dropna().copy()
    return data


def dividir_temporal(data: pd.DataFrame, train_pct: float = 0.8):
    corte = int(len(data) * train_pct)
    train = data.iloc[:corte].copy()
    test = data.iloc[corte:].copy()
    return train, test


def evaluar_modelo(nombre: str, modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    if hasattr(modelo, 'predict_proba'):
        prob = modelo.predict_proba(X_test)[:, 1]
    else:
        prob = None

    resultado = {
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, prob) if prob is not None and len(np.unique(y_test)) > 1 else np.nan,
        'Matriz': confusion_matrix(y_test, pred),
        'Predicciones': pred,
        'Probabilidades': prob,
        'Reporte': classification_report(y_test, pred, digits=4, zero_division=0),
        'ModeloEntrenado': modelo,
    }
    return resultado


def importancia_random_forest(modelo, columnas):
    estimador = modelo.named_steps['clf']
    return pd.DataFrame(
        {'Variable': columnas, 'Importancia': estimador.feature_importances_}
    ).sort_values('Importancia', ascending=False)


def coeficientes_logistic(modelo, columnas):
    estimador = modelo.named_steps['clf']
    return pd.DataFrame(
        {'Variable': columnas, 'Coeficiente': estimador.coef_[0]}
    ).assign(Abs=lambda x: x['Coeficiente'].abs()).sort_values('Abs', ascending=False)


def generar_interpretacion(metricas: pd.DataFrame, imp_rf: pd.DataFrame, coefs_log: pd.DataFrame) -> str:
    ganador = metricas.sort_values(['F1', 'ROC_AUC', 'Accuracy'], ascending=False).iloc[0]
    texto = []
    texto.append(
        f"El modelo con mejor equilibrio predictivo en el conjunto de prueba es {ganador['Modelo']}, "
        f"con F1={ganador['F1']:.3f}, precision={ganador['Precision']:.3f}, recall={ganador['Recall']:.3f} "
        f"y accuracy={ganador['Accuracy']:.3f}."
    )

    top_rf = ', '.join(imp_rf.head(5)['Variable'].tolist())
    texto.append(
        f"En Random Forest, las variables más influyentes son: {top_rf}. "
        "Eso sugiere que el modelo está captando sobre todo inercia reciente, tendencia y volatilidad."
    )

    top_log_pos = coefs_log[coefs_log['Coeficiente'] > 0].head(3)['Variable'].tolist()
    top_log_neg = coefs_log[coefs_log['Coeficiente'] < 0].head(3)['Variable'].tolist()
    texto.append(
        f"En la Regresión Logística, las variables con efecto positivo más claro sobre la probabilidad de subida son: {', '.join(top_log_pos) if top_log_pos else 'ninguna dominante'}. "
        f"Las que empujan hacia la clase de bajada son: {', '.join(top_log_neg) if top_log_neg else 'ninguna dominante'}."
    )

    texto.append(
        "Desde una óptica financiera, este tipo de clasificación puede ser útil como filtro táctico de apoyo, "
        "pero no debería tomarse como una señal autónoma de inversión. Conviene combinarla con control de costes, "
        "gestión de riesgo y validación fuera de muestra."
    )
    return '\n\n'.join(texto)


st.title('Actividad 2 - Aprendizaje supervisado: clasificacion')
st.markdown(
    'App para descargar datos de mercado, construir variables tecnicas y comparar dos modelos de clasificacion: **Regresion Logistica** y **Random Forest**.'
)

with st.sidebar:
    st.header('Parametros')
    ticker = st.text_input('Ticker del activo', value='SAN.MC')
    benchmark = st.text_input('Ticker del indice de referencia', value='^IBEX')
    inicio = st.date_input('Fecha inicio', value=date(2018, 1, 1))
    fin = st.date_input('Fecha fin', value=date(2025, 12, 31))
    train_pct = st.slider('Porcentaje para entrenamiento', min_value=0.6, max_value=0.9, value=0.8, step=0.05)
    ejecutar = st.button('Ejecutar analisis', type='primary')

st.info(
    'La variable objetivo es binaria: 1 si la rentabilidad del siguiente periodo es positiva; 0 si es negativa o cero.'
)

if ejecutar:
    try:
        activo, indice = descargar_datos(ticker, benchmark, str(inicio), str(fin))

        if activo.empty:
            st.error('No se han descargado datos para el ticker del activo. Revisa el simbolo introducido.')
            st.stop()
        if indice.empty:
            st.error('No se han descargado datos para el indice de referencia. Revisa el simbolo introducido.')
            st.stop()

        data = preparar_dataset(activo, indice)

        if len(data) < 120:
            st.error('Hay muy pocas observaciones tras construir las variables. Amplia el rango temporal.')
            st.stop()

        columnas_features = [
            'ret_1', 'ret_5', 'ret_10', 'ret_20',
            'vol_chg_1', 'vol_chg_5',
            'sma_ratio_5_20', 'macd', 'macd_signal', 'macd_hist',
            'rsi_14', 'volatility_10', 'volatility_20', 'range_intraday',
            'bench_ret_1', 'bench_ret_5', 'bench_ret_20'
        ]

        train, test = dividir_temporal(data, train_pct=train_pct)
        X_train, y_train = train[columnas_features], train['target']
        X_test, y_test = test[columnas_features], test['target']

        logistic = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced')),
        ])

        rf = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced_subsample',
            )),
        ])

        res_log = evaluar_modelo('Regresion Logistica', logistic, X_train, y_train, X_test, y_test)
        res_rf = evaluar_modelo('Random Forest', rf, X_train, y_train, X_test, y_test)

        metricas = pd.DataFrame([
            {k: v for k, v in res_log.items() if k in ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']},
            {k: v for k, v in res_rf.items() if k in ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']},
        ])

        imp_rf = importancia_random_forest(res_rf['ModeloEntrenado'], columnas_features)
        coefs_log = coeficientes_logistic(res_log['ModeloEntrenado'], columnas_features)
        interpretacion = generar_interpretacion(metricas, imp_rf, coefs_log)

        c1, c2, c3 = st.columns(3)
        c1.metric('Observaciones totales', len(data))
        c2.metric('Train', len(train))
        c3.metric('Test', len(test))

        st.subheader('Resumen de datos')
        resumen = pd.DataFrame({
            'Periodo': [f"{data.index.min().date()} a {data.index.max().date()}"],
            'Pct clase 1 (sube)': [round(data['target'].mean() * 100, 2)],
            'Pct clase 0 (baja)': [round((1 - data['target'].mean()) * 100, 2)],
        })
        st.dataframe(resumen, use_container_width=True)

        st.subheader('Metricas de validacion')
        st.dataframe(metricas.style.format({
            'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}', 'ROC_AUC': '{:.3f}'
        }), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Matriz de confusion - Regresion Logistica**')
            fig1, ax1 = plt.subplots()
            ConfusionMatrixDisplay(res_log['Matriz']).plot(ax=ax1, colorbar=False)
            st.pyplot(fig1)
            plt.close(fig1)
        with col2:
            st.markdown('**Matriz de confusion - Random Forest**')
            fig2, ax2 = plt.subplots()
            ConfusionMatrixDisplay(res_rf['Matriz']).plot(ax=ax2, colorbar=False)
            st.pyplot(fig2)
            plt.close(fig2)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('**Importancia de variables - Random Forest**')
            st.dataframe(imp_rf.head(10), use_container_width=True)
        with col4:
            st.markdown('**Coeficientes - Regresion Logistica**')
            st.dataframe(coefs_log[['Variable', 'Coeficiente']].head(10), use_container_width=True)

        st.subheader('Interpretacion automatica')
        st.write(interpretacion)

        st.subheader('Anexo tecnico para el informe')
        st.markdown(
            f"""
            - **Activo analizado:** `{ticker}`
            - **Indice de referencia:** `{benchmark}`
            - **Variable objetivo:** 1 si la rentabilidad de `t+1` es positiva, 0 en caso contrario.
            - **Separacion temporal:** {int(train_pct * 100)}% entrenamiento / {int((1 - train_pct) * 100)}% prueba.
            - **Modelos aplicados:** Regresion Logistica y Random Forest.
            - **Variables explicativas:** rentabilidades rezagadas, cambios de volumen, SMA ratio, MACD, RSI, volatilidad y retornos del indice.
            """
        )

        csv_metricas = metricas.to_csv(index=False).encode('utf-8')
        st.download_button('Descargar metricas CSV', csv_metricas, file_name='metricas_modelos.csv', mime='text/csv')

    except Exception as e:
        st.exception(e)
else:
    st.markdown(
        """
        ### Recomendacion para tu entrega
        1. Ejecuta varios tickers del IBEX 35 o del mercado que elijas.
        2. Escoge el caso con resultados mas coherentes y defendibles.
        3. Copia las metricas, la matriz de confusion y las variables mas relevantes en el informe.
        4. Añade una interpretacion critica: utilidad como apoyo tactico, pero no como sistema autonomo de trading.
        """
    )

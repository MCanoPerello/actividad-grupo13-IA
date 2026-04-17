import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Clasificacion de activos financieros", layout="wide")

FREQUENCY_CONFIG = {
    "Diaria": {"interval": "1d", "periods_per_year": 252, "label": "diaria"},
    "Semanal": {"interval": "1wk", "periods_per_year": 52, "label": "semanal"},
    "Mensual": {"interval": "1mo", "periods_per_year": 12, "label": "mensual"},
}

VARIABLE_DEFINITIONS = {
    "ret_1": "Rentabilidad del último periodo. Mide el impulso más reciente del precio.",
    "ret_5": "Rentabilidad acumulada de los últimos 5 periodos. Resume momentum de corto plazo.",
    "ret_10": "Rentabilidad acumulada de los últimos 10 periodos. Captura tendencia intermedia.",
    "ret_20": "Rentabilidad acumulada de los últimos 20 periodos. Refleja sesgo más persistente.",
    "vol_chg_1": "Cambio porcentual del volumen frente al periodo anterior. Señala aceleración o pérdida de interés.",
    "vol_chg_5": "Cambio porcentual del volumen frente a 5 periodos atrás. Mide confirmación de actividad.",
    "sma_ratio_5_20": "Relación entre media móvil corta y larga. Positivo suele indicar sesgo alcista reciente.",
    "macd": "Diferencia entre EMA 12 y EMA 26. Mide momentum tendencial.",
    "macd_signal": "Media exponencial del MACD. Se usa como referencia para cruces de señal.",
    "macd_hist": "Diferencia entre MACD y su señal. Mide fuerza de aceleración del momentum.",
    "rsi_14": "Índice de fuerza relativa a 14 periodos. Evalúa presión compradora o vendedora.",
    "volatility_10": "Desviación típica de retornos a 10 periodos. Aproxima volatilidad reciente.",
    "volatility_20": "Desviación típica de retornos a 20 periodos. Refleja riesgo algo más estable.",
    "range_intraday": "Amplitud relativa entre máximo y mínimo del periodo. Mide rango de negociación.",
    "bench_ret_1": "Rentabilidad del benchmark en el último periodo. Aporta contexto de mercado.",
    "bench_ret_5": "Rentabilidad del benchmark a 5 periodos. Resume momentum del índice.",
    "bench_ret_20": "Rentabilidad del benchmark a 20 periodos. Captura tendencia más amplia del mercado.",
}


@st.cache_data(show_spinner=False)
def descargar_datos(ticker: str, benchmark: str | None, inicio: str, fin: str, interval: str):
    activo = yf.download(ticker, start=inicio, end=fin, interval=interval, auto_adjust=True, progress=False)
    indice = pd.DataFrame()
    if benchmark:
        indice = yf.download(benchmark, start=inicio, end=fin, interval=interval, auto_adjust=True, progress=False)
    return activo, indice


def rsi(series: pd.Series, periodo: int = 14) -> pd.Series:
    delta = series.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    media_ganancia = ganancia.ewm(alpha=1 / periodo, adjust=False).mean()
    media_perdida = perdida.ewm(alpha=1 / periodo, adjust=False).mean()
    rs = media_ganancia / media_perdida.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def preparar_dataset(df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None, usar_benchmark: bool = True):
    data = df.copy()
    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()

    close_safe = data["Close"].replace(0, np.nan)
    volume_safe = data["Volume"].replace(0, np.nan)

    data["ret_1"] = close_safe.pct_change(1)
    data["ret_5"] = close_safe.pct_change(5)
    data["ret_10"] = close_safe.pct_change(10)
    data["ret_20"] = close_safe.pct_change(20)
    data["vol_chg_1"] = volume_safe.pct_change(1)
    data["vol_chg_5"] = volume_safe.pct_change(5)

    data["sma_5"] = close_safe.rolling(5).mean()
    data["sma_20"] = close_safe.rolling(20).mean()
    data["sma_ratio_5_20"] = data["sma_5"] / data["sma_20"].replace(0, np.nan) - 1

    data["ema_12"] = close_safe.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close_safe.ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    data["rsi_14"] = rsi(close_safe, 14)
    data["volatility_10"] = data["ret_1"].rolling(10).std()
    data["volatility_20"] = data["ret_1"].rolling(20).std()
    data["range_intraday"] = (data["High"] - data["Low"]) / close_safe

    base_features = [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "vol_chg_1",
        "vol_chg_5",
        "sma_ratio_5_20",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "volatility_10",
        "volatility_20",
        "range_intraday",
    ]

    benchmark_features = []
    if usar_benchmark and benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.copy()
        bench.columns = [c[0] if isinstance(c, tuple) else c for c in bench.columns]
        bench = bench[["Close"]].rename(columns={"Close": "Bench_Close"})
        bench_close_safe = bench["Bench_Close"].replace(0, np.nan)
        bench["bench_ret_1"] = bench_close_safe.pct_change(1)
        bench["bench_ret_5"] = bench_close_safe.pct_change(5)
        bench["bench_ret_20"] = bench_close_safe.pct_change(20)
        data = data.join(bench[["bench_ret_1", "bench_ret_5", "bench_ret_20"]], how="left")
        benchmark_features = ["bench_ret_1", "bench_ret_5", "bench_ret_20"]

    data["target"] = (close_safe.shift(-1) / close_safe - 1 > 0).astype(int)
    data["next_return"] = close_safe.shift(-1) / close_safe - 1

    data = data.replace([np.inf, -np.inf], np.nan)
    feature_cols = base_features + benchmark_features
    data = data.dropna(subset=feature_cols + ["target", "next_return", "Close", "sma_5", "sma_20"]).copy()
    return data, feature_cols


def dividir_temporal(data: pd.DataFrame, train_pct: float = 0.8):
    corte = int(len(data) * train_pct)
    train = data.iloc[:corte].copy()
    test = data.iloc[corte:].copy()
    return train, test


def filtrar_por_correlacion(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.95):
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    keep_cols = [c for c in X_train.columns if c not in drop_cols]
    return X_train[keep_cols].copy(), X_test[keep_cols].copy(), keep_cols, drop_cols


def construir_modelos(random_state: int = 42):
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return logistic, rf


def optimizar_modelo(nombre: str, modelo, X_train: pd.DataFrame, y_train: pd.Series, cv):
    if nombre == "Regresion Logistica":
        param_dist = {
            "clf__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__solver": ["lbfgs", "liblinear"],
        }
        n_iter = 6
    else:
        param_dist = {
            "clf__n_estimators": [150, 250, 350, 500],
            "clf__max_depth": [4, 5, 6, 8, None],
            "clf__min_samples_leaf": [2, 3, 5, 8, 12],
            "clf__max_features": ["sqrt", "log2", None],
        }
        n_iter = 8

    search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluar_cv(nombre: str, modelo, X_train: pd.DataFrame, y_train: pd.Series, cv):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    res = cross_validate(modelo, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    return {
        "Modelo": nombre,
        "CV_Accuracy_Media": np.nanmean(res["test_accuracy"]),
        "CV_Accuracy_STD": np.nanstd(res["test_accuracy"]),
        "CV_Precision_Media": np.nanmean(res["test_precision"]),
        "CV_Recall_Media": np.nanmean(res["test_recall"]),
        "CV_F1_Media": np.nanmean(res["test_f1"]),
        "CV_ROC_AUC_Media": np.nanmean(res["test_roc_auc"]),
    }


def evaluar_modelo(nombre: str, modelo, X_train, y_train, X_test, y_test, threshold: float = 0.5):
    modelo.fit(X_train, y_train)
    prob = modelo.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan
    fpr, tpr, _ = roc_curve(y_test, prob) if len(np.unique(y_test)) > 1 else (None, None, None)
    matriz = confusion_matrix(y_test, pred, labels=[0, 1])

    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc,
        "Matriz": matriz,
        "Predicciones": pred,
        "Probabilidades": prob,
        "ModeloEntrenado": modelo,
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": threshold,
        "Pct_Predice_1": float(np.mean(pred)),
    }


def importancia_random_forest(modelo, columnas):
    estimador = modelo.named_steps["clf"]
    df = pd.DataFrame({"Variable": columnas, "Importancia": estimador.feature_importances_})
    return df.sort_values("Importancia", ascending=False).reset_index(drop=True)


def coeficientes_logistic(modelo, columnas):
    estimador = modelo.named_steps["clf"]
    df = pd.DataFrame({"Variable": columnas, "Coeficiente": estimador.coef_[0]})
    df["Abs"] = df["Coeficiente"].abs()
    return df.sort_values("Abs", ascending=False).reset_index(drop=True)


def preparar_predicciones_test(test_df: pd.DataFrame, resultado_modelo: dict):
    pred_df = pd.DataFrame(index=test_df.index)
    pred_df["Close"] = test_df["Close"]
    pred_df["Real"] = test_df["target"]
    pred_df["Prob_Subida"] = resultado_modelo["Probabilidades"]
    pred_df["Prediccion"] = resultado_modelo["Predicciones"]
    pred_df["Next_Return"] = test_df["next_return"]
    pred_df["Strategy_Return"] = np.where(pred_df["Prob_Subida"] >= resultado_modelo["Threshold"], pred_df["Next_Return"], 0.0)
    pred_df["BuyHold_Return"] = pred_df["Next_Return"]
    pred_df["Strategy_Cum"] = (1 + pred_df["Strategy_Return"].fillna(0)).cumprod()
    pred_df["BuyHold_Cum"] = (1 + pred_df["BuyHold_Return"].fillna(0)).cumprod()
    return pred_df


def resumir_backtest(pred_df: pd.DataFrame, periods_per_year: int):
    estrategia_total = pred_df["Strategy_Cum"].iloc[-1] - 1
    buyhold_total = pred_df["BuyHold_Cum"].iloc[-1] - 1
    operado_pct = (pred_df["Strategy_Return"] != 0).mean()

    def calc_sharpe(returns: pd.Series):
        returns = returns.fillna(0)
        std = returns.std()
        if pd.notna(std) and std > 0:
            return (returns.mean() / std) * np.sqrt(periods_per_year)
        return np.nan

    def calc_max_drawdown(cum_series: pd.Series):
        drawdown = cum_series / cum_series.cummax() - 1
        return drawdown.min()

    sharpe_estrategia = calc_sharpe(pred_df["Strategy_Return"])
    sharpe_buyhold = calc_sharpe(pred_df["BuyHold_Return"])

    max_drawdown_estrategia = calc_max_drawdown(pred_df["Strategy_Cum"])
    max_drawdown_buyhold = calc_max_drawdown(pred_df["BuyHold_Cum"])

    return {
        "Rentabilidad estrategia": estrategia_total,
        "Rentabilidad buy_hold": buyhold_total,
        "Pct periodos invertido": operado_pct,
        "Sharpe estrategia": sharpe_estrategia,
        "Sharpe buy_hold": sharpe_buyhold,
        "Max drawdown estrategia": max_drawdown_estrategia,
        "Max drawdown buy_hold": max_drawdown_buyhold,
    }


def grafico_precio(data: pd.DataFrame, ticker: str, benchmark_df: pd.DataFrame | None, benchmark: str | None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name=f"{ticker} Close"))
    fig.add_trace(go.Scatter(x=data.index, y=data["sma_5"], mode="lines", name="SMA 5"))
    fig.add_trace(go.Scatter(x=data.index, y=data["sma_20"], mode="lines", name="SMA 20"))

    if benchmark_df is not None and not benchmark_df.empty and benchmark:
        bench = benchmark_df.copy()
        bench.columns = [c[0] if isinstance(c, tuple) else c for c in bench.columns]
        bench = bench[["Close"]].rename(columns={"Close": "Bench_Close"}).dropna()
        if not bench.empty:
            bench_norm = bench["Bench_Close"] / bench["Bench_Close"].iloc[0] * 100
            activo_norm = data["Close"] / data["Close"].iloc[0] * 100
            fig.add_trace(go.Scatter(x=data.index, y=activo_norm, mode="lines", name=f"{ticker} Base100", visible="legendonly"))
            fig.add_trace(go.Scatter(x=bench.index, y=bench_norm, mode="lines", name=f"{benchmark} Base100", visible="legendonly"))

    fig.update_layout(
        title="Precio de cierre y medias móviles",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        hovermode="x unified",
        height=500,
    )
    return fig


def grafico_matriz_confusion(matriz: np.ndarray, titulo: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=matriz,
            x=["Predice 0", "Predice 1"],
            y=["Real 0", "Real 1"],
            text=matriz,
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title=titulo, height=380, yaxis=dict(autorange="reversed"))
    return fig


def grafico_roc(res_log: dict, res_rf: dict):
    fig = go.Figure()
    if res_log["FPR"] is not None:
        fig.add_trace(
            go.Scatter(
                x=res_log["FPR"],
                y=res_log["TPR"],
                mode="lines",
                name=f"Regresion Logistica (AUC={res_log['ROC_AUC']:.3f})",
            )
        )
    if res_rf["FPR"] is not None:
        fig.add_trace(
            go.Scatter(
                x=res_rf["FPR"],
                y=res_rf["TPR"],
                mode="lines",
                name=f"Random Forest (AUC={res_rf['ROC_AUC']:.3f})",
            )
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(dash="dash")))
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    return fig


def grafico_probabilidad_vs_real(pred_df: pd.DataFrame, threshold: float, nombre_modelo: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["Prob_Subida"],
            mode="lines",
            name="Probabilidad predicha",
        )
    )

    reales_sube = pred_df[pred_df["Real"] == 1]
    reales_no_sube = pred_df[pred_df["Real"] == 0]

    fig.add_trace(
        go.Scatter(
            x=reales_sube.index,
            y=reales_sube["Real"],
            mode="markers",
            name="Real = 1 (sube)",
            marker=dict(symbol="circle", size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reales_no_sube.index,
            y=reales_no_sube["Real"],
            mode="markers",
            name="Real = 0 (no sube)",
            marker=dict(symbol="x", size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=np.repeat(threshold, len(pred_df)),
            mode="lines",
            name=f"Umbral {threshold:.2f}",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Probabilidad predicha vs resultado real - {nombre_modelo}",
        xaxis_title="Fecha",
        yaxis_title="Probabilidad / clase real",
        yaxis=dict(range=[-0.05, 1.05]),
        hovermode="x unified",
        height=450,
    )
    return fig


def grafico_distribucion_probabilidades(pred_log: pd.DataFrame, pred_rf: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pred_log["Prob_Subida"], nbinsx=25, name="Regresion Logistica", opacity=0.65))
    fig.add_trace(go.Histogram(x=pred_rf["Prob_Subida"], nbinsx=25, name="Random Forest", opacity=0.65))
    fig.update_layout(
        title="Distribución de probabilidades predichas",
        xaxis_title="Probabilidad de subida",
        yaxis_title="Frecuencia",
        barmode="overlay",
        height=420,
    )
    return fig


def grafico_backtest(pred_df: pd.DataFrame, nombre_modelo: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Strategy_Cum"], mode="lines", name=f"Estrategia {nombre_modelo}"))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["BuyHold_Cum"], mode="lines", name="Buy & Hold"))
    fig.update_layout(
        title="Backtest simple: estrategia vs buy & hold",
        xaxis_title="Fecha",
        yaxis_title="Capital acumulado (base 1)",
        hovermode="x unified",
        height=450,
    )
    return fig


def grafico_importancias(imp_rf: pd.DataFrame, coefs_log: pd.DataFrame):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Random Forest", "Regresion Logistica"))
    top_rf = imp_rf.head(10).sort_values("Importancia")
    top_log = coefs_log.head(10).sort_values("Coeficiente")

    fig.add_trace(
        go.Bar(x=top_rf["Importancia"], y=top_rf["Variable"], orientation="h", name="RF"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=top_log["Coeficiente"], y=top_log["Variable"], orientation="h", name="Logit"),
        row=1,
        col=2,
    )
    fig.update_layout(title="Variables más relevantes", height=500, showlegend=False)
    return fig


def tabla_definiciones_variables(columnas_finales: list[str]):
    filas = []
    for variable in columnas_finales:
        filas.append(
            {
                "Variable": variable,
                "Definición": VARIABLE_DEFINITIONS.get(variable, "Variable derivada del pipeline de mercado."),
            }
        )
    return pd.DataFrame(filas)


def comentario_rendimiento(ganador):
    if ganador["ROC_AUC"] < 0.55 or ganador["F1"] < 0.10:
        return (
            "La capacidad discriminativa del modelo es **muy limitada** en el test fuera de muestra. "
            "Está muy cerca de un comportamiento aleatorio, así que debe presentarse como evidencia de las dificultades reales del problema, no como un sistema listo para operar."
        )
    if ganador["Precision"] >= 0.55 and ganador["Recall"] < 0.20:
        return (
            "El modelo tiene un perfil **muy conservador**: cuando lanza señal positiva suele filtrar bastante, pero detecta muy pocas subidas reales. "
            "Eso suele ocurrir con umbrales exigentes o con probabilidades poco separadas."
        )
    if ganador["ROC_AUC"] >= 0.60 and ganador["F1"] >= 0.35:
        return (
            "El modelo muestra una señal **razonable** para una serie financiera ruidosa. No es una ventaja decisiva, pero sí puede servir como filtro táctico junto con otras reglas de control."
        )
    return (
        "El modelo ofrece una señal **intermedia**: mejora ligeramente sobre una referencia aleatoria, pero su fortaleza no es estable ni suficientemente alta como para usarlo de forma aislada."
    )


def comentario_estabilidad(ganador, ganador_cv):
    gap_auc = abs(float(ganador["ROC_AUC"]) - float(ganador_cv["CV_ROC_AUC_Media"])) if pd.notna(ganador["ROC_AUC"]) else np.nan
    gap_f1 = abs(float(ganador["F1"]) - float(ganador_cv["CV_F1_Media"])) if pd.notna(ganador["F1"]) else np.nan
    if pd.notna(gap_auc) and pd.notna(gap_f1) and (gap_auc > 0.08 or gap_f1 > 0.15):
        return (
            "La diferencia entre test final y validación temporal interna sugiere **inestabilidad temporal**. "
            "Eso encaja con la no estacionariedad del mercado y obliga a evitar conclusiones demasiado fuertes a partir de un solo tramo de prueba."
        )
    return (
        "Las métricas de test y de validación temporal están **razonablemente alineadas**, lo que sugiere una señal más estable dentro del periodo analizado."
    )


def comentario_operativo(ganador, threshold: float):
    if ganador["Pct_Predice_1"] < 0.05:
        return (
            f"Con un umbral operativo de **{threshold:.2f}**, el modelo apenas genera señales positivas. "
            "Eso reduce operativa y drawdown, pero también puede hundir el recall y dejar escapar gran parte de los movimientos alcistas."
        )
    if threshold >= 0.60:
        return (
            f"El umbral operativo de **{threshold:.2f}** hace la estrategia más prudente que el estándar 0,50. "
            "Es útil para filtrar ruido, aunque a cambio sacrifica sensibilidad ante subidas reales."
        )
    return (
        f"El umbral operativo de **{threshold:.2f}** busca un equilibrio más cercano al estándar de clasificación. "
        "Suele mejorar la sensibilidad, aunque también puede aumentar falsas señales."
    )


def comentario_variables(imp_rf: pd.DataFrame, coefs_log: pd.DataFrame):
    top_rf = ", ".join(imp_rf.head(5)["Variable"].tolist())
    top_log_pos = coefs_log[coefs_log["Coeficiente"] > 0].head(3)["Variable"].tolist()
    top_log_neg = coefs_log[coefs_log["Coeficiente"] < 0].head(3)["Variable"].tolist()
    return (
        f"En Random Forest, las variables más influyentes son: **{top_rf}**. "
        f"En la Regresión Logística, las variables con efecto positivo más claro sobre la probabilidad de subida son: **{', '.join(top_log_pos) if top_log_pos else 'ninguna dominante'}**; "
        f"las que empujan hacia la clase de bajada son: **{', '.join(top_log_neg) if top_log_neg else 'ninguna dominante'}**."
    )


def generar_interpretacion(metricas: pd.DataFrame, cv_metricas: pd.DataFrame, imp_rf: pd.DataFrame, coefs_log: pd.DataFrame, threshold: float, drop_cols: list[str], backtest_resumen: dict):
    ganador = metricas.sort_values(["F1", "ROC_AUC", "Accuracy"], ascending=False).iloc[0]
    ganador_cv = cv_metricas.loc[cv_metricas["Modelo"] == ganador["Modelo"]].iloc[0]

    comentarios = [
        f"El mejor modelo en el test fuera de muestra es **{ganador['Modelo']}**, con F1={ganador['F1']:.3f}, precision={ganador['Precision']:.3f}, recall={ganador['Recall']:.3f}, accuracy={ganador['Accuracy']:.3f} y ROC-AUC={ganador['ROC_AUC']:.3f}.",
        comentario_rendimiento(ganador),
        comentario_estabilidad(ganador, ganador_cv),
        comentario_operativo(ganador, threshold),
        comentario_variables(imp_rf, coefs_log),
    ]

    if drop_cols:
        comentarios.append(
            f"El filtro de correlación ha eliminado **{len(drop_cols)}** variables redundantes: **{', '.join(drop_cols)}**. Eso reduce ruido y riesgo de sobreajuste."
        )

    if backtest_resumen["Rentabilidad estrategia"] > backtest_resumen["Rentabilidad buy_hold"]:
        comentarios.append(
            "En el backtest simple, la estrategia basada en probabilidades supera a buy & hold en el periodo test. Es una señal interesante, aunque sigue siendo necesario incorporar costes de transacción y validación adicional."
        )
    else:
        comentarios.append(
            "En el backtest simple, la estrategia no supera a buy & hold en el periodo test. Por tanto, el modelo debe interpretarse como apoyo táctico o ejercicio académico, no como prueba de ventaja económica sostenible."
        )

    comentarios.append(
        "Desde una óptica financiera, el modelo no sustituye la gestión del riesgo, los costes de transacción, el juicio del analista ni la validación continua fuera de muestra."
    )
    return "\n\n".join(comentarios)


def ejecutar_analisis(params: dict):
    ticker = params["ticker"]
    benchmark = params["benchmark"]
    usar_benchmark = params["usar_benchmark"]
    inicio = params["inicio"]
    fin = params["fin"]
    train_pct = params["train_pct"]
    threshold = params["threshold"]
    aplicar_filtro_corr = params["aplicar_filtro_corr"]
    threshold_corr = params["threshold_corr"]
    optimizar = params["optimizar"]
    cv_splits = params["cv_splits"]
    frecuencia = params["frecuencia"]

    interval = FREQUENCY_CONFIG[frecuencia]["interval"]
    periods_per_year = FREQUENCY_CONFIG[frecuencia]["periods_per_year"]
    activo, indice = descargar_datos(ticker, benchmark, str(inicio), str(fin), interval)

    if activo.empty:
        raise ValueError("No se han descargado datos para el ticker del activo. Revisa el símbolo introducido.")
    if usar_benchmark and (indice is None or indice.empty):
        raise ValueError("No se han descargado datos para el benchmark. Revisa el símbolo o desactiva el benchmark.")

    data, columnas_features = preparar_dataset(activo, indice, usar_benchmark=usar_benchmark)
    if len(data) < 180:
        raise ValueError("Hay muy pocas observaciones tras construir las variables. Amplía el rango temporal o usa mayor frecuencia de datos.")

    train, test = dividir_temporal(data, train_pct=train_pct)
    X_train, y_train = train[columnas_features], train["target"]
    X_test, y_test = test[columnas_features], test["target"]

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    dropped_features = []
    if aplicar_filtro_corr:
        X_train, X_test, columnas_finales, dropped_features = filtrar_por_correlacion(X_train, X_test, threshold=threshold_corr)
    else:
        columnas_finales = list(X_train.columns)

    if len(columnas_finales) < 3:
        raise ValueError("La selección de variables ha dejado muy pocas features. Sube el umbral de correlación o desactiva el filtro.")

    logistic, rf = construir_modelos()
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    mejores_params = []
    if optimizar:
        logistic, best_params_log, best_score_log = optimizar_modelo("Regresion Logistica", logistic, X_train, y_train, tscv)
        rf, best_params_rf, best_score_rf = optimizar_modelo("Random Forest", rf, X_train, y_train, tscv)
        mejores_params = [
            {"Modelo": "Regresion Logistica", "Mejor ROC-AUC CV": best_score_log, "Parámetros": str(best_params_log)},
            {"Modelo": "Random Forest", "Mejor ROC-AUC CV": best_score_rf, "Parámetros": str(best_params_rf)},
        ]

    cv_log = evaluar_cv("Regresion Logistica", logistic, X_train, y_train, tscv)
    cv_rf = evaluar_cv("Random Forest", rf, X_train, y_train, tscv)
    cv_metricas = pd.DataFrame([cv_log, cv_rf])

    res_log = evaluar_modelo("Regresion Logistica", logistic, X_train, y_train, X_test, y_test, threshold=threshold)
    res_rf = evaluar_modelo("Random Forest", rf, X_train, y_train, X_test, y_test, threshold=threshold)

    metricas = pd.DataFrame(
        [
            {k: v for k, v in res_log.items() if k in ["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Pct_Predice_1"]},
            {k: v for k, v in res_rf.items() if k in ["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Pct_Predice_1"]},
        ]
    )

    imp_rf = importancia_random_forest(res_rf["ModeloEntrenado"], columnas_finales)
    coefs_log = coeficientes_logistic(res_log["ModeloEntrenado"], columnas_finales)

    pred_log = preparar_predicciones_test(test, res_log)
    pred_rf = preparar_predicciones_test(test, res_rf)

    mejor_modelo_nombre = metricas.sort_values(["F1", "ROC_AUC", "Accuracy"], ascending=False).iloc[0]["Modelo"]
    pred_mejor = pred_log if mejor_modelo_nombre == "Regresion Logistica" else pred_rf
    backtest_resumen = resumir_backtest(pred_mejor, periods_per_year=periods_per_year)
    interpretacion = generar_interpretacion(metricas, cv_metricas, imp_rf, coefs_log, threshold, dropped_features, backtest_resumen)

    resumen = pd.DataFrame(
        {
            "Activo": [ticker],
            "Benchmark": [benchmark if usar_benchmark else "No usado"],
            "Frecuencia": [frecuencia],
            "Periodo": [f"{data.index.min().date()} a {data.index.max().date()}"],
            "Pct clase 1 (sube)": [round(data["target"].mean() * 100, 2)],
            "Pct clase 0 (baja)": [round((1 - data["target"].mean()) * 100, 2)],
            "Pct clase 1 en test": [round(test["target"].mean() * 100, 2)],
            "Fecha inicio test": [test.index.min().date()],
            "Fecha fin test": [test.index.max().date()],
        }
    )

    balance = pd.DataFrame(
        {
            "Clase": ["0 = no sube", "1 = sube"],
            "Observaciones": [int((data["target"] == 0).sum()), int((data["target"] == 1).sum())],
        }
    )

    definiciones_variables = tabla_definiciones_variables(columnas_finales)
    export_metricas = metricas.merge(cv_metricas, on="Modelo", how="left")

    return {
        "params": params,
        "data": data,
        "indice": indice,
        "train": train,
        "test": test,
        "columnas_finales": columnas_finales,
        "dropped_features": dropped_features,
        "metricas": metricas,
        "cv_metricas": cv_metricas,
        "mejores_params": mejores_params,
        "res_log": res_log,
        "res_rf": res_rf,
        "imp_rf": imp_rf,
        "coefs_log": coefs_log,
        "pred_log": pred_log,
        "pred_rf": pred_rf,
        "mejor_modelo_nombre": mejor_modelo_nombre,
        "pred_mejor": pred_mejor,
        "backtest_resumen": backtest_resumen,
        "interpretacion": interpretacion,
        "resumen": resumen,
        "balance": balance,
        "definiciones_variables": definiciones_variables,
        "export_metricas": export_metricas,
        "periods_per_year": periods_per_year,
    }


st.title("Actividad 2 - Aprendizaje supervisado: clasificacion")
st.markdown(
    "App para descargar datos de mercado, construir variables técnicas, aplicar selección de características y comparar **Regresión Logística** y **Random Forest** con validación temporal."
)

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

with st.sidebar:
    st.header("Parámetros")
    ticker = st.text_input("Ticker del activo", value="SAN.MC")
    frecuencia = st.selectbox("Frecuencia de los datos", options=list(FREQUENCY_CONFIG.keys()), index=0)
    usar_benchmark = st.checkbox("Usar benchmark", value=True)
    benchmark = st.text_input("Ticker del índice de referencia", value="^IBEX") if usar_benchmark else None
    inicio = st.date_input("Fecha inicio", value=date(2018, 1, 1))
    fin = st.date_input("Fecha fin", value=date(2025, 12, 31))
    train_pct = st.slider("Porcentaje para entrenamiento", min_value=0.60, max_value=0.90, value=0.80, step=0.05)
    threshold = st.slider("Umbral para señal positiva", min_value=0.50, max_value=0.80, value=0.55, step=0.01)
    aplicar_filtro_corr = st.checkbox("Aplicar filtro de correlación", value=True)
    threshold_corr = st.slider("Umbral de correlación", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    optimizar = st.checkbox(
        "Optimizar hiperparámetros",
        value=False,
        help="Prueba varias configuraciones internas del modelo para maximizar ROC-AUC en validación temporal. Puede tardar bastante y no garantiza mejorar F1 con el umbral elegido.",
    )
    cv_splits = st.slider("Splits TimeSeriesSplit", min_value=3, max_value=6, value=4, step=1)
    ejecutar = st.button("Ejecutar análisis", type="primary")

params = {
    "ticker": ticker,
    "frecuencia": frecuencia,
    "usar_benchmark": usar_benchmark,
    "benchmark": benchmark,
    "inicio": inicio,
    "fin": fin,
    "train_pct": train_pct,
    "threshold": threshold,
    "aplicar_filtro_corr": aplicar_filtro_corr,
    "threshold_corr": threshold_corr,
    "optimizar": optimizar,
    "cv_splits": cv_splits,
}

st.info(
    "La variable objetivo es binaria: **1** si la rentabilidad del siguiente periodo es positiva y **0** si es negativa o cero. El modelo operativo usa un enfoque **long/cash**: entra si la probabilidad de subida supera el umbral y se queda fuera en caso contrario."
)

st.caption(
    "Nota: las medias móviles, RSI, volatilidad y demás indicadores se calculan en la frecuencia seleccionada. Por ejemplo, en frecuencia semanal, 5 periodos equivalen a 5 semanas."
)

if ejecutar:
    try:
        with st.spinner("Descargando datos y ejecutando el análisis..."):
            st.session_state.analysis_results = ejecutar_analisis(params)
    except Exception as e:
        st.session_state.analysis_results = None
        st.exception(e)

results = st.session_state.analysis_results

if results is not None:
    data = results["data"]
    indice = results["indice"]
    train = results["train"]
    test = results["test"]
    columnas_finales = results["columnas_finales"]
    dropped_features = results["dropped_features"]
    metricas = results["metricas"]
    cv_metricas = results["cv_metricas"]
    mejores_params = results["mejores_params"]
    res_log = results["res_log"]
    res_rf = results["res_rf"]
    imp_rf = results["imp_rf"]
    coefs_log = results["coefs_log"]
    pred_log = results["pred_log"]
    pred_rf = results["pred_rf"]
    mejor_modelo_nombre = results["mejor_modelo_nombre"]
    pred_mejor = results["pred_mejor"]
    backtest_resumen = results["backtest_resumen"]
    interpretacion = results["interpretacion"]
    resumen = results["resumen"]
    balance = results["balance"]
    definiciones_variables = results["definiciones_variables"]
    export_metricas = results["export_metricas"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones totales", len(data))
    c2.metric("Train", len(train))
    c3.metric("Test", len(test))
    c4.metric("Features finales", len(columnas_finales))

    st.subheader("Resumen de datos")
    st.dataframe(resumen, use_container_width=True)

    st.subheader("Balance de clases")
    fig_balance = go.Figure(data=[go.Bar(x=balance["Clase"], y=balance["Observaciones"], text=balance["Observaciones"], textposition="outside")])
    fig_balance.update_layout(height=350)
    st.plotly_chart(fig_balance, use_container_width=True)

    st.subheader("Gráfico interactivo del activo")
    st.plotly_chart(grafico_precio(data, ticker, indice if usar_benchmark else None, benchmark), use_container_width=True)

    st.subheader("Selección de variables")
    cols_fs1, cols_fs2 = st.columns([2, 1])
    with cols_fs1:
        st.dataframe(pd.DataFrame({"Variables finales": columnas_finales}), use_container_width=True)
    with cols_fs2:
        if dropped_features:
            st.dataframe(pd.DataFrame({"Variables eliminadas": dropped_features}), use_container_width=True)
        else:
            st.success("No se eliminaron variables por correlación.")

    st.subheader("Definición corta de las variables seleccionadas")
    st.dataframe(definiciones_variables, use_container_width=True, hide_index=True)

    st.subheader("Métricas de validación en test")
    st.dataframe(
        metricas.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "ROC_AUC": "{:.3f}",
                "Pct_Predice_1": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Validación temporal (TimeSeriesSplit)")
    st.dataframe(
        cv_metricas.style.format(
            {
                "CV_Accuracy_Media": "{:.3f}",
                "CV_Accuracy_STD": "{:.3f}",
                "CV_Precision_Media": "{:.3f}",
                "CV_Recall_Media": "{:.3f}",
                "CV_F1_Media": "{:.3f}",
                "CV_ROC_AUC_Media": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    if mejores_params:
        st.subheader("Optimización de hiperparámetros")
        st.dataframe(pd.DataFrame(mejores_params), use_container_width=True)

    st.subheader("Matrices de confusión")
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        st.plotly_chart(grafico_matriz_confusion(res_log["Matriz"], "Regresión Logística"), use_container_width=True)
    with col_conf2:
        st.plotly_chart(grafico_matriz_confusion(res_rf["Matriz"], "Random Forest"), use_container_width=True)

    st.caption(
        "Consejo de lectura: si la columna 'Predice 1' sale muy baja, el modelo está lanzando muy pocas señales alcistas. Suele deberse a un umbral alto o a probabilidades poco separadas."
    )

    st.subheader("Curva ROC")
    st.plotly_chart(grafico_roc(res_log, res_rf), use_container_width=True)

    st.subheader("Probabilidad predicha vs resultado real")
    modelo_vista = st.radio(
        "Modelo para visualizar",
        options=["Regresion Logistica", "Random Forest"],
        horizontal=True,
        key="modelo_vista_probabilidades",
    )
    pred_vista = pred_log if modelo_vista == "Regresion Logistica" else pred_rf
    st.plotly_chart(grafico_probabilidad_vs_real(pred_vista, threshold, modelo_vista), use_container_width=True)

    st.subheader("Distribución de probabilidades predichas")
    st.plotly_chart(grafico_distribucion_probabilidades(pred_log, pred_rf), use_container_width=True)

    st.subheader("Importancia de variables y coeficientes")
    st.plotly_chart(grafico_importancias(imp_rf, coefs_log), use_container_width=True)

    st.subheader("Backtest simple")

    bt_row1_col1, bt_row1_col2, bt_row1_col3 = st.columns(3)
    bt_row1_col1.metric("Modelo usado", mejor_modelo_nombre)
    bt_row1_col2.metric("Rent. estrategia", f"{backtest_resumen['Rentabilidad estrategia'] * 100:.2f}%")
    bt_row1_col3.metric("Rent. buy&hold", f"{backtest_resumen['Rentabilidad buy_hold'] * 100:.2f}%")
    
    bt_row2_col1, bt_row2_col2 = st.columns(2)
    bt_row2_col1.metric(
        "Sharpe estrategia",
        f"{backtest_resumen['Sharpe estrategia']:.2f}" if pd.notna(backtest_resumen['Sharpe estrategia']) else "n.d."
    )
    bt_row2_col2.metric(
        "Sharpe buy&hold",
        f"{backtest_resumen['Sharpe buy_hold']:.2f}" if pd.notna(backtest_resumen['Sharpe buy_hold']) else "n.d."
    )
    
    bt_row3_col1, bt_row3_col2 = st.columns(2)
    bt_row3_col1.metric(
        "Max drawdown estrategia",
        f"{backtest_resumen['Max drawdown estrategia'] * 100:.2f}%"
    )
    bt_row3_col2.metric(
        "Max drawdown buy&hold",
        f"{backtest_resumen['Max drawdown buy_hold'] * 100:.2f}%"
    )
    
    st.caption(f"Porcentaje de periodos invertido: {backtest_resumen['Pct periodos invertido']:.1%}")
    
    
    
    
    
    st.plotly_chart(grafico_backtest(pred_mejor, mejor_modelo_nombre), use_container_width=True)

    st.subheader("Interpretación automática")
    st.write(interpretacion)

    st.subheader("Anexo técnico para el informe")
    st.markdown(
        f"""
- **Activo analizado:** `{ticker}`
- **Benchmark:** `{benchmark if usar_benchmark else 'No utilizado'}`
- **Frecuencia de trabajo:** `{frecuencia}`.
- **Variable objetivo:** 1 si la rentabilidad de `t+1` es positiva; 0 si es negativa o cero.
- **Separación temporal:** {int(train_pct * 100)}% entrenamiento / {int((1 - train_pct) * 100)}% prueba.
- **Validación interna:** TimeSeriesSplit con {cv_splits} particiones.
- **Umbral operativo:** {threshold:.2f}.
- **Modelos aplicados:** Regresión Logística y Random Forest.
- **Selección de variables:** {'Sí, filtro de correlación' if aplicar_filtro_corr else 'No aplicada'}.
- **Optimización de hiperparámetros:** {'Sí' if optimizar else 'No'}.
- **Variables explicativas usadas:** {', '.join(columnas_finales)}.
        """
    )

    csv_metricas = export_metricas.to_csv(index=False).encode("utf-8")
    csv_pred_mejor = pred_mejor.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
    csv_imp = imp_rf.to_csv(index=False).encode("utf-8")
    csv_coef = coefs_log.to_csv(index=False).encode("utf-8")
    csv_defs = definiciones_variables.to_csv(index=False).encode("utf-8")

    dl1, dl2, dl3, dl4, dl5 = st.columns(5)
    dl1.download_button("Descargar métricas", csv_metricas, file_name="metricas_modelos.csv", mime="text/csv")
    dl2.download_button("Descargar predicciones", csv_pred_mejor, file_name="predicciones_mejor_modelo.csv", mime="text/csv")
    dl3.download_button("Descargar importancias RF", csv_imp, file_name="importancias_random_forest.csv", mime="text/csv")
    dl4.download_button("Descargar coeficientes variables", csv_coef, file_name="coeficientes_logistica.csv", mime="text/csv")
    dl5.download_button("Descargar definiciones variables", csv_defs, file_name="definiciones_variables.csv", mime="text/csv")
else:
    st.markdown(
        """
### Recomendación de uso
1. Prueba varios tickers y varias frecuencias para comparar estabilidad.
2. Empieza con umbral 0.55 y optimización desactivada; después prueba variantes.
3. Si la columna `Predice 1` es muy baja, revisa el umbral o la separación temporal.
4. Usa la gráfica de probabilidad predicha vs resultado real para explicar cuándo el modelo acierta o se equivoca.
        """
    )

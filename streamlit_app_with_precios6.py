import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
import warnings

# --- Librerías de Modelado ---
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objs as go

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Simulador: Demanda & Precipitación")
warnings.filterwarnings("ignore")

# --- Manejo de importaciones opcionales ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# -----------------------------------------------------------------------------
# ------------------------- CONFIGURACIÓN GLOBAL ------------------------------
# -----------------------------------------------------------------------------

# Rutas de archivos por defecto
PATH_QUITO = "Precipitación Quito.xlsx"
PATH_GUAYAQUIL = "Precipitación Guayaquil.xlsx"

LOOKBACK_DEFAULT = 12
TRAIN_START_DEFAULT = "2001-01-01"
TRAIN_END_DEFAULT   = "2022-12-01"
TEST_START_DEFAULT  = "2023-01-01"
TEST_END_DEFAULT    = "2025-10-01"
FUTURE_PERIODS_DEFAULT = 24
SEED = 42
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# ------------------------- CARGA DE DATOS ------------------------------------
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    try:
        dy = pd.read_excel('Data Ventas ene2017 -oct2024.xlsx')
        external_vars = pd.read_excel("Variables Externas Demanda.xlsx")
        external_vars['ds'] = pd.to_datetime(external_vars['ds'])
        prices = pd.read_excel('Lista de Precios BI V1.xlsx')
        normales = pd.read_excel("Normales_Climatologicas.xlsx")
        budget = pd.read_excel("Budget 2025.xlsx")
        
        if 'Marca' in dy.columns:
            dy['Marca'] = dy['Marca'].fillna("Desconocido")
            
        return dy, external_vars, prices, normales, budget
    except Exception:
        return None, None, None, None, None

dy, base_external_vars, prices, normales_climatologicas, budget = load_data()

if base_external_vars is not None:
    for i in range(1, 4):
        key = f'external_vars_esc{i}'
        if key not in st.session_state:
            st.session_state[key] = base_external_vars.copy()

# -----------------------------------------------------------------------------
# ------------------------- FUNCIONES UTILITARIAS -----------------------------
# -----------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "R^2": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.nan
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape, "R^2": r2}

def load_and_preprocess_precip(file_bytes=None, ruta_archivo=None):
    if file_bytes:
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(ruta_archivo)
    
    col_fecha_real = None
    col_prec_real = None
    
    posibles_fecha = ["fecha", "date", "periodo"]
    for c in df.columns:
        if c.lower().strip() in posibles_fecha:
            col_fecha_real = c
            break
    if not col_fecha_real: col_fecha_real = df.columns[0]

    posibles_prec = ["precipitación", "precipitacion", "rain", "pp", "mm", "valor"]
    for c in df.columns:
        if c.lower().strip() in posibles_prec:
            col_prec_real = c
            break
    if not col_prec_real: col_prec_real = df.columns[1]

    df[col_fecha_real] = pd.to_datetime(df[col_fecha_real], errors="coerce")
    df = df.dropna(subset=[col_fecha_real, col_prec_real])
    
    df["mes"] = df[col_fecha_real].dt.to_period("M").dt.to_timestamp()
    df_mes = df.groupby("mes", as_index=False)[col_prec_real].sum().sort_values("mes").reset_index(drop=True)
    df_mes.columns = ["mes", "Precipitacion"]
    
    all_mes = pd.date_range(df_mes["mes"].min(), df_mes["mes"].max(), freq="MS")
    df_mes = df_mes.set_index("mes").reindex(all_mes)
    df_mes["Precipitacion"] = df_mes["Precipitacion"].interpolate(method="time").fillna(method="bfill").fillna(method="ffill")
    df_mes = df_mes.reset_index().rename(columns={"index": "mes"})
    
    df_mes["month"] = df_mes["mes"].dt.month
    df_mes["sin_month"] = np.sin(2 * np.pi * (df_mes["month"] - 1) / 12)
    df_mes["cos_month"] = np.cos(2 * np.pi * (df_mes["month"] - 1) / 12)
    
    return df_mes

def create_sequences(y_scaled, exog_scaled, lookback):
    X, y, idxs = [], [], []
    n = len(y_scaled)
    for i in range(n - lookback):
        y_w = y_scaled[i:i+lookback]
        exog_w = exog_scaled[i:i+lookback]
        X.append(np.column_stack([y_w, exog_w]))
        y.append(y_scaled[i+lookback])
        idxs.append(i+lookback)
    return np.array(X), np.array(y), np.array(idxs)

# -----------------------------------------------------------------------------
# ------------------------- MODELOS (Forecast) --------------------------------
# -----------------------------------------------------------------------------

def run_xgboost_precip(df_mes, lookback, train_start, train_end, test_end, future_periods):
    df_mes = df_mes[df_mes['mes'] >= pd.to_datetime(train_start)].reset_index(drop=True)
    
    scaler_y = RobustScaler()
    y_raw = df_mes["Precipitacion"].values.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y_raw).flatten()
    
    scaler_x = RobustScaler()
    x_exog = df_mes[["sin_month", "cos_month"]].values
    x_exog_scaled = scaler_x.fit_transform(x_exog)
    
    X_seq, y_seq, idxs = create_sequences(y_scaled, x_exog_scaled, lookback)
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    
    dates = df_mes["mes"].values
    seq_dates = dates[idxs]
    train_mask = (seq_dates <= np.datetime64(train_end))
    test_mask = (seq_dates > np.datetime64(train_end)) & (seq_dates <= np.datetime64(test_end))
    
    X_train, y_train = X_flat[train_mask], y_seq[train_mask]
    X_test, y_test = X_flat[test_mask], y_seq[test_mask]
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=3, n_jobs=-1, random_state=SEED)
    model.fit(X_train, y_train)
    
    if len(X_test) > 0:
        pred_test = scaler_y.inverse_transform(model.predict(X_test).reshape(-1,1)).flatten()
        true_test = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
    else:
        pred_test, true_test = [], []

    last_idx = idxs[train_mask][-1] if len(X_test) == 0 else idxs[test_mask][-1]
    
    start_fut = df_mes.loc[last_idx, "mes"] + pd.DateOffset(months=1)
    fut_dates = [start_fut + pd.DateOffset(months=i) for i in range(future_periods)]
    fut_months = pd.to_datetime(fut_dates).month
    fut_sin = np.sin(2 * np.pi * (fut_months - 1) / 12)
    fut_cos = np.cos(2 * np.pi * (fut_months - 1) / 12)
    fut_exog = np.column_stack([fut_sin, fut_cos])
    fut_exog_scaled = scaler_x.transform(fut_exog)
    
    full_hist_y = list(y_scaled[:last_idx+1])
    full_hist_exog = list(x_exog_scaled[:last_idx+1])
    for row in fut_exog_scaled:
        full_hist_exog.append(row)
    full_hist_exog = np.array(full_hist_exog)
    
    current_step_idx = last_idx + 1
    future_preds_scaled = []
    
    for i in range(future_periods):
        win_y = full_hist_y[-lookback:]
        win_exog = full_hist_exog[current_step_idx-lookback : current_step_idx]
        win_comb = np.column_stack([win_y, win_exog]).flatten().reshape(1, -1)
        pred_s = model.predict(win_comb)[0]
        future_preds_scaled.append(pred_s)
        full_hist_y.append(pred_s)
        current_step_idx += 1
        
    future_preds = scaler_y.inverse_transform(np.array(future_preds_scaled).reshape(-1,1)).flatten()
    metrics = calculate_metrics(true_test, pred_test)
    
    return {
        "name": "XGBoost",
        "pred_test": pred_test,
        "true_test": true_test,
        "test_dates": seq_dates[test_mask],
        "future_dates": fut_dates,
        "future_preds": future_preds,
        "metrics": metrics
    }

def run_lstm_precip(df_mes, lookback, train_start, train_end, test_end, future_periods):
    if not TF_AVAILABLE: return None
    df_mes = df_mes[df_mes['mes'] >= pd.to_datetime(train_start)].reset_index(drop=True)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(df_mes["Precipitacion"].values.reshape(-1, 1)).flatten()
    scaler_x = RobustScaler()
    x_exog_scaled = scaler_x.fit_transform(df_mes[["sin_month", "cos_month"]].values)
    
    X_seq, y_seq, idxs = create_sequences(y_scaled, x_exog_scaled, lookback)
    
    dates = df_mes["mes"].values
    seq_dates = dates[idxs]
    train_mask = (seq_dates <= np.datetime64(train_end))
    test_mask = (seq_dates > np.datetime64(train_end)) & (seq_dates <= np.datetime64(test_end))
    
    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]
    
    tf.random.set_seed(SEED)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 3)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    
    if len(X_test) > 0:
        pred_test = scaler_y.inverse_transform(model.predict(X_test).flatten().reshape(-1,1)).flatten()
        true_test = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
    else:
        pred_test, true_test = [], []
    
    last_idx = idxs[train_mask][-1] if len(X_test) == 0 else idxs[test_mask][-1]
    
    start_fut = df_mes.loc[last_idx, "mes"] + pd.DateOffset(months=1)
    fut_dates = [start_fut + pd.DateOffset(months=i) for i in range(future_periods)]
    fut_months = pd.to_datetime(fut_dates).month
    fut_sin = np.sin(2 * np.pi * (fut_months - 1) / 12)
    fut_cos = np.cos(2 * np.pi * (fut_months - 1) / 12)
    fut_exog_scaled = scaler_x.transform(np.column_stack([fut_sin, fut_cos]))
    
    full_y = list(y_scaled[:last_idx+1])
    full_exog = list(x_exog_scaled[:last_idx+1])
    for row in fut_exog_scaled: full_exog.append(row)
    full_exog = np.array(full_exog)
    
    fut_preds_s = []
    curr_idx = last_idx + 1
    
    for i in range(future_periods):
        win_y = np.array(full_y[-lookback:]).reshape(-1,1)
        win_exog = full_exog[curr_idx-lookback : curr_idx]
        win = np.column_stack([win_y, win_exog]).reshape(1, lookback, 3)
        pred = model.predict(win, verbose=0).flatten()[0]
        fut_preds_s.append(pred)
        full_y.append(pred)
        curr_idx += 1
        
    future_preds = scaler_y.inverse_transform(np.array(fut_preds_s).reshape(-1,1)).flatten()
    metrics = calculate_metrics(true_test, pred_test)
    
    return {
        "name": "LSTM",
        "pred_test": pred_test,
        "true_test": true_test,
        "test_dates": seq_dates[test_mask],
        "future_dates": fut_dates,
        "future_preds": future_preds,
        "metrics": metrics
    }

def run_prophet_precip(df_mes, train_start, train_end, test_end, future_periods):
    df_p = df_mes[df_mes['mes'] >= pd.to_datetime(train_start)].reset_index(drop=True)
    df_p = df_p[["mes", "Precipitacion"]].rename(columns={"mes": "ds", "Precipitacion": "y"})
    
    train_mask = (df_p['ds'] <= train_end)
    df_train = df_p[train_mask]
    df_test = df_p[(df_p['ds'] > train_end) & (df_p['ds'] <= test_end)]
    
    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df_train)
    
    forecast_test = m.predict(df_test[['ds']])
    pred_test = forecast_test['yhat'].values
    true_test = df_test['y'].values
    
    future_dates = m.make_future_dataframe(periods=future_periods + len(df_test), freq='MS')
    all_forecast = m.predict(future_dates)
    future_preds_df = all_forecast[all_forecast['ds'] > pd.to_datetime(test_end)].head(future_periods)
    
    metrics = calculate_metrics(true_test, pred_test)
    
    return {
        "name": "Prophet",
        "pred_test": pred_test,
        "true_test": true_test,
        "test_dates": df_test['ds'],
        "future_dates": future_preds_df['ds'],
        "future_preds": future_preds_df['yhat'].values,
        "metrics": metrics
    }

def run_sarimax_precip(df_mes, train_start, train_end, test_end, future_periods):
    if not STATSMODELS_AVAILABLE: return None
    
    # 1. PREPARACIÓN ROBUSTA DE DATOS
    # Convertir inputs a Timestamp y forzar al día 1 del mes para evitar ambigüedades
    ts_train_start = pd.to_datetime(train_start).replace(day=1)
    ts_train_end = pd.to_datetime(train_end).replace(day=1)
    ts_test_end = pd.to_datetime(test_end).replace(day=1)
    
    # Filtrar desde el inicio y establecer índice
    df_s = df_mes[df_mes['mes'] >= ts_train_start].copy()
    df_s = df_s.set_index('mes')
    
    # IMPORTANTE: Establecer frecuencia explicita 'MS' (Month Start)
    # Esto evita que SARIMAX adivine mal el número de pasos
    df_s = df_s.asfreq('MS')
    
    # Llenar posibles huecos generados por asfreq (aunque interpolate previo ya debió hacerlo)
    df_s = df_s.interpolate().fillna(method='bfill')

    # 2. DEFINIR RANGOS EXACTOS
    # Train: Desde inicio hasta train_end (inclusive)
    train = df_s.loc[:ts_train_end, 'Precipitacion']
    exog_train = df_s.loc[:ts_train_end, ['sin_month', 'cos_month']]
    
    # Test: Estrictamente desde el mes SIGUIENTE a train_end hasta test_end
    # Esto elimina el uso de [1:] que causaba el error de dimensiones
    ts_test_start = ts_train_end + pd.DateOffset(months=1)
    
    # Validar que test_start no sea mayor que test_end
    if ts_test_start > ts_test_end:
        st.warning(f"⚠️ El rango de Test es inválido o vacío. La fecha fin de entrenamiento ({ts_train_end.date()}) es igual o posterior al fin de test ({ts_test_end.date()}). Ajusta las fechas.")
        return None

    test = df_s.loc[ts_test_start:ts_test_end, 'Precipitacion']
    exog_test = df_s.loc[ts_test_start:ts_test_end, ['sin_month', 'cos_month']]
    
    # Validación final de longitudes
    if len(test) == 0:
        return None

    try:
        # 3. MODELADO
        model = SARIMAX(train, exog=exog_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        
        # Predicción Test (Usando índices explícitos de la data recortada)
        pred_test = results.get_prediction(start=test.index[0], end=test.index[-1], exog=exog_test).predicted_mean
        
        # 4. FUTURO
        last_date = test.index[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(future_periods)]
        
        # Crear Exógenas Futuras
        fut_months = pd.to_datetime(future_dates).month
        fut_sin = np.sin(2 * np.pi * (fut_months - 1) / 12)
        fut_cos = np.cos(2 * np.pi * (fut_months - 1) / 12)
        exog_future = pd.DataFrame({'sin_month': fut_sin, 'cos_month': fut_cos}, index=future_dates)
        
        future_preds = results.get_forecast(steps=future_periods, exog=exog_future).predicted_mean
        
        metrics = calculate_metrics(test.values, pred_test.values)
        
        return {
            "name": "SARIMAX",
            "pred_test": pred_test.values,
            "true_test": test.values,
            "test_dates": test.index,
            "future_dates": future_dates,
            "future_preds": future_preds.values,
            "metrics": metrics
        }
    except Exception as e:
        # Mostrar el error de forma amigable sin romper la app
        st.error(f"Error SARIMAX (Data insuficiente o parámetros inválidos): {e}")
        return None
# -----------------------------------------------------------------------------
# ------------------------- DEMANDA (Prophet) ---------------------------------
# -----------------------------------------------------------------------------

def filter_and_aggregate_data(dy, id_material=None, marca=None, canal=None):
    df = dy.copy()
    if id_material: df = df[df['id_material'] == id_material]
    elif marca: df = df[df['Marca'] == marca]
    if canal: df = df[df['Canal'].isin(canal if isinstance(canal, list) else [canal])]
    df = df.groupby('ds', as_index=False)['y'].sum()
    return df

def forecast_scenario(dy, external_vars, prices, id_material, marca, canal):
    df = filter_and_aggregate_data(dy, id_material=id_material, marca=marca, canal=canal)
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    df = df.merge(external_vars, on='ds', how='left')
    for col in ['Precipitacion', 'Economia', 'push_pos']:
        if col in df.columns: df[col] = df[col].interpolate().fillna(method='bfill').fillna(method='ffill')
        else: df[col] = 0

    m = Prophet()
    m.add_regressor('Precipitacion'); m.add_regressor('Economia'); m.add_regressor('push_pos')
    m.fit(df[['ds', 'y', 'Precipitacion', 'Economia', 'push_pos']])

    future = m.make_future_dataframe(periods=72, freq='MS')
    future = future.merge(external_vars, on='ds', how='left')
    for col in ['Precipitacion', 'Economia', 'push_pos']:
        future[col] = future[col].interpolate().fillna(method='bfill').fillna(method='ffill')

    forecast = m.predict(future)
    
    avg_price = 0
    if marca and 'CATEGORIA 5 (MARCA)' in prices.columns:
        filtered = prices[prices['CATEGORIA 5 (MARCA)'] == marca]
        vals = []
        for c in (canal if isinstance(canal, list) else [canal]):
            if c in filtered.columns:
                vals.extend(filtered[c].replace(0, np.nan).dropna().values)
        if vals:
            avg_price = np.mean(vals)
            if marca == "CEMENTO ASFALTICO": avg_price *= 0.47
            elif marca == "ALUMBAND ROLLO": avg_price *= 0.98
            elif marca == "ASFALUM": avg_price *= 1.3
            else: avg_price *= 0.73

    forecast_dollars = forecast[['ds']].copy()
    forecast_dollars['RF'] = forecast['yhat'] * avg_price
    return forecast_dollars, forecast, df, m

# -----------------------------------------------------------------------------
# ------------------------- INTERFAZ DE USUARIO (UI) --------------------------
# -----------------------------------------------------------------------------

st.sidebar.title("Menú Principal")
nav_option = st.sidebar.radio("Ir a:", ["Simulador Demanda", "Forecast Precipitación"])

# --- VISTA 1: DEMANDA ---
if nav_option == "Simulador Demanda":
    st.title("Simulador Comparativo: Escenarios de Demanda")
    if dy is None: st.error("⚠️ Error cargando datos.")
    else:
        st.sidebar.markdown("---")
        filter_type = st.sidebar.radio("Filtrar por:", ("Código del Producto", "Marca"))
        if filter_type == "Código del Producto":
            selected_product = st.sidebar.selectbox("Código", sorted(dy['id_material'].unique()))
            selected_brand = None
        else:
            selected_brand = st.sidebar.selectbox("Marca", sorted(dy['Marca'].unique()))
            selected_product = None
        channels = sorted(dy['Canal'].unique())
        selected_channels = st.sidebar.multiselect("Canales", channels, default=channels)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Variables Externas")
        years_ext = sorted(base_external_vars['ds'].dt.year.unique()) if base_external_vars is not None else [2025]
        sel_year = st.sidebar.selectbox("Año", years_ext)
        sel_month = st.sidebar.selectbox("Mes", ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])
        sel_var = st.sidebar.selectbox("Variable", ['Precipitacion', 'Economia', 'push_pos'])
        month_idx = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'].index(sel_month) + 1
        sel_date = pd.to_datetime(f"{sel_year}-{month_idx:02d}-01")
        
        def get_val(esc):
            df = st.session_state.get(f'external_vars_esc{esc}')
            return float(df[df['ds'] == sel_date][sel_var].values[0]) if df is not None and not df[df['ds'] == sel_date].empty else 0.0

        c1, c2, c3 = st.sidebar.columns(3)
        v1 = c1.number_input("Esc 1", value=get_val(1), key="v1")
        v2 = c2.number_input("Esc 2", value=get_val(2), key="v2")
        v3 = c3.number_input("Esc 3", value=get_val(3), key="v3")
        
        if st.sidebar.button("Actualizar y Calcular"):
            for i, val in enumerate([v1, v2, v3], 1):
                key = f'external_vars_esc{i}'
                df = st.session_state[key]
                if sel_date in df['ds'].values: df.loc[df['ds'] == sel_date, sel_var] = val
                else: df = pd.concat([df, pd.DataFrame({'ds': [sel_date], sel_var: [val]})]).sort_values('ds').fillna(0)
                st.session_state[key] = df
            
            res_scenarios = []
            for i in range(1, 4):
                rf, fcast, hist, _ = forecast_scenario(dy, st.session_state[f'external_vars_esc{i}'], prices, selected_product, selected_brand, selected_channels)
                res_scenarios.append((rf, fcast))
            st.session_state['demand_results'] = {'hist': hist, 'scenarios': res_scenarios}

        if 'demand_results' in st.session_state:
            res = st.session_state['demand_results']; hist = res['hist']
            fig = go.Figure()
            if not hist.empty: fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Histórico', mode='lines+markers', line=dict(color='black')))
            colors = ['blue', 'green', 'orange']
            for i, (rf, fcast) in enumerate(res['scenarios']):
                if not fcast.empty: fig.add_trace(go.Scatter(x=fcast['ds'], y=fcast['yhat'], name=f'Esc {i+1}', line=dict(color=colors[i])))
            st.plotly_chart(fig, use_container_width=True)
            
            dfs = []
            for i, (rf, _) in enumerate(res['scenarios']):
                if not rf.empty: dfs.append(rf.set_index('ds')[['RF']].rename(columns={'RF': f'Escenario {i+1}'}))
            if dfs: st.dataframe(pd.concat(dfs, axis=1).loc[sel_date:sel_date+pd.DateOffset(months=12)].style.format("${:,.2f}"))

# --- VISTA 2: PRECIPITACIÓN ---
else:
    st.title("Módulo de Pronóstico Climático (Precipitación)")
    
    # 1. Selector Ciudad
    st.markdown("##### Seleccione la ciudad base:")
    city_option = st.radio("", ["Quito", "Guayaquil"], horizontal=True, label_visibility="collapsed")
    default_path = PATH_QUITO if city_option == "Quito" else PATH_GUAYAQUIL
    
    # 2. Carga Archivo
    col1, col2 = st.columns([1, 2])
    with col1:
        file_precip = st.file_uploader("Cargar Excel Precipitación (Opcional)", type=["xlsx"])
        use_default = st.checkbox(f"Usar archivo servidor: {default_path}", value=True)
    
    df_precip = None
    if file_precip: df_precip = load_and_preprocess_precip(file_bytes=file_precip.read())
    elif use_default:
        try: df_precip = load_and_preprocess_precip(ruta_archivo=default_path)
        except: st.warning(f"No se encontró: {default_path}")
            
    if df_precip is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Configuración")
        available_models = ["XGBoost", "Prophet", "SARIMAX"]
        if TF_AVAILABLE: available_models.insert(1, "LSTM")
        selected_models = st.sidebar.multiselect("Modelos", available_models, default=["XGBoost"])
        lookback = st.sidebar.number_input("Lookback", 3, 36, LOOKBACK_DEFAULT)
        future_p = st.sidebar.number_input("Meses Forecast", 1, 60, FUTURE_PERIODS_DEFAULT)
        
        st.sidebar.markdown("### Fechas")
        train_start = st.sidebar.date_input("Inicio Entrenamiento", pd.to_datetime(TRAIN_START_DEFAULT))
        train_end = st.sidebar.date_input("Fin Entrenamiento", pd.to_datetime(TRAIN_END_DEFAULT))
        test_end = st.sidebar.date_input("Fin Test", pd.to_datetime(TEST_END_DEFAULT))
        
        if st.sidebar.button("Ejecutar Pronósticos"):
            results_store = []
            prog_bar = st.progress(0); step = 1.0 / len(selected_models); curr = 0.0
            
            if "XGBoost" in selected_models:
                with st.spinner("XGBoost..."):
                    res = run_xgboost_precip(df_precip, lookback, str(train_start), str(train_end), str(test_end), future_p)
                    if res: results_store.append(res)
                curr += step; prog_bar.progress(min(curr, 1.0))
                
            if "LSTM" in selected_models:
                with st.spinner("LSTM..."):
                    res = run_lstm_precip(df_precip, lookback, str(train_start), str(train_end), str(test_end), future_p)
                    if res: results_store.append(res)
                curr += step; prog_bar.progress(min(curr, 1.0))
                
            if "Prophet" in selected_models:
                with st.spinner("Prophet..."):
                    res = run_prophet_precip(df_precip, str(train_start), str(train_end), str(test_end), future_p)
                    if res: results_store.append(res)
                curr += step; prog_bar.progress(min(curr, 1.0))
                
            if "SARIMAX" in selected_models:
                with st.spinner("SARIMAX..."):
                    res = run_sarimax_precip(df_precip, str(train_start), str(train_end), str(test_end), future_p)
                    if res: results_store.append(res)
                curr += step; prog_bar.progress(1.0)
            
            st.session_state['precip_results'] = results_store

        if 'precip_results' in st.session_state and st.session_state['precip_results']:
            results = st.session_state['precip_results']
            
            # A. Métricas
            st.subheader("Métricas (Test Set)")
            metrics_data = [{"Modelo": r['name'], **r['metrics']} for r in results]
            st.dataframe(pd.DataFrame(metrics_data).set_index("Modelo").style.highlight_min(subset=["mae", "rmse", "mape"], color='lightgreen').highlight_max(subset=["R^2"], color='lightgreen'))
            
            # B. Gráfico
            fig = go.Figure()
            mask_plot = df_precip['mes'] >= pd.to_datetime(train_start)
            fig.add_trace(go.Scatter(x=df_precip.loc[mask_plot, 'mes'], y=df_precip.loc[mask_plot, 'Precipitacion'], mode='lines', name='Real', line=dict(color='black', width=1)))
            
            colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
            for i, r in enumerate(results):
                c = colors[i % len(colors)]
                fig.add_trace(go.Scatter(x=r['test_dates'], y=r['pred_test'], mode='lines', name=f"{r['name']} (Test)", line=dict(color=c, dash='dot')))
                fig.add_trace(go.Scatter(x=r['future_dates'], y=r['future_preds'], mode='lines+markers', name=f"{r['name']} (Futuro)", line=dict(color=c, width=2)))
            st.plotly_chart(fig, use_container_width=True)
            
            # C. Tabla Comparativa (Real vs Pred)
            st.subheader("Comparativa Detallada: Real vs Predichos")
            all_dates = set()
            for r in results:
                all_dates.update(r['test_dates']); all_dates.update(r['future_dates'])
            df_comp = pd.DataFrame({'Fecha': sorted(list(all_dates))})
            df_comp = df_comp.merge(df_precip[['mes', 'Precipitacion']], left_on='Fecha', right_on='mes', how='left').drop(columns=['mes']).rename(columns={'Precipitacion': 'Valor Real'})
            
            for r in results:
                dates_model = list(r['test_dates']) + list(r['future_dates'])
                preds_model = list(r['pred_test']) + list(r['future_preds'])
                df_temp = pd.DataFrame({'Fecha': dates_model, f'Pred {r["name"]}': preds_model})
                df_comp = df_comp.merge(df_temp, on='Fecha', how='left')
            st.dataframe(df_comp.set_index('Fecha').style.format("{:.2f}"))

            # D. Exportar
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Metricas', index=False)
                df_comp.to_excel(writer, sheet_name='Comparativa_Detallada', index=False)
            st.download_button("Descargar Excel Completo", data=output.getvalue(), file_name=f"Forecast_{city_option}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            # E. Aplicar a Demanda
            st.markdown("---")
            model_to_apply = st.selectbox("Aplicar a Demanda:", ["Ninguno"] + [r['name'] for r in results])
            if model_to_apply != "Ninguno" and st.button("Aplicar"):
                sel = next((r for r in results if r['name'] == model_to_apply), None)
                if sel:
                    df_app = pd.DataFrame({"ds": sel['future_dates'], "Precipitacion": sel['future_preds']})
                    for i in range(1, 4):
                        key = f'external_vars_esc{i}'; base = st.session_state[key].set_index('ds'); new = df_app.set_index('ds')
                        base.update(new); st.session_state[key] = pd.concat([base, new[~new.index.isin(base.index)]]).sort_index().reset_index()
                    st.success(f"Datos de {city_option} aplicados.")
    else: st.info("Carga archivo.")
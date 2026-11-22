"""Dashboard ShopFast (Streamlit). Consume solo la API REST (patrón backend intermedio):
nunca llama a SageMaker directamente ni guarda credenciales de AWS."""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
API_KEY = os.getenv("DASHBOARD_API_KEY", "dev-key")
H = {"X-API-Key": API_KEY}
DEMO_USERS = {
    "Cliente frecuente (29 productos, 20 carros)": 1515915625532264305,
    "Cliente con historial mixto": 1515915625537806926,
    "Cliente con muchos carros": 1515915625572739225,
    "Cliente nuevo (sin historial)": 999,
}

st.set_page_config(page_title="ShopFast · Recomendaciones IA", layout="wide")
st.title("ShopFast · Recomendaciones de productos con IA")


def get(path, **kw):
    r = requests.get(f"{API_URL}{path}", headers=H, timeout=15, **kw)
    r.raise_for_status()
    return r.json()


try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.caption(f"API en línea · modelo servido por **{health['predictor']}**"
               + (f" (endpoint `{health['endpoint']}`)" if health.get("endpoint") else ""))
except Exception:
    st.error("La API no responde")
    st.stop()

tab_rec, tab_model, tab_ops = st.tabs(["Recomendaciones", "Calidad del modelo", "Operación"])

with tab_rec:
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Cliente de ejemplo", list(DEMO_USERS))
    user_id = c2.text_input("…o ingresa un user_id", value=str(DEMO_USERS[choice]))
    n = st.slider("Cantidad de recomendaciones", 3, 20, 10)
    data = get(f"/v1/recommendations/{user_id}", params={"n": n})
    m1, m2, m3 = st.columns(3)
    m1.metric("Cliente conocido", "Sí" if data["known_user"] else "No (cold-start)")
    m2.metric("Inferencia", f"{data['inference_ms']:.0f} ms")
    m3.metric("Alta probabilidad", sum(i["decision"] == "alta" for i in data["items"]))
    df = pd.DataFrame([{"#": i["rank"], "Producto": i["product_id"], "Categoría": i["category"], "Marca": i["brand"],
                        "Precio US$": i["price"], "Prob. compra": i["purchase_probability"], "Decisión": i["decision"],
                        "Origen": i["source"], "Principal motivo": i["reasons"][0]["text"]} for i in data["items"]])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"Prob. compra": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.subheader("¿Por qué se recomienda?")
    sel = st.selectbox("Producto", [i["product_id"] for i in data["items"]])
    item = next(i for i in data["items"] if i["product_id"] == sel)
    for r in item["reasons"]:
        st.write(f"{'⬆️' if r['impact'] > 0 else '⬇️'} {r['text']} (impacto {r['impact']:+.2f})")
    st.info("Acción sugerida: " + ("ofrecer cupón o envío gratis, alta intención de compra" if item["decision"] == "alta"
                                   else "mostrar en carrusel y remarketing" if item["decision"] == "media"
                                   else "no priorizar en la campaña"))
    if st.button("Simular: el cliente agrega este producto al carro"):
        r = requests.post(f"{API_URL}/v1/interactions", headers=H, timeout=10,
                          json={"user_id": int(user_id), "product_id": int(sel), "event_type": "cart"})
        st.success(f"Evento enviado a RabbitMQ ({r.json().get('event_id', r.status_code)}). Recarga para ver la nueva probabilidad.")

with tab_model:
    m = get("/v1/model")
    t = m["test_at_tuned_threshold"]
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{t['accuracy']:.3f}")
    b.metric("Precision", f"{t['precision']:.3f}")
    c.metric("Recall", f"{t['recall']:.3f}")
    d.metric("F1-score", f"{t['f1']:.3f}")
    e.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.caption(f"Conjunto de prueba: febrero 2021 ({m['rows']['test']:,} pares usuario-producto, "
               f"{m['baseline_positive_rate']:.1%} compras). Umbral {t['threshold']} elegido en validación.")
    st.write(f"**Ranking:** el producto que el modelo pone primero fue comprado en el {m['hit_at_1']:.1%} de los casos "
             f"(al azar: {m['hit_at_1_random']:.1%}).")
    cm = pd.DataFrame([[t["tn"], t["fp"]], [t["fn"], t["tp"]]], index=["Real: no compra", "Real: compra"],
                      columns=["Predicho: no compra", "Predicho: compra"])
    st.dataframe(cm)
    imp = pd.Series(m["feature_importance_gain"]).head(10).sort_values()
    st.bar_chart(imp, horizontal=True)

with tab_ops:
    try:
        s = get("/v1/stats")
        st.metric("Interacciones procesadas por el worker", s["total"])
        st.write(s["by_type"])
        st.dataframe(pd.DataFrame(s["recent"]), hide_index=True, use_container_width=True)
    except requests.RequestException:
        st.warning("Base de datos de interacciones no disponible")

import time
import zipfile
import io
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="AI 마법 주문 판별기", page_icon="🪄", layout="centered")

SPELL_NAMES = [
    "0_봄바르다 (Bombarda)", "1_아바다 케다브라 (Avada Kedavra)",
    "2_알로호모라 (Alohomora)", "3_익스펠리아무스 (Expelliarmus)",
    "4_익스펙토 패트로눔 (Expecto Patronum)", "5_인센디오 (Incendio)",
    "6_크루시오 (Crucio)", "7_프로테고 (Protego)"
]

@st.cache_resource
def load_and_train_model():
    all_zip_files = sorted(glob.glob("*.zip"))
    X_data, y_data = [], []

    for label_idx in range(8):
        matched_zips = [f for f in all_zip_files if f.startswith(f"{label_idx}_")]
        for z_path in matched_zips:
            with zipfile.ZipFile(z_path, "r") as outer_zip:
                inner_zips = sorted([name for name in outer_zip.namelist() if name.endswith(".zip")])
                for inner_name in inner_zips:
                    with zipfile.ZipFile(io.BytesIO(outer_zip.read(inner_name))) as z_in:
                        acc_arr = np.array(pd.read_csv(z_in.open("Accelerometer.csv"))[['x', 'y', 'z']])
                        gyro_arr = np.array(pd.read_csv(z_in.open("Gyroscope.csv"))[['x', 'y', 'z']])
                    
                    min_len = min(len(acc_arr), len(gyro_arr))
                    combined = np.hstack([acc_arr[:min_len], gyro_arr[:min_len]])
                    combined = combined[:300] if len(combined) >= 300 else np.pad(combined, ((0, 300 - len(combined)), (0, 0)), mode='constant')
                    
                    features = []
                    for seg in np.array_split(combined, 3, axis=0):
                        features.extend(np.concatenate([
                            np.mean(seg, axis=0), np.std(seg, axis=0),
                            np.max(seg, axis=0), np.min(seg, axis=0)
                        ]))
                    X_data.append(features)
                    y_data.append(label_idx)

    X_data, y_data = np.array(X_data), np.array(y_data)
    train_input, test_input, train_target, test_target = train_test_split(
        X_data, y_data, test_size=0.1, stratify=y_data, random_state=42
    )

    ss = StandardScaler()
    train_scaled = ss.fit_transform(train_input)
    kn = KNeighborsClassifier(n_neighbors=3).fit(train_scaled, train_target)
    return kn, ss, test_input, test_target, all_zip_files

kn, ss, test_input, test_target, all_zip_files = load_and_train_model()

st.title("🪄 AI 마법 지팡이 주문 판별 시스템")
st.caption("스마트폰 가속도·자이로스코프 센서(100Hz) 및 KNN 머신러닝 모델 기반 실시간 시연")

with st.sidebar:
    st.header("⚙️ 시연 컨트롤 패널")
    selected_spell = st.selectbox("시연할 동작 선택", range(8), format_func=lambda x: SPELL_NAMES[x])
    flip_x = -1 if st.checkbox("좌우 대칭이동 (Flip X)", value=True) else 1
    flip_y = -1 if st.checkbox("상하 대칭이동 (Flip Y)", value=True) else 1

st.divider()

if st.button("📡 스마트폰 센서 수신 및 주문 판별 시작", type="primary", use_container_width=True):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i in range(1, 101):
        status_text.markdown(f"**🪄 마법 지팡이 모션 데이터 수신 중... ({i*3} / 300 프레임, 100Hz)**")
        progress_bar.progress(i)
        time.sleep(0.025)
        
    status_text.success("✅ 센서 데이터 수신 완료! 학습된 KNN 모델 추론 결과:")
    
    demo_idx = np.where(test_target == selected_spell)[0][0]
    sample_scaled = ss.transform(test_input[demo_idx].reshape(1, -1))
    pred_idx = kn.predict(sample_scaled)[0]
    pred_proba = kn.predict_proba(sample_scaled)[0][pred_idx] * 100
    
    col1, col2 = st.columns(2)
    col1.metric("판별된 마법 주문", SPELL_NAMES[pred_idx].split(" ")[0])
    col2.metric("AI 예측 신뢰도 (확률)", f"{pred_proba:.1f}%")
    
    matched_zips = [f for f in all_zip_files if f.startswith(f"{selected_spell}_")]
    with zipfile.ZipFile(matched_zips[0], "r") as outer_zip:
        inner_zips = sorted([name for name in outer_zip.namelist() if name.endswith(".zip")])
        with zipfile.ZipFile(io.BytesIO(outer_zip.read(inner_zips[0]))) as z_in:
            demo_acc = pd.read_csv(z_in.open("Accelerometer.csv"))[['x', 'y', 'z']].to_numpy()[:300]

    vel_x = np.cumsum(demo_acc[:, 0] - np.mean(demo_acc[:, 0]))
    vel_y = np.cumsum(demo_acc[:, 1] - np.mean(demo_acc[:, 1]))
    pos_x = flip_x * np.cumsum(vel_x)
    pos_y = flip_y * np.cumsum(vel_y)

    chart_placeholder = st.empty()
    
    for step in range(10, 301, 15):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.set_xlim(np.min(pos_x)*1.2, np.max(pos_x)*1.2)
        ax.set_ylim(np.min(pos_y)*1.2, np.max(pos_y)*1.2)
        ax.set_facecolor("#0f172a")
        ax.plot(pos_x[:step], pos_y[:step], color='#38bdf8', linewidth=3)
        ax.plot(pos_x[step-1], pos_y[step-1], marker='*', markersize=18, color='#fde047')
        ax.set_title(f"Wand Trajectory: {SPELL_NAMES[pred_idx]}", fontsize=12, fontweight='bold')
        ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
        
        chart_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.03)

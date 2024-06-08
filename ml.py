import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# Function to check for missing values in the input data
def check_missing_values(df):
    if df.isnull().values.any():
        raise ValueError("입력 데이터에 누락된 값이 있습니다.")

# Function to predict using the input data and a given model and scalers
def predict_with_input(model, scaler_X, scaler_y, input_data, columns):
    input_df = pd.DataFrame([input_data], columns=columns)
    check_missing_values(input_df)

    input_scaled = scaler_X.transform(input_df)
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)

    return prediction[0]

# Function to run the Streamlit app
def run_ml_app():
    st.set_page_config(layout="wide")
    st.title("Project Mirror")

    if 'change_count' not in st.session_state:
        st.session_state.change_count = 0
    if 'change_history' not in st.session_state:
        st.session_state.change_history = {'academic_achievement': [], 'math_interest': []}

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.subheader("기본학교 정보")

        # User input sliders for 기타 변수
        input_data_others = {}

        st.subheader("학교 정보")
        input_data_others.update({
            'MATACH': st.slider('학업성취도', 0, 1000, 500),
            'SCHKOR': st.slider('한국어를 안 쓰는 학생 비율', 0, 5, 3),
            'SCHPOP': st.slider('학교주변 인구', 0, 7, 5),
            'SCHPOOR': st.slider('경제적으로 어려운 학생비율', 0, 5, 1),
            'SGENDER': st.slider('학생성별 비율', 0.0, 1.0, 1.0, step=0.1),
            'SGAREDU': st.slider('보호자 평균학력', 0.0, 10.0, 5.5),
            'SEDUASPR': st.slider('학생들의 교육포부', 0, 10, 8),
            'SMATINT': st.slider('수학흥미', 0, 4, 4),
            'SMATEFF': st.slider('수학효능/자신감', 0, 4, 3),
            'STCHRCAP': st.slider('교사능력에 대한 학생들의 인식', 0, 4, 3)
        })

        st.subheader("교사 정보")
        input_data_others.update({
            'TCAREER': st.slider('교사경력', 0, 60, 10),
            'TGENDER': st.slider('교사성별', 0, 1, 0),
            'TAGE': st.slider('교사나이', 20, 70, 30),
            'TMAJMATH': st.slider('수학과전공', 0, 1, 0),
            'TMAJME': st.slider('수학교육과전공', 0, 1, 1),
            'TSTUTEXP': st.slider('학생 공부에 대한 교사의 기대', 0, 5, 5),
            'TSTUPEXP': st.slider('학생 공부에 대한 학부모의 기대', 0, 5, 2),
            'TSTUSEXP': st.slider('학생들은 학교에서 얼마나 잘 지내려고 하나요?', 0, 5, 3),
            'TSTUCAP': st.slider('학생들의 역량은?', 0, 5, 4),
            'TSTURES': st.slider('학생들은 교사들을 존중하나요?', 0, 10, 1)
        })

        st.subheader("교사 만족도")
        input_data_others.update({
            'TCOLLAB': st.slider('선생님들간의 협력은 어느 정도인가요?', 0, 4, 2),
            'TSATIS': st.slider('선생님의 직무만족은 어느 수준이세요?', 0, 4, 2),
            'TPASSION': st.slider('선생님의 직무에 대한 열정은 어느 정도인가요?', 0, 4, 3),
            'TCRWD': st.slider('선생님은 담당하는 학생이 너무 많다고 생각하시나요?', 0, 4, 3),
            'TOVERADM': st.slider('선생님의 행정업무는 어느 수준이세요?', 0, 4, 2)
        })

        st.subheader("교사 발전")
        input_data_others.update({
            'TPDMCP': st.slider('수학교과 내용에 대한 연수를 받은 적이 있다', 0, 1, 1),
            'TPDMCF': st.slider('수학교과 내용에 대한 연수를 받을 계획이 있다', 0, 1, 1),
            'TPDMDP': st.slider('수학교육학에 대한 연수를 받은 적이 있다.', 0, 1, 0),
            'TPDMDF': st.slider('수학교육학에 대한 연수를 받을 계획이 있다', 0, 1, 0),
            'TPDMEP': st.slider('수학교육과정에 대한 연수를 받은 적이 있다', 0, 1, 0),
            'TPDMEF': st.slider('수학교육과정에 대한 연수를 받을 계획이 있다', 0, 1, 0),
            'TPDSVP': st.slider('수학 문제해결 역량에 대한 연수를 받은 적이 있다', 0, 1, 1),
            'TPDSVF': st.slider('수학 문제해결 역량에 대한 연수를 받을 계획이 있다', 0, 1, 1),
            'TPDTIME': st.slider('전문성 향상에 투자하는 시간은 어느 정도인가요? ', 0, 5, 3)
        })

        st.subheader("학생 학습")
        input_data_others.update({
            'TSTUPC': st.slider('학급 내 학생 수는 몇 명인가요?', 0, 50, 28),
            'TLACKK': st.slider('수학지식이 부족한 학생은 어느 정도인가요?', 0, 3, 3),
            'TLACKI': st.slider('수학 흥미가 부족한 학생은 어느 정도인가요?', 0, 3, 3),
            'TMTIME': st.slider('1주일에 수학 시간은 몇 분인가요?', 0, 1800, 700)
        })

        st.subheader("학생들이 배운 교과내용")
        input_data_others.update({
            'TPRIK1': st.slider('음수계산', 0, 2, 1),
            'TPRIK2': st.slider('분수와 소수', 0, 2, 1),
            'TPRIK3': st.slider('비율과 퍼센트', 0, 2, 0),
            'TPRIK4': st.slider('대수의 표현', 0, 2, 0),
            'TPRIK5': st.slider('단순 선형 방정식', 0, 2, 0),
            'TPRIK6': st.slider('단순 선형 부등식', 0, 2, 0),
            'TPRIK7': st.slider('연립방정식', 0, 2, 1),
            'TPRIK8': st.slider('선형/2차 함수 표현', 0, 2, 1),
            'TPRIK9': st.slider('함수의 특성', 0, 2, 1),
            'TPRIK10': st.slider('패턴을 표현하는 방법', 0, 2, 1),
            'TPRIK11': st.slider('각, 평형 모양에 대한 특성', 0, 2, 2),
            'TPRIK12': st.slider('지름, 둘레, 면적', 0, 2, 1),
            'TPRIK13': st.slider('피타고라스 정리', 0, 2, 2),
            'TPRIK14': st.slider('반사, 회전', 0, 2, 0),
            'TPRIK15': st.slider('합동과 닮음', 0, 2, 0),
            'TPRIK16': st.slider('3차원 도형', 0, 2, 0),
            'TPRIK17': st.slider('자료 해석법', 0, 2, 2),
            'TPRIK18': st.slider('자료 수집을 위한 절차', 0, 2, 1),
            'TPRIK19': st.slider('자료의 조직법', 0, 2, 1),
            'TPRIK20': st.slider('자료 요약통계', 0, 2, 1),
            'TPRIK21': st.slider('단순한 사건의 확률', 0, 2, 2),
            'TPRIK22': st.slider('다양한 사건의 확률', 0, 2, 1)
        })

    with col2:

        # User input sliders for 세부 변수
        input_data_details = {}

        st.subheader("교수 전략")
        input_data_details.update({
            'TINSSTG1': st.slider('실생활과 연결시켜 설명', 1, 4, 2, key='TINSSTG1'),
            'TINSSTG2': st.slider('답에 대해 설명', 1, 4, 1, key='TINSSTG2'),
            'TINSSTG3': st.slider('도전적인 활동을 제공', 1, 4, 4, key='TINSSTG3'),
            'TINSSTG4': st.slider('반 친구들과 토론', 1, 4, 1, key='TINSSTG4'),
            'TINSSTG5': st.slider('기존 지식의 연계', 1, 4, 3, key='TINSSTG5'),
            'TINSSTG6': st.slider('문제해결과정 설명', 1, 4, 4, key='TINSSTG6'),
            'TINSSTG7': st.slider('학생들의 생각을 표현하도록 함', 1, 4, 3, key='TINSSTG7')
        })

        st.subheader("학생에게 요청하는 내용")
        input_data_details.update({
            'TINSASK1': st.slider('수학 내용에 대한 교사의 설명을 듣도록 함', 1, 4, 1, key='TINSASK1'),
            'TINSASK2': st.slider('교사의 문제풀이 방법을 듣도록 함', 1, 4, 1, key='TINSASK2'),
            'TINSASK3': st.slider('법칙, 절차, 사실을 외우도록 함', 1, 4, 3, key='TINSASK3'),
            'TINSASK4': st.slider('스스로 절차를 연습해보도록 함', 1, 4, 2, key='TINSASK4'),
            'TINSASK5': st.slider('새로운 문제에 스스로 적용해보도록 함', 1, 4, 1, key='TINSASK5'),
            'TINSASK6': st.slider('교사의 지도 아래 반 전체가 문제를 풀어보도록 함', 1, 4, 1, key='TINSASK6'),
            'TINSASK7': st.slider('다양한 능력의 학생들끼리 협업하게 함', 1, 4, 2, key='TINSASK7'),
            'TINSASK8': st.slider('비슷한 능력의 학생들끼리 협업하게 함', 1, 4, 5, key='TINSASK8')
        })

        st.subheader("과제 빈도 및 시간")
        input_data_details.update({
            'THWFRQ': st.slider('과제를 얼마나 자주 내주십니까?', 1, 5, 5, key='THWFRQ'),
            'THWTIME': st.slider('과제에 소요되는 시간은 어느 수준입니까?', 1, 5, 4, key='THWTIME')
        })

        st.subheader("과제 전략")
        input_data_details.update({
            'THWSTG1': st.slider('과제에 대한 피드백을 줌', 1, 3, 1, key='THWSTG1'),
            'THWSTG2': st.slider('과제를 스스로 고치도록 함', 1, 3, 2, key='THWSTG2'),
            'THWSTG3': st.slider('수업에서 과제에 대한 토론을 함', 1, 3, 2, key='THWSTG3'),
            'THWSTG4': st.slider('과제가 완료되었는지 모니터링함', 1, 3, 3, key='THWSTG4'),
            'THWSTG5': st.slider('과제를 성적에 활용함', 1, 3, 3, key='THWSTG5'),
        })

        st.subheader("이외 중요한 교수학습 전략")
        input_data_details.update({
            'THWSTG6': st.slider('학생을 관찰함', 1, 3, 1, key='THWSTG6'),
            'THWSTG7': st.slider('학생에게 지속적으로 질문함', 1, 3, 3, key='THWSTG7'),
            'THWSTG8': st.slider('짧은 형태의 평가를 활용함', 1, 3, 3, key='THWSTG8'),
            'THWSTG9': st.slider('긴 형태의 평가를 활용함', 1, 3, 1, key='THWSTG9'),
            'THWSTG10': st.slider('장기간의 프로젝트를 부여함', 1, 3, 3, key='THWSTG10')
        })



    input_data = {**input_data_others, **input_data_details}

    def plot_changes(history, title):
        plt.figure(figsize=(10, 6), facecolor='none')  # figure 배경색 투명
        plt.plot(history, marker='o', color='#FFA500', linewidth=5)  # 그래프 색상 주황색, 선 굵기 조정
        plt.title(title, fontsize=20, color='white')  # 제목 글씨 크기 및 색상 설정
        plt.xlabel('Try', fontsize=15, color='white')  # x축 라벨 글씨 크기 및 색상 설정
        plt.ylabel('Value', fontsize=15, color='white')  # y축 라벨 글씨 크기 및 색상 설정
        plt.grid(True)
        plt.gca().patch.set_alpha(0)  # axes 배경색 투명

        # x, y 축 숫자 색상과 글씨 크기 설정
        plt.tick_params(axis='x', colors='white', labelsize=12)
        plt.tick_params(axis='y', colors='white', labelsize=12)

        st.pyplot(plt)


    with col3:
        st.subheader("학업성취도 예측결과")

        # Load Model 1
        model_path1 = 'C:/Users/namch/OneDrive/문서/streamlit/models/model1.pkl'
        if os.path.exists(model_path1):
            with open(model_path1, 'rb') as f:
                model1_serialized_loaded, model1_weights_loaded = joblib.load(f)

            # Reconstruct Model 1
            model1_loaded = tf.keras.models.model_from_json(model1_serialized_loaded)
            model1_loaded.set_weights(model1_weights_loaded)

            # Load scalers for Model 1
            scaler_X_loaded = joblib.load('C:/Users/namch/OneDrive/문서/streamlit/models/scaler_X.pkl')
            scaler_y_loaded = joblib.load('C:/Users/namch/OneDrive/문서/streamlit/models/scaler_y.pkl')

            # Prepare input data for Model 1
            input_data_model1 = {k: v for k, v in input_data.items() if k != 'MATACH'}
            columns1 = list(input_data_model1.keys())
            columns_sorted1 = scaler_X_loaded.feature_names_in_
            input_data_model1_sorted = {key: input_data_model1[key] for key in columns_sorted1}

            # Prediction for Model 1
            prediction_model1 = predict_with_input(model1_loaded, scaler_X_loaded, scaler_y_loaded,
                                                   input_data_model1_sorted, columns_sorted1)
            # prediction_model1 값에서 숫자만 추출하고 소수점 1자리까지 반올림
            if isinstance(prediction_model1, (list, np.ndarray)):
                prediction_value = round(prediction_model1.item(), 1)
            else:
                prediction_value = round(prediction_model1, 1)

            st.session_state.change_history['academic_achievement'].append(prediction_value)

            # CSS 스타일 적용하여 결과 출력
            st.markdown(
                f"""
                <style>
                .prediction-container {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                }}
                .prediction-box {{
                    text-align: center;
                    color: white;
                    font-size: 15pt;
                    background: linear-gradient(90deg, #FFA500, #FF4500);  /* 주황색과 그라데이션 */
                    border-radius: 10px;
                    padding: 10px;
                    display: inline-block;
                    margin-bottom: 20px;
                }}
                </style>
                <div class="prediction-container">
                    <div class="prediction-box">{prediction_value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            plot_changes(st.session_state.change_history['academic_achievement'], 'Math Score')
        else:
            st.write("모델 1 파일이 존재하지 않습니다. 모델 1 파일을 확인해주세요.")
            # Variables to optimize
        variables_to_optimize = [
            'TINSSTG1', 'TINSSTG2', 'TINSSTG3', 'TINSSTG4', 'TINSSTG5', 'TINSSTG6', 'TINSSTG7',
            'TINSASK1', 'TINSASK2', 'TINSASK3', 'TINSASK4', 'TINSASK5', 'TINSASK6', 'TINSASK7', 'TINSASK8',
            'THWFRQ', 'THWTIME', 'THWSTG1', 'THWSTG2', 'THWSTG3', 'THWSTG4', 'THWSTG5', 'THWSTG6', 'THWSTG7',
            'THWSTG8', 'THWSTG9', 'THWSTG10'
        ]

        # Ensure scalers and models are loaded correctly for optimization
        if 'scaler_X_loaded' in locals() and 'scaler_y_loaded' in locals():
            input_data_df_model1 = pd.DataFrame([input_data_model1], columns=scaler_X_loaded.feature_names_in_)
            input_data_scaled_model1 = scaler_X_loaded.transform(input_data_df_model1)
            optimize_indices_model1 = [scaler_X_loaded.feature_names_in_.tolist().index(var) for var in
                                       variables_to_optimize]

            optimal_input_scaled_model1 = tf.Variable(input_data_scaled_model1.copy(), dtype=tf.float32)
            optimizer_model1 = Adam(learning_rate=0.01)

            @tf.function
            def optimize_step_model1():
                with tf.GradientTape() as tape:
                    prediction = model1_loaded(optimal_input_scaled_model1, training=False)
                    loss = -prediction  # maximize prediction by minimizing negative prediction
                gradients = tape.gradient(loss, [optimal_input_scaled_model1])[0]
                updates = tf.zeros_like(optimal_input_scaled_model1)
                for i in optimize_indices_model1:
                    updates = tf.tensor_scatter_nd_update(updates, [[0, i]], [gradients[0, i]])
                optimizer_model1.apply_gradients(zip([updates], [optimal_input_scaled_model1]))
                return loss

            iterations = 50
            for i in range(iterations):
                loss = optimize_step_model1()

            optimal_inputs_model1 = scaler_X_loaded.inverse_transform(optimal_input_scaled_model1.numpy())
            optimal_prediction_scaled_model1 = model1_loaded.predict(optimal_input_scaled_model1)
            optimal_prediction_model1 = scaler_y_loaded.inverse_transform(
                optimal_prediction_scaled_model1.reshape(-1, 1))

            difference_scaled_model1 = optimal_input_scaled_model1.numpy() - input_data_scaled_model1
            difference_model1 = optimal_inputs_model1 - input_data_df_model1.values

            original_pred_value_model1 = prediction_model1
            optimal_pred_value_model1 = optimal_prediction_model1[0, 0]
            difference_value_model1 = optimal_pred_value_model1 - original_pred_value_model1
            percentage_increase_model1 = (difference_value_model1 / original_pred_value_model1) * 100

            # 소수점 첫째자리까지 반올림
            original_pred_value_model1 = round(float(original_pred_value_model1), 1)
            optimal_pred_value_model1 = round(float(optimal_pred_value_model1), 1)
            difference_value_model1 = round(float(difference_value_model1), 1)
            percentage_increase_model1 = round(float(percentage_increase_model1[0]), 1)

            # CSS 스타일 적용하여 결과 출력
            st.markdown(
                f"""
                   <style>
                   .result-container {{
                       display: flex;
                       flex-direction: column;
                       align-items: center;
                       margin-bottom: 20px;  /* 아래 여백 추가 */
                   }}
                   .result-text {{
                       text-align: center;
                       margin-bottom: 5px;  /* 텍스트와 숫자 사이 여백 추가 */
                   }}
                   .result-box {{
                       text-align: center;
                       color: #f63366;  /* 주황색과 슬라이더 색의 그라데이션 */
                       font-size: 15pt;
                       background-color: white;
                       border-radius: 10px;
                       padding: 5px 10px;
                       display: inline-block;
                       margin-bottom: 10px;  /* 아래 여백 추가 */
                   }}
                   </style>
                   <div class="result-container">
                       <div class="result-text">지금 조건에서는 성취도가 이정도인거 같아요</div>
                       <div class="result-box">{original_pred_value_model1}</div>
                   </div>
                   <div class="result-container">
                       <div class="result-text">현재 상황에서 어디까지 늘려볼 수 있을까요?</div>
                       <div class="result-box">{optimal_pred_value_model1}</div>
                   </div>
                   <div class="result-container">
                       <div class="result-text">교수학습전략을 수정해보면 얼마나 늘릴 수 있을까요?</div>
                       <div class="result-box">{difference_value_model1}</div>
                   </div>
                   <div class="result-container">
                       <div class="result-text">몇 퍼센트나 개선되는 걸까요?</div>
                       <div class="result-box">{percentage_increase_model1}%</div>
                   </div>
                   """,
                unsafe_allow_html=True
            )

            st.write(f'교수학습전략을 어떻게 수정해봐야 할까요?')

            results_df_model1 = pd.DataFrame({
                '전략': scaler_X_loaded.feature_names_in_,
                '현재': np.round(input_data_df_model1.values.flatten(), 0),
                '최적값': np.round(optimal_inputs_model1.flatten(), 0),
                '차이': np.round(difference_model1.flatten(), 0)
            })

            pd.set_option('display.max_rows', None)
            # difference 항이 0이 아닌 행만 필터링
            filtered_results_df_model1 = results_df_model1[results_df_model1['차이'] != 0]
            st.dataframe(filtered_results_df_model1)

    with col4:
        st.subheader("수학흥미도 예측결과")

        # Load Model 2
        model_path2 = 'C:/Users/namch/OneDrive/문서/streamlit/models/model2.pkl'
        if os.path.exists(model_path2):
            with open(model_path2, 'rb') as f2:
                model2_serialized_loaded, model2_weights_loaded = joblib.load(f2)

            # Reconstruct Model 2
            model2_loaded = tf.keras.models.model_from_json(model2_serialized_loaded)
            model2_loaded.set_weights(model2_weights_loaded)

            # Load scalers for Model 2
            scaler_W_loaded = joblib.load('C:/Users/namch/OneDrive/문서/streamlit/models/scaler_W.pkl')
            scaler_z_loaded = joblib.load('C:/Users/namch/OneDrive/문서/streamlit/models/scaler_z.pkl')

            # Prepare input data for Model 2
            input_data_model2 = {k: v for k, v in input_data.items() if k != 'SMATINT'}
            columns2 = list(input_data_model2.keys())
            columns_sorted2 = scaler_W_loaded.feature_names_in_
            input_data_model2_sorted = {key: input_data_model2[key] for key in columns_sorted2}

            # Prediction for Model 2
            prediction_model2 = predict_with_input(model2_loaded, scaler_W_loaded, scaler_z_loaded,
                                                   input_data_model2_sorted, columns_sorted2)
            # prediction_model2 값에서 숫자만 추출하고 소수점 2자리까지 반올림
            if isinstance(prediction_model2, (list, np.ndarray)):
                prediction_value_model2 = round(prediction_model2.item(), 2)
            else:
                prediction_value_model2 = round(prediction_model2, 2)

            st.session_state.change_history['math_interest'].append(prediction_value_model2)

            # CSS 스타일 적용하여 결과 출력
            st.markdown(
                f"""
                <style>
                .prediction-container {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                }}
                .prediction-box {{
                    text-align: center;
                    color: white;
                    font-size: 15pt;
                    background: linear-gradient(90deg, #f63366, #32CD32);  /* 슬라이더색과 연두색의 그라데이션 */
                    border-radius: 10px;
                    padding: 10px;
                    display: inline-block;
                    margin-bottom: 20px;  /* 아래 여백 추가 */
                }}
                </style>
                <div class="prediction-container">
                    <div class="prediction-box">{prediction_value_model2}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            plot_changes(st.session_state.change_history['math_interest'], 'math interest')
        else:
            st.write("모델 2 파일이 존재하지 않습니다. 모델 2 파일을 확인해주세요.")
        if 'scaler_W_loaded' in locals() and 'scaler_z_loaded' in locals():
            input_data_df_model2 = pd.DataFrame([input_data_model2], columns=scaler_W_loaded.feature_names_in_)
            input_data_scaled_model2 = scaler_W_loaded.transform(input_data_df_model2)
            optimize_indices_model2 = [scaler_W_loaded.feature_names_in_.tolist().index(var) for var in
                                       variables_to_optimize]

            optimal_input_scaled_model2 = tf.Variable(input_data_scaled_model2.copy(), dtype=tf.float32)
            optimizer_model2 = Adam(learning_rate=0.01)

            @tf.function
            def optimize_step_model2():
                with tf.GradientTape() as tape:
                    prediction = model2_loaded(optimal_input_scaled_model2, training=False)
                    loss = -prediction  # maximize prediction by minimizing negative prediction
                gradients = tape.gradient(loss, [optimal_input_scaled_model2])[0]
                updates = tf.zeros_like(optimal_input_scaled_model2)
                for i in optimize_indices_model2:
                    updates = tf.tensor_scatter_nd_update(updates, [[0, i]], [gradients[0, i]])
                optimizer_model2.apply_gradients(zip([updates], [optimal_input_scaled_model2]))
                return loss

            for i in range(iterations):
                loss = optimize_step_model2()

            optimal_inputs_model2 = scaler_W_loaded.inverse_transform(optimal_input_scaled_model2.numpy())
            optimal_prediction_scaled_model2 = model2_loaded.predict(optimal_input_scaled_model2)
            optimal_prediction_model2 = scaler_z_loaded.inverse_transform(
                optimal_prediction_scaled_model2.reshape(-1, 1))

            difference_scaled_model2 = optimal_input_scaled_model2.numpy() - input_data_scaled_model2
            difference_model2 = optimal_inputs_model2 - input_data_df_model2.values

            original_pred_value_model2 = prediction_model2
            optimal_pred_value_model2 = optimal_prediction_model2[0, 0]
            difference_value_model2 = optimal_pred_value_model2 - original_pred_value_model2
            percentage_increase_model2 = (difference_value_model2 / original_pred_value_model2) * 100

            original_pred_value_model2 = round(float(original_pred_value_model2), 2)
            optimal_pred_value_model2 = round(float(optimal_pred_value_model2), 2)
            difference_value_model2 = round(float(difference_value_model2), 2)
            percentage_increase_model2 = round(float(percentage_increase_model2[0]), 2)

            st.markdown(
                f"""
                                    <style>
                                    .result-container {{
                                        display: flex;
                                        flex-direction: column;
                                        align-items: center;
                                        margin-bottom: 20px;  /* 아래 여백 추가 */
                                    }}
                                    .result-text {{
                                        text-align: center;
                                        margin-bottom: 5px;  /* 텍스트와 숫자 사이 여백 추가 */
                                    }}
                                    .result-box {{
                                        text-align: center;
                                        color: #f63366;  /* 주황색과 슬라이더 색의 그라데이션 */
                                        font-size: 15pt;
                                        background-color: white;
                                        border-radius: 10px;
                                        padding: 5px 10px;
                                        display: inline-block;
                                        margin-bottom: 10px;  /* 아래 여백 추가 */
                                    }}
                                    </style>
                                    <div class="result-container">
                                        <div class="result-text">지금 조건에서는 수학흥미도가 이정도인거 같아요</div>
                                        <div class="result-box">{original_pred_value_model2}</div>
                                    </div>
                                    <div class="result-container">
                                        <div class="result-text">현재 상황에서 어디까지 늘려볼 수 있을까요?</div>
                                        <div class="result-box">{optimal_pred_value_model2}</div>
                                    </div>
                                    <div class="result-container">
                                        <div class="result-text">교수학습전략을 수정해보면 얼마나 늘릴 수 있을까요?</div>
                                        <div class="result-box">{difference_value_model2}</div>
                                    </div>
                                    <div class="result-container">
                                        <div class="result-text">몇 퍼센트나 개선되는 걸까요?</div>
                                        <div class="result-box">{percentage_increase_model2}%</div>
                                    </div>
                                    """,
                unsafe_allow_html=True
            )

            st.write(f'교수학습전략을 어떻게 수정해봐야 할까요?')

            results_df_model2 = pd.DataFrame({
                '전략': scaler_W_loaded.feature_names_in_,
                '현재': np.round(input_data_df_model2.values.flatten(), 0),
                '최적값': np.round(optimal_inputs_model2.flatten(), 0),
                '차이': np.round(difference_model2.flatten(), 0)
            })

            filtered_results_df_model2 = results_df_model2[results_df_model2['차이'] != 0]
            st.dataframe(filtered_results_df_model2)


if __name__ == '__main__':
    run_ml_app()

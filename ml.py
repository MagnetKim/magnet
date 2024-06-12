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
    st.title("🪞Project Mirror✨")

    if 'change_count' not in st.session_state:
        st.session_state.change_count = 0
    if 'change_history' not in st.session_state:
        st.session_state.change_history = {'academic_achievement': [], 'math_interest': []}

    st.markdown(
        """
        <style>
        .custom-column {
            padding-left: 10px;
            padding-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.markdown('<div class="custom-column">', unsafe_allow_html=True)
        st.subheader("Step 1. 현재 상황을 알려주세요")
        st.markdown(
            """
            <div style="border-radius: 10px; background-color: #f9f9f9; padding: 10px; color: black; margin-bottom: 20px; ">
                📝 현재 선생님의 학교, 담당 학급, 직무, 개인상황을 알려주세요.
            </div>
            """, unsafe_allow_html=True
        )
        # User input sliders for 기타 변수
        input_data_others = {}
        st.write("")
        st.subheader("🏫학교 정보")
        
        # Academic Achievement
        st.write("🟠학업성취도 수준을 선택하세요.")
        matach = st.slider('625점 이상: ‘수월’, 550점: ‘우수’, 475점: ‘보통’, 400점: ‘기초’', 0, 1000, 500, key='MATACH')
        
        # Non-Korean Speaking Students Ratio
        st.write("🟠한국어가 모국어가 아닌 학생 비율을 선택하세요.")
        schkor = st.slider('1: 10% 미만, 2: 10-25%, 3:25%-50%, 4:50-75%, 5: 75% 이상', 1, 5, 1, key='SCHKOR')
        
        # Population Around School
        st.write("🟠학교 주변 인구를 선택하세요.")
        schpop = st.slider('1: 3천명 이하, 2: 3천-1.5만, 3: 1.5만-3만, 4: 3만-5만, 5: 5만-10만, 6: 10만-50만, 7: 50만 이상', 1, 7, 1, key='SCHPOP')
        
        # Economically Challenged Students Ratio
        st.write("🟠경제적으로 어려운 학생 비율을 선택하세요.")
        schpoor = st.slider('1: 0-10%, 2: 10-25%, 3: 25-50%, 4: 50%-', 1, 5, 1, key='SCHPOOR')
        
        # Gender Ratio of Students
        st.write("🟠학생 성별 비율을 선택하세요.")
        sgender = st.slider('0: 남자만, 반반: 0.5, 1: 여자만', 0.0, 1.0, 1.0, step=0.1, key='SGENDER')
        
        # Average Education Level of Guardians
        st.write("🟠전반적인 학생들의 보호자 평균 학력을 선택하세요.")
        sgaredu = st.slider('1: 초등이하, 2:중졸, 3:고졸, 4-5: 전문대, 6: 4년제졸, 7-8: 석박사', 1, 8, 5, step=1, key='SGAREDU')
        
        # Educational Aspirations of Students
        st.write("🟠학생들의 평균적인 교육 포부 수준을 선택하세요.")
        seduaspr = st.slider('1:중졸, 2:고졸, 3-4:전문대졸, 5:학사, 6:대학원', 1, 6, 1, key='SEDUASPR')
        
        # Interest in Mathematics
        st.write("🟠학생들의 수학 흥미 수준을을 선택하세요.")
        smatint = st.slider('1: 매우 낮음, 2: 낮음, 3: 높음, 4: 매우 높음', 1, 4, 1, key='SMATINT')
        
        # Self-Efficacy in Mathematics
        st.write("🟠학생들의 수학 효능/자신감 수준을 선택하세요.")
        smateff = st.slider('1: 매우 낮음, 2: 낮음, 3: 높음, 4: 매우 높음', 1, 4, 1, key='SMATEFF')
        
        # Perception of Teacher's Capability
        st.write("🟠학생들이 교사 능력에 대해 어떻게 인식하는지 선택하세요.")
        stchrcap = st.slider('1: 매우 낮음, 2: 낮음, 3: 높음, 4: 매우 높음', 1, 4, 1, key='STCHRCAP')
        
        input_data_others = {
            'MATACH': matach,
            'SCHKOR': schkor,
            'SCHPOP': schpop,
            'SCHPOOR': schpoor,
            'SGENDER': sgender,
            'SGAREDU': sgaredu,
            'SEDUASPR': seduaspr,
            'SMATINT': smatint,
            'SMATEFF': smateff,
            'STCHRCAP': stchrcap
        }
        st.write("")
        st.subheader("📒교사 정보")
               # Teacher Information
        st.write("🟡교사경력을 선택하세요.")
        tcareer = st.slider('총 경력 연수', 1, 60, 1, key='TCAREER')
        
        st.write("🟡교사성별을 선택하세요.")
        tgender = st.slider('남자:1, 여자:0', 0, 1, 0, key='TGENDER')
        
        st.write("🟡교사 연령을 선택하세요.")
        tage = st.slider('만 기준', 20, 70, 20, key='TAGE')
        
        st.write("🟡수학과전공 여부를 선택하세요.")
        tmajmath = st.slider('수학과 전공이면 1', 0, 1, 0, key='TMAJMATH')
        
        st.write("🟡수학교육과전공 여부를 선택하세요.")
        tmajme = st.slider('수학교육과전공이면 1', 0, 1, 1, key='TMAJME')
        
        st.write("🟡학생 공부에 대한 교사의 기대를 입력하세요.")
        tstutexp = st.slider('1:매우 낮음 - 5:매우 높음', 1, 5, 1, key='TSTUTEXP')
        
        st.write("🟡학생 공부에 대한 학부모의 기대를 선택하세요.")
        tstupexp = st.slider('1:매우 낮음 - 5:매우 높음', 1, 5, 1, key='TSTUPEXP')
        
        st.write("🟡학생들은 학교에서 잘 지내려고 한다")
        tstusexp = st.slider('1:매우 아니다 - 5:매우 그렇다', 1, 5, 1, key='TSTUSEXP')
        
        st.write("🟡선생님이 생각하시는 학생들의 역량을 선택하세요.")
        tstucap = st.slider('1:매우 낮음 - 5:매우 높음', 1, 5, 1, key='TSTUCAP')
        
        st.write("🟡학생들은 교사들을 존중하나요?")
        tstures = st.slider('1:매우 아니다 - 5:매우 그렇다', 1, 5, 1, key='TSTURES')
        
        input_data_others.update({
            'TCAREER': tcareer,
            'TGENDER': tgender,
            'TAGE': tage,
            'TMAJMATH': tmajmath,
            'TMAJME': tmajme,
            'TSTUTEXP': tstutexp,
            'TSTUPEXP': tstupexp,
            'TSTUSEXP': tstusexp,
            'TSTUCAP': tstucap,
            'TSTURES': tstures
        })
        st.write("")
        st.subheader("🥰교사 만족도")
        # Teacher Satisfaction
        st.write("🟠선생님들간의 협력 정도를 선택하세요.")
        tcollab = st.slider('1: 매우 낮음 - 4: 매우 높음', 1, 4, 1, key='TCOLLAB')
        
        st.write("🟠선생님의 직무만족도를 선택하세요.")
        tsatis = st.slider('1: 매우 낮음 - 4: 매우 높음', 1, 4, 1, key='TSATIS')
        
        st.write("🟠선생님의 직무에 대한 열정을 선택하세요.")
        tpassion = st.slider('1: 매우 낮음 - 4: 매우 높음', 1, 4, 1, key='TPASSION')
        
        st.write("🟠선생님은 담당하는 학생이 너무 많다고 생각하시나요?")
        tcrwd = st.slider('1: 전혀 비동의 - 4: 매우 동의', 1, 4, 1, key='TCRWD')
        
        st.write("🟠선생님의 행정업무양이 많다고 생각하시나요?")
        toveradm = st.slider('1: 전혀 비동의 - 4: 매우 동의', 1, 4, 1, key='TOVERADM')
        
        input_data_others.update({
            'TCOLLAB': tcollab,
            'TSATIS': tsatis,
            'TPASSION': tpassion,
            'TCRWD': tcrwd,
            'TOVERADM': toveradm
        })
        st.write("")
        st.subheader("🏃‍♂️교사 연수와 전문성 개발")
        # Teacher Development
        st.write("🟡수학교과 내용에 대한 연수를 받은 적이 있는지 선택하세요.")
        tpdmcp = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDMCP')
        
        st.write("🟡수학교과 내용에 대한 연수를 받을 계획이 있는지 선택하세요.")
        tpdmdf = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDMCF')
        
        st.write("🟡수학교육학에 대한 연수를 받은 적이 있는지 선택하세요.")
        tpdmdp = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDMDP')
        
        st.write("🟡수학교육학에 대한 연수를 받을 계획이 있는지 선택하세요.")
        tpdmcf = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDMDF')
        
        st.write("🟡수학교육과정에 대한 연수를 받은 적이 있는지 선택하세요.")
        tpdmep = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDMEP')
        
        st.write("🟡수학교육과정에 대한 연수를 받을 계획이 있는지 선택하세요.")
        tpdmef = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDMEF')
        
        st.write("🟡수학 문제해결 역량에 대한 연수를 받은 적이 있는지 선택하세요.")
        tpdsvp = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDSVP')
        
        st.write("🟡수학 문제해결 역량에 대한 연수를 받을 계획이 있는지 선택하세요.")
        tpdsvf = st.slider('1: 있음, 0: 없음', 0, 1, 0, key='TPDSVF')
        
        st.write("🟡전문성 향상에 투자하는 시간을 선택하세요.")
        tpdtime = st.slider('1: 있음, 0: 없음', 1, 5, 0, key='TPDTIME')
        
        input_data_others.update({
            'TPDMCP': tpdmcp,
            'TPDMCF': tpdmcf,
            'TPDMDP': tpdmdp,
            'TPDMDF': tpdmdf,
            'TPDMEP': tpdmep,
            'TPDMEF': tpdmef,
            'TPDSVP': tpdsvp,
            'TPDSVF': tpdsvf,
            'TPDTIME': tpdtime
        })
        st.write("")
        st.subheader("🪑학습환경")
        # Student Learning
        st.write("🟠학급 내 학생 수를 선택하세요.")
        tstupc = st.slider('인원수 명', 1, 50, 28, key='TSTUPC')
        
        st.write("🟠수학지식이 부족한 학생 비율을 선택하세요.")
        tlackk = st.slider('1: 많음, 2: 약간, 3: 적음', 1, 3, 1, key='TLACKK')
        
        st.write("🟠수학 흥미가 부족한 학생 비율을 선택하세요.")
        tlacki = st.slider('1: 많음, 2: 약간, 3: 적음', 1, 3, 1, key='TLACKI')
        
        st.write("🟠1주일에 수학 시간을 선택하세요.")
        tmtime = st.slider('분 기준', 1, 1800, 700, key='TMTIME')
        
        input_data_others.update({
            'TSTUPC': tstupc,
            'TLACKK': tlackk,
            'TLACKI': tlacki,
            'TMTIME': tmtime
        })
        st.write("")
        st.subheader("📚학생들이 배운 교과내용")
        # Mathematics Curriculum
        st.write("🟡음수계산")
        tprik1 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK1')
        
        st.write("🟡분수와 소수")
        tprik2 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK2')
        
        st.write("🟡비율과 퍼센트")
        tprik3 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK3')
        
        st.write("🟡대수의 표현")
        tprik4 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK4')
        
        st.write("🟡단순 선형 방정식")
        tprik5 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK5')
        
        st.write("🟡단순 선형 부등식")
        tprik6 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK6')
        
        st.write("🟡연립방정식")
        tprik7 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK7')
        
        st.write("🟡선형/2차 함수 표현")
        tprik8 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK8')
        
        st.write("🟡함수의 특성")
        tprik9 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK9')
        
        st.write("🟡패턴을 표현하는 방법")
        tprik10 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK10')
        
        st.write("🟡각, 평형 모양에 대한 특성")
        tprik11 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK11')
        
        st.write("🟡지름, 둘레, 면적")
        tprik12 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK12')
        
        st.write("🟡피타고라스 정리")
        tprik13 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK13')
        
        st.write("🟡반사, 회전")
        tprik14 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK14')
        
        st.write("🟡합동과 닮음")
        tprik15 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK15')
        
        st.write("🟡3차원 도형")
        tprik16 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK16')
        
        st.write("🟡자료 해석법")
        tprik17 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK17')
        
        st.write("🟡자료 수집을 위한 절차")
        tprik18 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK18')
        
        st.write("🟡자료의 조직법")
        tprik19 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK19')
        
        st.write("🟡자료 요약통계")
        tprik20 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK20')
        
        st.write("🟡단순한 사건의 확률")
        tprik21 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK21')
        
        st.write("🟡다양한 사건의 확률")
        tprik22 = st.slider('0: 아직 안 배움, 1: 작년에 배움, 2: 올해 배움', 0, 2, 0, key='TPRIK22')
        
        input_data_others.update({
            'TPRIK1': tprik1,
            'TPRIK2': tprik2,
            'TPRIK3': tprik3,
            'TPRIK4': tprik4,
            'TPRIK5': tprik5,
            'TPRIK6': tprik6,
            'TPRIK7': tprik7,
            'TPRIK8': tprik8,
            'TPRIK9': tprik9,
            'TPRIK10': tprik10,
            'TPRIK11': tprik11,
            'TPRIK12': tprik12,
            'TPRIK13': tprik13,
            'TPRIK14': tprik14,
            'TPRIK15': tprik15,
            'TPRIK16': tprik16,
            'TPRIK17': tprik17,
            'TPRIK18': tprik18,
            'TPRIK19': tprik19,
            'TPRIK20': tprik20,
            'TPRIK21': tprik21,
            'TPRIK22': tprik22
        })
    with col2:
        st.markdown('<div class="custom-column">', unsafe_allow_html=True)
        # User input sliders for 세부 변수
        input_data_details = {}
        st.subheader("Step 2. 교수학습 전략")
        st.markdown(
            """
            <div style="border-radius: 10px; background-color: #f9f9f9; padding: 10px; color: black; margin-bottom: 20px; ">
                📝 선생님이 생각하시는 교수학습 방법과 수준을 조정해보면서, 학업성취도와 수학흥미도 예측결과가 변동하는 것을 보며 수업설계를 참조하세요.
            </div>
            """, unsafe_allow_html=True
        )
        def add_slider_with_description(label, min_value, max_value, default_value, step=1, key=None, description=None):
            slider_value = st.slider(label, min_value, max_value, default_value, step=step, key=key)
            st.write(description)
            return slider_value
        st.write("")
        st.subheader("🧑‍🏫가르치는 방법")

        st.write("🟠실생활과 연결시켜 설명")
        tinsstg1 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSSTG1')
        
        st.write("🟠답에 대해 설명")
        tinsstg2 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSSTG2')
        
        st.write("🟠도전적인 활동을 제공")
        tinsstg3 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSSTG3')
        
        st.write("🟠반 친구들과 토론")
        tinsstg4 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSSTG4')
        
        st.write("🟠기존 지식의 연계")
        tinsstg5 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSSTG5')
        
        st.write("🟠문제해결과정 설명")
        tinsstg6 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSSTG6')
        
        st.write("🟠학생들의 생각을 표현하도록 함")
        tinsstg7 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSSTG7')
        st.write("")
        st.subheader("🎒학생들에게 수업에서")
        
        st.write("🟠수학 내용에 대한 교사의 설명을 듣도록 함")
        tinsask1 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK1')
        
        st.write("🟠교사의 문제풀이 방법을 듣도록 함")
        tinsask2 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK2')
        
        st.write("🟠법칙, 절차, 사실을 외우도록 함")
        tinsask3 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK3')
        
        st.write("🟠스스로 절차를 연습해보도록 함")
        tinsask4 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK4')
        
        st.write("🟠새로운 문제에 스스로 적용해보도록 함")
        tinsask5 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK5')
        
        st.write("🟠교사의 지도 아래 반 전체가 문제를 풀어보도록 함")
        tinsask6 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK6')
        
        st.write("🟠다양한 능력의 학생들끼리 협업하게 함")
        tinsask7 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK7')
        
        st.write("🟠비슷한 능력의 학생들끼리 협업하게 함")
        tinsask8 = st.slider('1:아예 적용 안 함, 2: 몇몇 수업에서 적용, 3: 주당 1-2번 이상 적용, 4: 거의 매 수업 적용', 1, 4, 1, key='TINSASK8')
        st.write("")
        st.subheader("⏰과제 빈도 및 시간")
        
        st.write("🟠과제를 얼마나 자주 내주십니까?")
        thwfrq = st.slider('1:안 냄, 2: 주 1회 이하, 3: 주 1-2회, 4: 주 3-4회, 5: 매일', 1, 5, 1, key='THWFRQ')
        
        st.write("🟠과제에 소요되는 시간은 어느 수준입니까?")
        thwtime = st.slider('1: 15분 미만, 2: 15-30분, 3: 30-60분, 4: 60-90분, 5: 90분 이상', 1, 5, 1, key='THWTIME')
        st.write("")
        st.subheader("📖과제 전략")
        
        st.write("🟠과제에 대한 피드백을 줌")
        thwstg1 = st.slider('1:거의 안함, 2: 가끔, 3: 거의 항상', 1, 3, 1, key='THWSTG1')
        
        st.write("🟠과제를 스스로 고치도록 함")
        thwstg2 = st.slider('1:거의 안함, 2: 가끔, 3: 거의 항상', 1, 3, 1, key='THWSTG2')
        
        st.write("🟠수업에서 과제에 대한 토론을 함")
        thwstg3 = st.slider('1:거의 안함, 2: 가끔, 3: 거의 항상', 1, 3, 1, key='THWSTG3')
        
        st.write("🟠과제가 완료되었는지 모니터링함")
        thwstg4 = st.slider('1:거의 안함, 2: 가끔, 3: 거의 항상', 1, 3, 1, key='THWSTG4')
        
        st.write("🟠과제를 성적에 활용함")
        thwstg5 = st.slider('1:거의 안함, 2: 가끔, 3: 거의 항상', 1, 3, 1, key='THWSTG5')
        st.write("")
        st.subheader("💯평가 전략")
        
        st.write("🟠학생을 관찰함")
        thwstg6 = st.slider('1:중요하지 않음, 2: 약간 중요함, 3: 매우 중요하게 생각함', 1, 3, 1, key='THWSTG6')
        
        st.write("🟠학생에게 지속적으로 질문함")
        thwstg7 = st.slider('1:중요하지 않음, 2: 약간 중요함, 3: 매우 중요하게 생각함', 1, 3, 1, key='THWSTG7')
        
        st.write("🟠짧은 형태의 평가를 활용함")
        thwstg8 = st.slider('1:중요하지 않음, 2: 약간 중요함, 3: 매우 중요하게 생각함', 1, 3, 1, key='THWSTG8')
        
        st.write("🟠긴 형태의 평가를 활용함")
        thwstg9 = st.slider('1:중요하지 않음, 2: 약간 중요함, 3: 매우 중요하게 생각함', 1, 3, 1, key='THWSTG9')
        
        st.write("🟠장기간의 프로젝트를 부여함")
        thwstg10 = st.slider('1:중요하지 않음, 2: 약간 중요함, 3: 매우 중요하게 생각함', 1, 3, 1, key='THWSTG10')
        
        input_data_details = {
            'TINSSTG1': tinsstg1,
            'TINSSTG2': tinsstg2,
            'TINSSTG3': tinsstg3,
            'TINSSTG4': tinsstg4,
            'TINSSTG5': tinsstg5,
            'TINSSTG6': tinsstg6,
            'TINSSTG7': tinsstg7,
            'TINSASK1': tinsask1,
            'TINSASK2': tinsask2,
            'TINSASK3': tinsask3,
            'TINSASK4': tinsask4,
            'TINSASK5': tinsask5,
            'TINSASK6': tinsask6,
            'TINSASK7': tinsask7,
            'TINSASK8': tinsask8,
            'THWFRQ': thwfrq,
            'THWTIME': thwtime,
            'THWSTG1': thwstg1,
            'THWSTG2': thwstg2,
            'THWSTG3': thwstg3,
            'THWSTG4': thwstg4,
            'THWSTG5': thwstg5,
            'THWSTG6': thwstg6,
            'THWSTG7': thwstg7,
            'THWSTG8': thwstg8,
            'THWSTG9': thwstg9,
            'THWSTG10': thwstg10
        }

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
        st.subheader("✨학업성취도 예측결과✨")
        st.markdown(
                """
                <div style="border-radius: 10px; background-color: #f9f9f9; padding: 10px; color: black; margin-bottom: 20px; ">
                    📝 Mirror가 산출한 결과입니다. 625점 이상은 ‘수월수준’, 550점 이상은 ‘우수수준 이상’, 475점 이상은 ‘보통수준 이상’, 400점 이상은 ‘기초수준 이상’으로 볼 수 있습니다.
                </div>
                """, unsafe_allow_html=True
        )

        # Load Model 1
        model_path1 = 'models/model1.pkl'
        if os.path.exists(model_path1):
            with open(model_path1, 'rb') as f:
                model1_serialized_loaded, model1_weights_loaded = joblib.load(f)

            # Reconstruct Model 1
            model1_loaded = tf.keras.models.model_from_json(model1_serialized_loaded)
            model1_loaded.set_weights(model1_weights_loaded)

            # Load scalers for Model 1
            scaler_X_loaded = joblib.load('models/scaler_X.pkl')
            scaler_y_loaded = joblib.load('models/scaler_y.pkl')

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

            variable_name_mapping = {
                'TINSSTG1': '실생활과 연결시켜 설명',
                'TINSSTG2': '답에 대해 설명',
                'TINSSTG3': '도전적인 활동을 제공',
                'TINSSTG4': '반 친구들과 토론',
                'TINSSTG5': '기존 지식의 연계',
                'TINSSTG6': '문제해결과정 설명',
                'TINSSTG7': '학생들의 생각을 표현하도록 함',
                'TINSASK1': '수학 내용에 대한 교사의 설명을 듣도록 함',
                'TINSASK2': '교사의 문제풀이 방법을 듣도록 함',
                'TINSASK3': '법칙, 절차, 사실을 외우도록 함',
                'TINSASK4': '스스로 절차를 연습해보도록 함',
                'TINSASK5': '새로운 문제에 스스로 적용해보도록 함',
                'TINSASK6': '교사의 지도 아래 반 전체가 문제를 풀어보도록 함',
                'TINSASK7': '다양한 능력의 학생들끼리 협업하게 함',
                'TINSASK8': '비슷한 능력의 학생들끼리 협업하게 함',
                'THWFRQ': '과제를 얼마나 자주 내주십니까?',
                'THWTIME': '과제에 소요되는 시간은 어느 수준입니까?',
                'THWSTG1': '과제에 대한 피드백을 줌',
                'THWSTG2': '과제를 스스로 고치도록 함',
                'THWSTG3': '수업에서 과제에 대한 토론을 함',
                'THWSTG4': '과제가 완료되었는지 모니터링함',
                'THWSTG5': '과제를 성적에 활용함',
                'THWSTG6': '학생을 관찰함',
                'THWSTG7': '학생에게 지속적으로 질문함',
                'THWSTG8': '짧은 형태의 평가를 활용함',
                'THWSTG9': '긴 형태의 평가를 활용함',
                'THWSTG10': '장기간의 프로젝트를 부여함'
            }
            
            st.write(f'🔍교수학습전략을 어떻게 수정해봐야 할까요?')
            
            # 예제 데이터프레임
            results_df_model1 = pd.DataFrame({
                '전략': scaler_X_loaded.feature_names_in_,
                '현재': np.round(input_data_df_model1.values.flatten(), 0),
                '최적값': np.round(optimal_inputs_model1.flatten(), 1),
                '차이': np.round(difference_model1.flatten(), 1)
            })
            
            # 변수명 매핑 적용
            results_df_model1['전략'] = results_df_model1['전략'].map(variable_name_mapping)
            
            # difference 항이 0이 아닌 행만 필터링
            filtered_results_df_model1 = results_df_model1[results_df_model1['차이'].abs() > 0.3]
            
            # 인덱스를 가리고 결과 출력
            st.dataframe(filtered_results_df_model1.reset_index(drop=True))

            st.markdown(
                    """
                    <div style="border-radius: 10px; background-color: #f9f9f9; padding: 10px; color: black; margin-bottom: 20px; ">
                        위 표에서는 현 조건에서 가장 좋은 점수를 받을 수 있는 방안을 소개해줍니다. 이를 활용해서 교수학습활동을 기획해보세요.   
                    </div>
                    """, unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="border-radius: 10px; background-color: #f9f9f9; padding: 20px; color: black;">
                    👀👀 Mirror에서 결과에 미치는 중요한 전략 목록<br>
                    (중요도 Top 15 순서대로)<br>
                    <ol>
                        <li>새로운 문제에 스스로 적용하기</li>
                        <li>학생들에게 지속적으로 질문</li>
                        <li>기존 지식과 연계하기</li>
                        <li>학생들을 관찰</li>
                        <li>수업에서 과제에 대한 토론</li>
                        <li>과제를 성적에 활용</li>
                        <li>과제완료 모니터링</li>
                        <li>법칙, 절차, 사실 외우게 하기</li>
                        <li>답에 대한 설명</li>
                        <li>과제에 대한 피드백</li>
                        <li>짧은 형태의 평가를 활용함</li>
                        <li>과제 빈도</li>
                        <li>과제하는 시간</li>
                        <li>교사의 문제풀이 방법 듣게 하기</li>
                        <li>반 친구들과 토론하게 하기</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True
            )
    

    with col4:
        st.subheader("🪄수학흥미도 예측결과")
        st.markdown(
                """
                <div style="border-radius: 10px; background-color: #f9f9f9; padding: 10px; color: black; margin-bottom: 20px; ">
                    📝 Mirror가 산출한 결과입니다. 수학흥미도의 평균은 2.5입니다. 3을 넘으면 높은 수준, 2보다 낮아지면 낮은 수준으로 볼 수 있습니다.
                </div>
                """, unsafe_allow_html=True
        )

        # Load Model 2
        model_path2 = './models/model2.pkl'
        if os.path.exists(model_path2):
            with open(model_path2, 'rb') as f2:
                model2_serialized_loaded, model2_weights_loaded = joblib.load(f2)

            # Reconstruct Model 2
            model2_loaded = tf.keras.models.model_from_json(model2_serialized_loaded)
            model2_loaded.set_weights(model2_weights_loaded)

            # Load scalers for Model 2
            scaler_W_loaded = joblib.load('./models/scaler_W.pkl')
            scaler_z_loaded = joblib.load('./models/scaler_z.pkl')

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
                    background: linear-gradient(90deg, #FFA500, #FF4500);  
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

            st.markdown(
                """
                <div style="border-radius: 10px; background-color: #f9f9f9; padding: 10px; color: black; margin-bottom: 20px; ">
                    📝 예측결과 해석 주의!
                </div>
                """, unsafe_allow_html=True
            )

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

            st.write(f'🔍교수학습전략을 어떻게 수정해봐야 할까요?')
            
            # 예제 데이터프레임
            results_df_model2 = pd.DataFrame({
                '전략': scaler_W_loaded.feature_names_in_,
                '현재': np.round(input_data_df_model2.values.flatten(), 0),
                '최적값': np.round(optimal_inputs_model2.flatten(), 1),
                '차이': np.round(difference_model2.flatten(), 1)
            })
            
            # 변수명 매핑 적용
            results_df_model2['전략'] = results_df_model2['전략'].map(variable_name_mapping)
            
            # difference 항이 0이 아닌 행만 필터링
            filtered_results_df_model2 = results_df_model2[results_df_model2['차이'].abs() > 0.3]
            
            # 인덱스를 가리고 결과 출력
            st.dataframe(filtered_results_df_model2.reset_index(drop=True))

            st.markdown(
                    """
                    <div style="border-radius: 10px; background-color: #f9f9f9; padding: 10px; color: black; margin-bottom: 20px; ">
                        위 표에서는 현 조건에서 가장 좋은 점수를 받을 수 있는 방안을 소개해줍니다. 이를 활용해서 교수학습활동을 기획해보세요.   
                    </div>
                    """, unsafe_allow_html=True
            )

if __name__ == '__main__':
    run_ml_app()



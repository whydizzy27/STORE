# 추천 시스템

분야에 따른 논문 추천의 경우 총 43개의 분야가 있어 2개 이상의 레이블을 가지고 있으므로 **다중 분류 (Multiclass classification)** 의 예입니다. 각 데이터의 포인터가 정확히 하나의 범주로 분류되기 때문에 **단일 레이블 다중 분류 (single-label, multiclass classification)** 문제입니다.



## 📌 목차

* [시작하기](#-시작하기)

  * [준비하기](#준비하기)

* [워크플로우](#-워크플로우)

* [논문 데이터](#-논문-데이터)

* [데이터 정제](#-데이터-정제)

  * [텍스트 전처리](#텍스트-전처리)
    * [정제 (Cleaning) & 정규화 (Normalization) & 토큰화 (Tokenization)](#정제-(Cleaning)-&-정규화-(Normalization)-&-토큰화-(Tokenization))
  * [모델링을 위한 데이터 전처리](#모델링을-위한-데이터-전처리)
    * [데이터 구분](#데이터-구분)
    * [데이터 변환](#데이터-변환)

* [분류 모델 생성](#-분류-모델-생성)

* [분류 모델 평가](#-분류-모델-평가)

* [추천 알고리즘](#-추천-알고리즘)

  * [콘텐츠 기반 필터링](#-콘텐츠-기반-필터링)
  * [아이템 기반 협업 필터링](#아이템-기반-협업-필터링)
    * [데이터](#데이터)
  * [잠재 요인 협업 필터링](잠재-요인-협업-필터링)
    * [데이터](데이터)

* [참고](#참고)

  

## :runner: 시작하기

### 준비하기

해당 알고리즘을 사용하기 위해서는 다음과 같은 라이브러리를 설치합니다:

```shell
$ pip install -r requirements.txt
```

```
# requirements.txt
numpy==1.19.1
pandas==1.1.1
tensorflow==2.0.0
keras==2.3.1
matplotlib==3.3.1
nltk==3.5
scikit-learn==0.23.2
pandas-profiling==2.9.0
```

## 💻 워크플로우

아래와 같은 워크플로우로 추천 알고리즘 시스템을 구현하였습니다.

1. **수집(Acquisition)**: 크롤링 및 OpenAPI를 사용하여 논문 데이터를 수집합니다.
2. **점검 및 탐색(Inspection and exploration)**: 탐색적 데이터 분석(Exploratory Data Analysis, EDA)를 통해 데이터를 어떻게 정제해야하는지 파악합니다.
3. **전처리 및 정제(Processing and Cleaning)**: 토큰화, 정제, 정규화, 불용어 제거 등의 단계를 통해 데이터 전처리를 수행합니다.
4. **모델링 및 훈련(Modeling and Training)**: 전처리가 완료된 데이터를 통해 모델을 학습시킵니다. 이 때, 전체데이터 중 8:2의 비율로 훈련 데이터와 테스트 데이터를 나눴습니다. 또한, 훈련 데이터 중 8:2의 비율로 훈련 데이터와 검증 데이터를 나눴습니다. (Training : Validation : Testing = 6.4 : 1.6 : 2.0)
5. **평가(Evaluation)**: 테스트 데이터로 성능을 평가합니다.
6. **배포(Deployment)**: 완성된 모델을 배포합니다.

## :page_with_curl: 논문 데이터

논문 데이터는 **KCI** (https://www.kci.go.kr/kciportal/main.kci) 홈페이지에서 크롤링 및 OpenAPI를 사용하여 수집한 데이터를 사용하겠습니다. 총 43개의 분류로 나뉘며 어떤 분류는 다른 것에 비해 데이터가 많습니다. 각 분류는 훈련 세트에 평균23개 이상의 샘플을 가지고 있습니다.

![image-20200929021753979](C:\Users\multicampus\AppData\Roaming\Typora\typora-user-images\image-20200929021753979.png)



## :gear: 데이터 정제

### 텍스트 전처리

#### 정제 (Cleaning) & 정규화 (Normalization) & 토큰화 (Tokenization)

정제란 가지고 있는 텍스트로부터 노이즈를 제거하는 역할을 하며, 정규화는 표현 방법이 다른 단어들을 통합시켜 같은 단어로 만들어주는 역할을 합니다. 주어진 말뭉치에 대해 토큰이라 불리는 단위로 나누는 작업을 토큰화라 얘기하며, 해당 알고리즘에서는 토큰의 단위를 단어로 만들기 위해 단어 토큰화(word tokenization)를 수행합니다. 아래와 같은 방법으로 데이터에 정제와 정규화 및 토큰화를 진행했습니다.

1. abstract의 특수문자, 길이가 짧은 단어를 제거하고 소문자로 변환합니다.

   ```python
   df["abstract_clean"] = df["abstract"].str.replace("[^a-zA-Z]", " ")
   df["abstract_clean"] = df["abstract_clean"].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
   df["abstract_clean"] = df["abstract_clean"].apply(lambda x: x.lower())
   ```

2. null값 및 empty값을 확인해 제거합니다.

   ```python
   # Null 값 및 empty값 확인
   df["abstract_clean"].isnull().values.any()
   df.replace("", float("NaN"), inplace=True)
   df["abstract_clean"].isnull().values.any()
   df.dropna(inplace=True)
   ```

3. label이 없는 값을 확인해 제거합니다.

   ```python
   df = df[df.label != 0]
   ```

4. nltk 자연어 처리 패키지를 이용하여 'is', 'the', 'a'와 같은 불용어(stopwords)를 삭제합니다.

   ```python
   from nltk.corpus import stopwords
   # 불용어 제거
   stop_words = stopwords.words('english')
   tokenized_doc = df['abstract_clean'].apply(lambda x: x.split())
   tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
   tokenized_doc = tokenized_doc.to_list()
   ```

5. 단어가 1개 이하인 샘플의 인덱스를 찾아서 저장하고, 해당 샘플들은 제거합니다.

   ```python
   drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
   tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)
   df['tokenized_doc'] = tokenized_doc
   ```

### 모델링을 위한 데이터 전처리

#### 데이터 구분

모델의 학습을 위해 훈련 데이터, 검증 데이터, 테스트 데이터를 구분합니다. 랜덤하게 섞은 전체데이터 중 8:2의 비율로 훈련 데이터와 테스트 데이터를 나눴습니다. 또한, 훈련 데이터 중 8:2의 비율로 훈련 데이터와 검증 데이터를 나눴습니다. (Training : Validation : Testing = 6.4 : 1.6 : 2.0)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['tokenized_doc'], df['label'], test_size= 0.2, random_state=1234)
# X_train, y_train: 훈련 데이터, 훈련 데이터 정답
# X_test, y_test: 테스트 데이터, 테스트 데이터 정답
```

#### 데이터 변환

정제, 정규화, 토큰화를 통해 전처리된 데이터를 모델링을 위해 벡터로 변환합니다. 레이블을 벡터로 바꾸기 위해서 레이블 리스트를 정수 텐서로 변환하는 원-핫 인코딩(one-hot encoding)을 사용했습니다. 이 때,  정수화된 데이터 중 가장 긴 길이인 398에 맞춰 정규화 및 벡터화시킵니다.

![image-20200929030524195](C:\Users\multicampus\AppData\Roaming\Typora\typora-user-images\image-20200929030524195.png)

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

# 문자열을 정수 인덱스의 리스트로 변환
tokenizer = Tokenizer(num_words=35000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# 데이터 정규화
X_train = pad_sequences(X_train,maxlen=350)
X_test = pad_sequences(X_test,maxlen=350)
y_train = to_categorical(np.asarray(y_train))
y_test = to_categorical(np.asarray(y_test))

# 데이터 벡터 변환
def vectorize_sequences(sequences, dimension=35000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

X_train = vectorize_sequences(X_train) # 훈련 데이터 벡터 변환
X_test = vectorize_sequences(X_test) # 테스트 데이터 벡터 변환
```

> **Tip**
>
> One-Hot Encoding?
>
> 문자를 숫자로 바꾸는 여러가지 기법 중 하나입니다. 중복되지 않는 단어들을 가진 단어 집합(vocabulary)에 있는 단어를 가지고 문자를 숫자(즉, 벡터)로 바꿉니다.
>
> (1) 각 단어에 고유한 인덱스를 부여합니다. (정수 인코딩)
> (2) 표현하고 싶은 단어의 인덱스의 위치(class)에 1을 부여하고, 다른 단어의 인덱스의 위치(class)에는 0을 부여합니다.

## :book: 분류 모델 생성

학습을 위해 relu 활성화 함수를 사용한 완전 연결 층을 사용했습니다. loss 함수는 categorical_crossentropy를 사용하였고 rmsprop 옵티마이저를 사용했습니다.

* 총 4개의 Dense 층을 사용해 각 층마다 차례대로 512, 256, 128개의 은닉 유닛(hidden unit)을 사용했고, relu활성화 함수를 이용해 다음 텐서 연산을 연결했습니다. 
* 마지막 층의 44개는 44차원(43개의 분류 + 0값을 가진 label) 의 벡터를 출력합니다. 
* 마지막 층에 softmax 활성화 함수가 사용되어 44개의 출력 클래스에 대한 확률 분포를 출력합니다. 즉, 44개의 출력 벡터를 만들며 output[i]는 어떤 샘플이 클래스 i에 속할 확률로 모두 더하면 1이됩니다.

```python
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(35000,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(44, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 30, batch_size=50, validation_split=0.2)
```

## :chart_with_upwards_trend: 분류 모델 평가

```python
results = model.evaluate(X_test, y_test)
```

## :link: 추천 알고리즘

### 컨텐츠 기반 필터링

추천 시스템 중 콘텐츠 기반 필터링 (Content Based Filtering)을 구현합니다. 콘텐츠 기반 필터링의 경우, 사용자가 특정 아이템을 선호하는 경우 그 아이템과 비슷한 콘텐츠를 가진 다른 아이템을 추천해줍니다. ㅇㅇ시스템에서는 회원가입 시 선호하는 분류 3개를 선택해 사용자의 선호도를 알 수 있습니다. 또한, 사용자가 요약을 위해 업로드한 논문을 앞서 학습시킨 분류기 모델을 통해 분야를 파악하고 해당 분야에서 내용이 비슷한 논문을 추천해줍니다.

1. 요약된 논문의 abstract를 분류기를 통해 분야를 파악합니다.
2. 해당 분야에서 keyword, title, scrap, quote를 전처리를 통해 스크랩수와 인용수가 많은 논문일수록 더 높은 점수를 줍니다. 
3. 높은 점수를 가진 논문 500개 중, 'keyword'와 'title'의 내용이 비슷한 논문을 추천해줍니다. 이 때, 코사인 유사도(cosine similarity)가 높은 논문을 추천해주게 됩니다.

### 아이템 기반 협업 필터링

최근접 이웃 기반(nearest neighbor based collaborative filtering) 협업필터링은 사용자 기반의 협업 필터링(user based collaborative filtering)과 아이템 기반 협업 필터링(item based collaborative filtering)으로 나뉘어집니다.

저희가 사용한 아이템 기반 협업 필터링의 경우, item-user 행렬 내에서 itemA과 itemB가 유사한 평점 분포를 가지고 있다면 itemA와 itemB가 유사하다고 판단합니다.

#### 데이터

데이터에는 사용자가 스크랩한 논문 데이터 (reports_scraps 테이블)와 논문 데이터 (all_data.pkl)를 사용합니다.

### 잠재 요인 협업 필터링

잠재 요인 협업 필터링(latent factor collaborative filtering)은 행렬 분해(matrix factorization)을 기반해 사용합니다. 다차원 행렬을 SVD와 같은 차원 감소 기법으로 분해하는 과정에서 잠재 요인(latent factor)를 추출합니다. 이러한 행렬 분해를 하게되면 공간을 효율적으로 사용할 수 있습니다.

기존의 item-user 행렬을 user-latent, item-latent 행렬로 분해할 수 있습니다.

* item-user: *R(u, i)*

  * u번째 유저가 i번째 아이템 평가 점수

* user-latent: *P(u, k)*

* item-latent: *Q(i, k)*

  => *R(u, i)* ≒ *P(u, k)* * *Q.T(k, i)*

#### 데이터

데이터에는 사용자가 스크랩한 논문 데이터 (reports_scraps 테이블)와 논문 데이터 (all_data.pkl)를 사용합니다.

## :books: 참고

* [텐서 플로우 블로그](https://tensorflow.blog/%EC%BC%80%EB%9D%BC%EC%8A%A4-%EB%94%A5%EB%9F%AC%EB%8B%9D/3-5-%EB%89%B4%EC%8A%A4-%EA%B8%B0%EC%82%AC-%EB%B6%84%EB%A5%98-%EB%8B%A4%EC%A4%91-%EB%B6%84%EB%A5%98-%EB%AC%B8%EC%A0%9C/)
* [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)
* [꿈 많은 사람의 이야기](https://lsjsj92.tistory.com/563)
* https://lsjsj92.tistory.com/571?category=853217


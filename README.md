※ 본 프로젝트는 삼성청년소프트웨어아카데미의 허가 없이 소스 코드를 업로드할 수 없다는 점 양해 부탁드립니다.

# :page_with_curl: STORE - 논문 요약 서비스 (Summary Thesis Online & Recommend for Everyone)

![로고](https://user-images.githubusercontent.com/67194249/95681875-2a685e80-0c1d-11eb-84b2-547df6e3bb1d.png)

`STORE`은 논문 검색, 요약, 추천을 위한 웹페이지 입니다. 사용자가 요약을 원하는 논문을 `pdf` 파일로 업로드 시, 논문 요약 및 비슷한 논문을 추천받을 수 있습니다.

1. Common Service
   - 논문 검색 서비스를 제공합니다.
   - 관심있는 논문 스크랩 서비스를 제공합니다.
2. Summary Service
   - 논문 요약 서비스를 제공합니다.
   - 키워드, 이미지, 워드클라우드를 논문에서 추출해 표시합니다.
3. Recommend Service
   - 논문 추천 서비스를 제공합니다.
   - 요약된 논문과 비슷한 논문을 추천합니다.
   - 상세보기한 논문과 비슷한 논문을 추천합니다.

[여기]()를 클릭해 사이트를 확인하세요 :smile:
(현재 배포 기간이 종료되었습니다.)



## 📌 목차

[:page_with_curl: STORE - 논문 요약 서비스](#-store---논문-요약-서비스)

* [시작하기](#-시작하기)
  * [시작하기에 앞서](#시작하기에-앞서)
  * [설치하기](#설치하기)
  * [실행하기](#실행하기)
  * [배포하기](#배포하기)
  * [데모](#데모)
* [지원하는 브라우저](#-지원하는-브라우저)
* [사용된 도구](#-사용된-도구)
* [사용된 기술](#-사용된-기술)
  * [프론트엔드](#프론트엔드)
  * [백엔드](#백엔드)
  * [요약 알고리즘](#요약-알고리즘)
  * [추천 알고리즘](#추천-알고리즘)
* [Commit Convention](#-commit-convention)
* [저자](#-저자)
* [라이센스](#-라이센스)
* [참고](#참고)



## :runner: 시작하기

아래 방법을 따르시면 프로젝트를 실행시킬 수 있습니다.

### 시작하기에 앞서

* Windows 10
* Python 3.6.8
* Node.js 8.10.0
* SQLite3

### 설치하기

1. 깃헙의 레포지토리를 클론합니다.

   ```shell
   $ git clone https://lab.ssafy.com/s03-bigdata-sub3/s03p23a406.git
   ```

2. npm을 설치합니다.

   ```shell
   $ npm install
   ```

### 실행하기

`STORE` 서비스를 사용하기 위해서는 다음과 같은 방법으로 실행합니다:

1. **백엔드** 서버를 실행합니다.

   - IDE에 import 후 실행합니다.

     ```shell
     $ python manage.py makemigrations
     ```

     ```shell
     $ python manage.py migrate
     ```

     ```shell
     $ python manage.py runserver 8080
     ```

2. **프론트엔드**를 실행합니다.

   ```shell
   $ npm runserve
   ```

### 배포하기

해당 서비스는 `AWS EC2`를 이용하여 배포하였습니다. 사전에 [여기]()를 참고해서 `AWS EC2`계정을 생성하세요.

배포를 하기위해서는 다음과 같은 방법으로 실행합니다:

1. AWS EC2 인스턴스 생성

2. git repository clone

3. 필요한 패키지 다운로드

   ```shell
   $ pip3 install -r requirement.txt
   ```

4. JDK 설치 (환경변수 설정)

5. python manage.py runserver 0:8080 (서버 실행)

6. npm build (dist 폴더 생성)

### 데모

[여기]()를 클릭하세요.
(현재 배포 기간이 종료되었습니다.)



## :globe_with_meridians: 지원하는 브라우저

| 크롬   | 사파리 | edge   | firefox |
| ------ | ------ | ------ | ------- |
| latest | latest | latest | latest  |



## :hammer_and_wrench: ​사용된 도구

* Vue.js 2.6.11
* vue/cli 4.4.6
* npm 6.14.8
* Django 3.1.1
* IDE: Visual Studio Code 1.48



## :desktop_computer: 사용된 기술

![stack](https://user-images.githubusercontent.com/67194249/95681873-29373180-0c1d-11eb-98cd-a7c03cbc235d.JPG)

#### 프론트엔드

| Technology | Description                     | Official website               |
| ---------- | ------------------------------- | ------------------------------ |
| Vue        | Front-end framework             | https://vuejs.org/             |
| Vue-router | Routing library                 | https://router.vuejs.org/      |
| Vuex       | Global State Management library | https://vuex.vuejs.org/        |
| Axios      | HTTP communication library      | https://github.com/axios/axios |

#### 백엔드

| Technology | Dscription                             | Official Website                                |
| ---------- | -------------------------------------- | ----------------------------------------------- |
| django     | Container + MVC framework              | https://www.djangoproject.com/download/         |
| csrf token | Authentication and authorization token | https://docs.djangoproject.com/en/3.1/ref/csrf/ |

#### 요약 알고리즘

[summary_algorithm_README.md](https://github.com/whydizzy27/STORE/blob/main/summary_algorithm_README.md)를 참고하세요.

#### 추천 알고리즘

[doc_Recommend_System.md](https://github.com/whydizzy27/STORE/blob/main/doc_Recommend_System.md)를 참고하세요.


## :straight_ruler: Commit Convention

1. __branch 종류__

  - __develop-_[이니셜]___ : 각 개발자들이 작업하는 개인 공간.

2. __Commit 메세지 Format__  
   ___"[type]commit message, [issue Key] "___  
     _ex) git commit -m "[Add] <기능설명>, [jira Key]"_

  - __Add :__ 새로운 기능 추가.
  - __Fix :__ 버그 수정.
  - __Modify :__ 기능에 버그는 없지만, 코드 수정.
  - __Test :__ 테스트용 코드.
  - __Style :__ 단순 코드 포멧팅.(세미콜론 누락, 들여쓰기 등).
  - __Doc :__ 문서(.md 등) 수정.







## :page_with_curl: 라이센스

```
Copyright (c) 2015 Juns Alen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```



## 참고

* https://gist.github.com/taeukme/e004e01963190615d308a16bcd6e6040

* https://github.com/naver/egjs-flicking

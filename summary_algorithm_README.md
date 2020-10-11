# SSAFY 특화 프로젝트 (논문 추출 프로젝트)

## 01. 파이썬 가상환경 구성 (2020.09.07.월)
```sh
python -m venv testing
cd testing
Scripts\activate.bat
cd ..
```

## 02. PDFFile Parsing 하는 방법 -> 시도 한 방법 전부 소개 (2020.09.07.월 ~ 09.16.수 -> pdf 파일을 Parsing)
```sh
* CERMINE을 이용 PDF논문 추출 테스트 방법
참조 사이트 : https://zelkun.tistory.com/entry/CERMINE-114snapshot%EC%9D%B4%EC%9A%A9-PDF%EB%85%BC%EB%AC%B8-%EC%B6%94%EC%B6%9C-%ED%85%8C%EC%8A%A4%ED%8A%B8

공식 사이트 : http://cermine.ceon.pl

- GNU라이센스라서 이 프로그램은 사용 불가 한 줄 알았으나, SSAFY 내에 물어본 결과 사용 가능한 걸로 확인되었습니다.
- 그러나 논문 메타 추출 프로그램에서 KCI 논문을 추출해보니 시간이 지나치게 많이 드는 문제가 일어났습니다.
- 또한 한글 논문이라서 정확히 다 추출이 되지 않아서 다른 방법을 찾아보기로 했습니다.

* GROBID를 이용 PDF논문 추출 테스트 방법
- 위 방법도 마찬가지로 한글 논문이라서 정확히 다 추출이 되지 않았습니다.


-> 결론 : PDF에서 기존 논문 내용을 구분하는 방법은 한글 논문에는 다른 방법이 없어서, 직접 추출해야 하는걸로 결론 지었습니다.
```

```sh
* PDF -> Text 변환 추출 테스트 방법

- PyPDF2
- 바이트 스트림으로 파일을 열어서, PDF의 파일 정보와 텍스트를 가져올 수 있는 모듈
- 사용하기 어렵지 않았으나, PDF 논문 파일에는 이게 제대로 적용이 되지 않았습니다.
- 띄어쓰기가 날라가고, 정확성이 떨어지며, 일부 논문에서 한글 깨짐 현상이 일어났습니다.

- pdftotext
- PDF문서의 텍스트만 필요하다고 하면 이 패키지를 사용
- 간단한 패키지이며, 따로 설치가 많이 필요하지 않았지만, 정확성이 떨어지는 문제가 일어났습니다.
- 위 문제는 머신러닝 기법이 들어간 pdf -> text 기법을 사용하기로 했습니다.

- pdftotree
- pdftotree는 pdf문서에 텍스트, 그림, 표를 추출해서 문서의 구조를 유지하여 html로 만들어주는 모듈
- 현재 논문에서 있는 table 전부 구분 가능
- 하지만 코덱 문제 때문에 일부 논문을 읽지 못하는 문제 발생 (테스트 파일의 약 10%)
- 또한 시간이 많이 걸리기 때문에, 대안 패키지를 사용하기로 했습니다.

- pdfminer
- 설치 방법 : pip install pdfminer.six (머신러닝 모듈이 들어가, 시간이 매우 많이 걸립니다.)
- pdf문서를 text나 html로 변경해주는 모듈
- table 구분 X, 그림 추출 X, 텍스트 정확도는 높고, 시간이 빠릅니다.
- 그래서 여러 문제점이 높지만, pdfminer를 사용하고, 따로 table 추출 모듈을 통해서 구분하기로 했습니다.

- hanspell
- 설치 방법 : pip install py-hanspell / git clone 이후 python setup.py install
- 사이트 : https://github.com/ssut/py-hanspell
- py-hanspell은 네이버 맞춤법 검사기를 이용한 파이썬용 한글 맞춤법 검사 라이브러리입니다.
- pdfminer + hanspell을 통한 맞춤법 검사를 하기로 했습니다
```

```sh
- tabula
- tabula-java가 원형이며, 입출력 파일을 가공 가능
- 현재 논문 PDF에서 table 인식이 불가능 합니다.

- camelot
- 가장 많이 사용하는 라이브러리
- 현재 논문 PDF에서 table 인식이 불가능 합니다.

- PDFPlumber
- 옵션도 많고, 지원하는 기능도 많습니다.
- 현재 논문 PDF에서 table 인식이 불가능 합니다.

- OCR_SPACE
- OCR 기반의 table 추출 API
- 25,000회 제한, API 표 인식 불가

- PDF_Tables
- 25회 제한, API
- 기존 텍스트까지 표로 인식하는 문제 발생
```

## 03. PDF Install
```sh
pip install pdfminer.six

git clone https://github.com/ssut/py-hanspell
cd hanspell
python setup.py install
cd ..

pip install re
pip install itertools
pip install collections
pip install os
pip install binascii

pip install gensim newspaper3k
pip install lexrankr
pip install wordcloud
pip install matplotlib
pip install konlpy googletrans
pip install beautifulsoup4
pip install pool
pip install googletrans
pip install multithreading
pip install selenium

https://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud
사이트에서 자신에 맞는 wordcloud 파일 다운로드
pip install wordcloud-1.8.0-cp36-cp36m-win_amd64.whl
```

## 04. 사용 설명서 (2020.10.06.수 Version 1.0 Update)
```sh
(1) 폴더, 파일 이름
documents Folder : PDF 파일을 저장하는 장소입니다. 파일 입력시 반영 됩니다.
images Folder : PDF에서 추출된 Image, WordCloud 파일이 저장됩니다.
testing Folder : Testing 했던 폴더입니다.

main.py : python main.py를 실행 시키면 메인 기능이 동작합니다.
mode_crawling.py : selenium을 이용한 KCI 페이지 크롤링이며, PDF를 통해 자세한 정보 검색이 가능합니다.
mode_imageconvert.py : image 관련 처리 함수이며, 현재는 images Path 안에 있는 파일을 Remove 해주는 함수 하나가 담겨 있습니다.
mode_pdfconvert.py : PDF를 처리하는 모든 함수 패키지가 담겨 있습니다.
mode_summarize.py : PDF를 요약해주는 모든 함수 패키지가 담겨 있습니다.

output1.txt : 논문 개요, 논문 전문을 간단히 볼수 있는 텍스트 파일입니다.
output2.txt : 논문 요약 (10줄)을 볼 수 있는 텍스트 파일 입니다.
```

```sh
(2) 변수 이름
link_data : KCI를 통해 검색한 논문의 URL입니다.
title_data_ko : 논문 제목 (한글) 입니다.
title_data_en : 논문 제목 (영어) 입니다.

title_data_plus1 : 논문이 피인용 된 횟수 입니다.
title_data_plus2 : 논문이 열람 된 횟수 입니다.

journalInfo1 : 학술지 이름입니다.
journalInfo2 : 논문 정보입니다.
journalInfo3 : 발행 기관입니다.

name1 : 저자 정보 (이름)가 담겨 있습니다.
name2 : 저자 정보 (학교)가 담겨 있습니다.

content1 : 논문 초록 (요약, 한글)이 담겨 있습니다.
content2 : 논문 초록 (요약, 영어)이 담겨 있습니다.

content3 : 키워드 (한글)가 담겨 있습니다.
content4 : 키워드 (영어)가 담겨 있습니다.

reference : 참고문헌이 담겨 있습니다.

figure_image_name : PDF에서 추출한 그림 데이터 (이름) 입니다.
figure_image_src : PDF에서 추출한 그림 데이터 (경로) 입니다.

final_result : PDF에서 추출한 논문 내용입니다. (초록, 참고문헌 제외한 모든 텍스트)
print_result : PDF에서 추출한 개요, 논문 내용을 output1.txt에 출력할 수 있도록 도와주는 변수입니다.

summarize_data : lexlank 프로그램을 이용한 본문 요약 리스트 입니다. (10줄)
summarize_result : 본문 요약 리스트를 output2.txt에 출력할 수 있도록 도와주는 변수입니다.
translate_result : 영어 논문의 경우 한글 논문으로 반영하는 번역 텍스트 입니다.

summarize_tags : 키워드 추출 변수입니다.
```

```sh
(3) 함수 이름
mode_pdfconvert.py
- save_image, determine_image_type, write_file : PDF에서 추출한 Image 데이터를 바이너리 데이터로 변경 후, 이미지 저장하는 함수 입니다.

- isEnglishOrKorean : 영어인지, 한글인지 반환해주는 함수입니다.

- pdfopen : pdfminer를 이용, pdf를 열어주는 함수입니다.

- pdfread : pdfminer를 이용, pdf에서 텍스트를 추출해줍니다.
- [텍스트 내용, 위치를 저장하고, 이미지를 텍스트 처리 즉시 저장해주는 함수입니다.]
- [text_list : 텍스트 변수 리스트, textfont_list : 텍스트 크기 리스트 (예 : 10px)]
- [textmiddle_list : 텍스트 시작 위치 리스트, textmiddle_average : 전체 텍스트 평균 위치 변수]
- [textfont_average : 전체 텍스트 중간 위치 변수, textfont_cnt : 텍스트 처리 카운트]
- [title_num : 타이틀 위치, title_data : 타이틀 이름]
- [image_name : 이미지 이름, image_list : 이미지 위치 리스트]

- title_return : pdf에서 가장 큰 텍스트를 찾고, 그걸 기반으로 타이틀을 반환하는 함수입니다.

- list_return : 기본적으로 pdfminer는 띄어쓰기를 두번한 상태로 처리되는데, 그 부분을 교정해주는 함수입니다.

- maxsize_return : 가장 많이 쓰인 텍스트 사이즈를 반환해주며, 이를 토대로 본문 텍스트 크기를 찾아주는 함수입니다.

- pdfsort : PDF 파일에 다단이 있는지 확인하고, 이를 처리해줍니다. 추가로 그림이나, 참고문헌 예외처리도 해줍니다.

- pdfgrap : PDF 텍스트 크기별로 묶어주는 함수입니다. 이를 통해서 쓸데없는 텍스트를 걸러줍니다.

- pdfcutter : 참고문헌을 삭제해주고, 서론 앞을 삭제해줍니다. 이후 maxsize_return을 통해 받은 본문 크기를 받고, 중간에 들어간 머리말, 표 글씨 등을 전부 삭제합니다.


mode_summarize.py
- summarize_function : 문장을 요약해주나, 부정확한 부분이 있어서 현재는 사용하지 않는 TextRank 방식의 요약 문장 반환 함수 입니다.

- lexlank_function : 문장을 lexlank (TextRank와 유사하나, 한글 처리 특화)를 통해 10문장으로 요약해주는 함수입니다.

- keywords_function, visualize_function : 키워드를 추출하고, 그걸 기반으로 워드클라우드를 만들어주는 함수입니다.


mode_crawling.py
- crawling_setting : selenium을 이용한 KCI 논문 검색 함수 입니다.


mode_imageconvert.py
- removeAllFile : images 안에 있는 파일을 삭제하고, 초기화 하는 함수입니다.
```
# Python 3.8을 사용한 베이스 이미지 설정
FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    libstan-dev \
    libstan-math-dev \
    libboost-dev \
    libboost-program-options-dev

# requirements.txt 파일을 컨테이너에 복사하고 종속성 설치
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 모든 파일을 컨테이너에 복사
COPY . .

# Streamlit 앱 실행 명령
CMD ["streamlit", "run", "app.py"]

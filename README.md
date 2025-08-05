# ClusteringServer
## 개요
유저가 방문 예정인 매장 리스트를 그룹화 해주는 서비스입니다.
## 배포주소
https://store-group-six.vercel.app/
## 기술스택
### 언어
- Python
### 프레임워크/라이브러리
- Fast API
### 인프라
- AWS EC2
- Nginx
### 배포
- Docker
- Github Actions
### 협업툴
- Disord
- Notion
## 아키텍처
<img width="1406" height="864" alt="image" src="https://github.com/user-attachments/assets/7926c2be-850f-4b74-8bb6-84a588777be9" /> </br>
## 주요기여
사용자로부터 가게정보와 나누고 싶은 그룹수를 입력으로 받고 kmeans 알고리즘을 사용하여 클러스터링을 수행합니다.
또한 k-means-constrained(BSD 3-Clause License) 오픈소스를 활용하여 군집의 크기를 제어하는 균등 클러스터링 기능을 제공하는 api를 설계하고 구현하였습니다.

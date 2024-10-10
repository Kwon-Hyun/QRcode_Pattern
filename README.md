# QRcode_Pattern
**QR code read algorithm - Position pattern, Alignment pattern 기반**
- Alignment Marker들을 발견하면 RGB에 맞춰서 b-box를 침.

1. 두 점 사이의 거리를 게산
2. 선 L-M 기준, 점 J에서 수직으로 떨어진 거리 계산
3. 두 점으로 이루어진 선의 기울기 계산
4. QR code 내 3개의 Position Pattern & Alignment Pattern을 이용하여 detection 방향 결정
5. QR code에서 3개의 Position Pattern detection 및 방향 계산
    <br>
    5-1. img GrayScale(img_gray) 진행 <br>
    5-2. img_gray에 대해 Canny Edge(img_canny) 진행 <br>
    5-3. Position Pattern 탐지하면 그 각 pattern에 윤곽선 그리기 <br>
6. QR code Position Pattern 탐지
7. <br>
    6-1. 패턴 3개 다 탐지 GOOD => QR code 방향, 위치(top, bottom, right) 터미널에 출력 및 b-box 그리기

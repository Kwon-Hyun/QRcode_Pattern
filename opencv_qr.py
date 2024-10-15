import cv2 as cv
import numpy as np
import math

# algirithm
# 1. position pattern 2개 기준으로 위쪽(0), 오(1), 아래(2), 왼(3)으로 설정해서 어느 쪽을 나타내는지 파악
# 2. position pattern 2개따리의 각 중심점을 기준으로 두 점 사이 거리 게산
# 3. 두 점 사이의 거리를 계산하며 그린 선분LM을 가지고, 세로 길이 측정.

# QR 코드 방향 정의 (북쪽부터 시계 방향으로 돌아가유~)
CV_QR_NORTH = 0 # 북
CV_QR_EAST = 1  # 동
CV_QR_SOUTH = 2 # 남
CV_QR_WEST = 3  # 서

# 두 점 사이의 거리를 계산하는 함수
def cv_distance(P, Q):
    return np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2) # sqrt : 제곱근 구하는 함수

# 선분 LM을 기준으로 점 J에서 수직으로 떨어진 거리를 계산하는 함수
def cv_lineEquation(L, M, J):
    a = -(M[1] - L[1]) / (M[0] - L[0])
    b = 1.0
    c = ((M[1] - L[1]) / (M[0] - L[0])) * L[0] - L[1]
    pdist = (a * J[0] + (b * J[1]) + c) / np.sqrt(a * a + b * b)
    return pdist


# 두 점으로 이루어진 선의 기울기를 계산하는 함수
def cv_lineSlope(L, M):
    dx = M[0] - L[0]
    dy = M[1] - L[1]
    if dy != 0:
        alignment = 1
        return dy / dx, alignment
    else:
        alignment = 0
        return 0.0, alignment

# QR 코드의 세 개의 위치 패턴을 이용하여 방향을 결정하는 함수
def find_qr_orientation(contours, mc):
    AB = cv_distance(mc[0], mc[1])
    BC = cv_distance(mc[1], mc[2])
    CA = cv_distance(mc[2], mc[0])

    if AB > BC and AB > CA:
        outlier = 2
        median1 = 0
        median2 = 1
    elif CA > AB and CA > BC:
        outlier = 1
        median1 = 0
        median2 = 2
    else:
        outlier = 0
        median1 = 1
        median2 = 2

    dist = cv_lineEquation(mc[median1], mc[median2], mc[outlier])
    slope, align = cv_lineSlope(mc[median1], mc[median2])

    if align == 0:
        bottom = median1
        right = median2
        orientation = CV_QR_NORTH

    elif slope < 0 and dist < 0:
        bottom = median1
        right = median2
        orientation = CV_QR_NORTH

    elif slope > 0 and dist < 0:
        right = median1
        bottom = median2
        orientation = CV_QR_EAST

    elif slope < 0 and dist > 0:
        right = median1
        bottom = median2
        orientation = CV_QR_SOUTH

    elif slope > 0 and dist > 0:
        bottom = median1
        right = median2
        orientation = CV_QR_WEST

    return outlier, bottom, right, orientation




#def qr_calibration():






# QR 코드에서 세 개의 위치 패턴 감지 및 방향 계산
def detect_qr(image):
    # 간단한 이미지 전처리
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)    # 이미지 grayscale
    img_canny = cv.Canny(img_gray, 100, 200)    # qr detection을 위한 Canny Edge detection

    # 윤곽선 찾기
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # contour에서 3개의 alignment pattern 탐지

    mark = 0
    A, B, C = None, None, None

    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[0][k][2] != -1:
            k = hierarchy[0][k][2]
            c += 1
        if hierarchy[0][k][2] != -1:
            c += 1

        if c >= 5:
            if mark == 0:
                A = i
            elif mark == 1:
                B = i
            elif mark == 2:
                C = i
            mark += 1

    if mark >= 3:  # QR 코드 위치 탐지 패턴 3개를 찾았을 때
        mu = [cv.moments(contours[A]), cv.moments(contours[B]), cv.moments(contours[C])]
        mc = [(
            mu[i]["m10"] / mu[i]["m00"],
            mu[i]["m01"] / mu[i]["m00"]
        ) for i in range(3)]

        outlier, bottom, right, orientation = find_qr_orientation(contours, mc)

        print(f"QR 코드 방향: {orientation}")
        print(f"Top: {outlier}, Bottom: {bottom}, Right: {right}")

        # 외곽선 그리기
        cv.drawContours(image, contours, A, (0, 255, 0), 2)
        cv.drawContours(image, contours, B, (255, 0, 0), 2)
        cv.drawContours(image, contours, C, (0, 0, 255), 2)

        return image
    return None

# 실시간 카메라 스트리밍 및 QR 코드 감지
def realtime_qr_detection():
    cap = cv.VideoCapture(0)  # 0번 : 카메라(웹캠) 사용

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라에서 영상을 읽을 수 없습니다.")
            break

        # QR 코드 감지 및 방향 표시
        processed_frame = detect_qr(frame)

        if processed_frame is not None:
            cv.imshow('QR Code Detection', processed_frame)
        else:
            cv.imshow('QR Code Detection', frame)

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    realtime_qr_detection()
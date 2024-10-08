import cv2
import numpy as np
import math

# QR 코드 방향 정의
CV_QR_NORTH = 0
CV_QR_EAST = 1
CV_QR_SOUTH = 2
CV_QR_WEST = 3

# 두 점 사이의 거리를 계산하는 함수
def cv_distance(P, Q):
    return np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)

# 선 L-M을 기준으로 점 J에서 수직으로 떨어진 거리를 계산하는 함수
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

# QR 코드에서 세 개의 위치 패턴 감지 및 방향 계산
def detect_qr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 윤곽선 찾기
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        mu = [cv2.moments(contours[A]), cv2.moments(contours[B]), cv2.moments(contours[C])]
        mc = [(
            mu[i]["m10"] / mu[i]["m00"],
            mu[i]["m01"] / mu[i]["m00"]
        ) for i in range(3)]

        outlier, bottom, right, orientation = find_qr_orientation(contours, mc)

        print(f"QR 코드 방향: {orientation}")
        print(f"Top: {outlier}, Bottom: {bottom}, Right: {right}")

        # 외곽선 그리기
        cv2.drawContours(image, contours, A, (0, 255, 0), 2)
        cv2.drawContours(image, contours, B, (255, 0, 0), 2)
        cv2.drawContours(image, contours, C, (0, 0, 255), 2)

        return image
    return None

# 실시간 카메라 스트리밍 및 QR 코드 감지
def realtime_qr_detection():
    cap = cv2.VideoCapture(0)  # 0번 카메라(웹캠) 사용

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
            cv2.imshow('QR Code Detection', processed_frame)
        else:
            cv2.imshow('QR Code Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_qr_detection()
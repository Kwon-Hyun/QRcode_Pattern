import cv2 as cv
import numpy as np
import math
import os
import time
import timeit

# algorithm
# 1. position pattern 2개 기준으로 위쪽(0), 오(1), 아래(2), 왼(3)으로 설정해서 어느 쪽을 나타내는지 파악
# 2. position pattern 2개따리의 각 중심점을 기준으로 두 점 사이 거리 게산
# 3. 두 점 사이의 거리를 계산하며 그린 선분LM을 가지고, 세로 길이 측정.

# QR 코드 방향 정의 (북쪽부터 시계 방향으로 돌아가유~)
CV_QR_UP = "위쪽" # 북 0
CV_QR_RIGHT = "오른쪽"  # 동 1
CV_QR_DOWN = "아래쪽" # 남 2
CV_QR_LEFT = "왼쪽"  # 서 3

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
        orientation = CV_QR_UP

    elif slope < 0 and dist < 0:
        bottom = median1
        right = median2
        orientation = CV_QR_UP

    elif slope > 0 and dist < 0:
        right = median1
        bottom = median2
        orientation = CV_QR_RIGHT

    elif slope < 0 and dist > 0:
        right = median1
        bottom = median2
        orientation = CV_QR_DOWN

    elif slope > 0 and dist > 0:
        bottom = median1
        right = median2
        orientation = CV_QR_LEFT

    return outlier, bottom, right, orientation, slope


# 기울기(회전 각도) 계산하는 함수 (position pattern의 기울기 활용)
def calculate_rotation_angle(slope):
    # arctan 사용해서 기울기에서 각도 계산~
    angle = np.degrees(np.arctan(slope))

    # 기존 (-) 값으로 보이던 것을 수정하기 위해 angle 값에 조건을 달아서 Return
    return angle if angle >=0 else 360 + angle


# QR code calibration을 위한 함수 (feat.원근변환ㅎㅎ)
def qr_calibration(image, mc, outlier, right, bottom, orientation):
    # QR code position pattern 좌표 (3개)
    src_pts = np.array([mc[0], mc[1], mc[2]], dtype="float32")

    top1 = mc[outlier]
    top2 = mc[right] if orientation in [CV_QR_UP, CV_QR_DOWN] else mc[bottom]
    width = int(cv_distance(top1, top2))
    height = width
    
    # calibration 후 사용할 기준 좌표 (이론상 정확히 QR code가 정사각형일 때)
    dst_pts = np.array([
        [0, 0],
        [width-1, 0],
        [0, height-1]
        #[image.shape[1], 0],
        #[image.shape[1], image.shape[0]]
    ], dtype="float32")

    '''
    # 원근 반환 행렬 계산
    M = cv.getAffineTransform(src_pts, dst_pts)

    # 변환 적용해서 image calibration
    calibrated_image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    '''

    # 원근 변환 행렬 계산
    M = cv.getAffineTransform(src_pts, dst_pts)

    # 원본 이미지에 QR 코드가 차지하는 부분에만 캘리브레이션 적용
    calibrated_image = image.copy()  # 원본 이미지를 복사

    warped_qr = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))  # 원본 이미지 크기에 맞게 변환
    mask = np.zeros_like(image, dtype=np.uint8)  # 마스크 생성

    # QR 코드 영역을 마스크로 설정
    mask = cv.fillConvexPoly(mask, np.int32([dst_pts]), (255, 255, 255))
    calibrated_image = cv.bitwise_and(calibrated_image, 255 - mask)  # QR 코드 영역 지우기

    calibrated_image = cv.add(calibrated_image, cv.bitwise_and(warped_qr, mask))  # 캘리브레이션된 QR 코드 영역 추가

    # 2개의 Position pattern의 기울기 이용 - calibration 회전각 게산
    calibrated_slope, _ = cv_lineSlope(mc[0], mc[1])
    calibrated_rotation_angle = calculate_rotation_angle(calibrated_slope)

    return calibrated_image, calibrated_rotation_angle


# calibration된 이미지를 저장하는 함수
def save_calibrated_image(image, folder="calibrated_qr_images"):
    # folder 없으면 생성
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 파일명에 timestamp 추가하여 중복 방지
    filename = f"calibrated_{int(time.time())}.jpg"
    filepath = os.path.join(folder, filename)

    # 이미지 저장
    cv.imwrite(filepath, image)
    print(" ")
    #print(f"저장된 캘리브레이션 이미지~ : {filepath}")


# QR code에서 3개의 Position Pattern Detection 및 방향 계산
def detect_qr(image):
    # 간단한 이미지 전처리
    # 1. GrayScale 변환
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)    # 이미지 grayscale
    
    # 2. 이진화 (QR code 윤곽선을 명확하게 검출하기 위함)
    _, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 3. Morphology 연산 (그 중 열림 연산 활용 - 작은 노이즈 제거)
        # 이 때 kernel 은 5x5 사용
        # 열림 연산 => 큰 객체들은 사이즈 변화 별로 없음. (노이즈 제거에 효과적)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_morph = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)

    # 4. Canny Edge Detection (윤곽선 검출 !!)
    img_canny = cv.Canny(img_gray, 100, 200)    # qr detection을 위함

    # 5. 윤곽선 찾기
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # contour에서 3개의 alignment pattern 탐지


    # 전처리를 완료했다면 이제 QR code Position Pattern을 탐지해보아요..
    mark = 0
    A, B, C = None, None, None

    for i in range(len(contours)):
        k = i
        c = 0

        # hierarchy array에서 마지막 index == -1이면, External contour라는 의미!! 이므로 이 때, 외곽선을 그리면 됨. (drawContours 함수 활용)
        # 만약 Internal contour를 그리고 싶으면, hierarchy array 마지막 index != -1인 걸 찾으면 됨~ 
        # 참고 : https://blog.naver.com/PostView.naver?blogId=zeus05100&logNo=221737728483&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView

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

    if mark >= 3:  # QR code Position Pattern 3개를 찾았을 때
        mu = [cv.moments(contours[A]), cv.moments(contours[B]), cv.moments(contours[C])]
        mc = [(
            mu[i]["m10"] / mu[i]["m00"],
            mu[i]["m01"] / mu[i]["m00"]
        ) for i in range(3)]

        outlier, bottom, right, orientation, slope = find_qr_orientation(contours, mc)
        initial_rotation_angle = calculate_rotation_angle(slope)

        print(f"QR 코드 방향: {orientation}")
        print(f"Top: {outlier}, Bottom: {bottom}, Right: {right}")

        print(f"초기 회전각 : {initial_rotation_angle:.2f} 도")

        # 회전 각도 계산 (QR code가 정사각형이라는 전제 하)
            # 3개의 position pattern 중, median1, median2 두 점 사이의 기울기 기준
            # 즉, QR code 2개의 position pattern 좌표를 이은 선분 기준으로 기울기 계산
        # arctan 활용 - "도(degree)"로 변환하여 나타냄.

        # 회전각(기울기) : +, - 각도로 표현되는데,,
            # + 각도 : QR code가 오른쪽(시계방향으로 회전)으로 기울어진 경우.
            # - 각도 : QR coee가 왼쪽(반시계방향으로 회전)으로 기울어진 경우.
        rotation_angle = calculate_rotation_angle(slope)
        print(f"QR code 기울기(회전각) : {rotation_angle} 도")

        # position pattern에 외곽선 그리기
        cv.drawContours(image, contours, A, (0, 255, 0), 2)
        cv.drawContours(image, contours, B, (255, 0, 0), 2)
        cv.drawContours(image, contours, C, (0, 0, 255), 2)

        ''' drawContours를 한 번에 해버리는 다른 방법으로는 
        
        for idx, color in zip([A, B, C], [(0, 255, 0), (255, 0, 0), (0, 0, 255)]):
            cv.drawContours(image, contours, idx, color, 2)

            이렇게도 가능. 이거는 좀 더 숙달되면 깔끔하게 써보자.'''

        # QR 그 자체의 외곽 사각형을 그려서 외곽선 그리기

        # QR code calibration
        calibrated_image, calibration_rotation_angle = qr_calibration(image, mc, outlier, right, bottom, orientation)

        # calibration된 이미지 저장
        save_calibrated_image(calibrated_image)

        # 화면에 캘리브레이션 회전각을 출력
        cv.putText(image, f"Calibration 회전각: {calibration_rotation_angle:.2f} 도", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

        try:
            # FPS 성능평가 해볼까요?
            # 참고 : https://deep-eye.tistory.com/15
            #! FPS algorithm start
            fps_start = timeit.default_timer()

            if ret is True:
                # QR code 감지 및 방향 표시
                processed_frame = detect_qr(frame)
                

                if processed_frame is not None:
                    # QR code가 인식되었을 때만 FPS 출력
                    #! FPS algorithm finish
                    fps_fin = timeit.default_timer()

                    FPS = int(1./(fps_fin - fps_start))

                    #! cv.CAP_PROP_FPS를 활용한 FPS algorithm
                    cv_FPS = cap.get(cv.CAP_PROP_FPS)   # 웹캠에서 FPS값 획득하여, 앞선 방식의 FPS 결과와 비교해보기

                    if cv_FPS == 0.0:
                        cv_FPS = 30.0

                    time_per_frame_video = 1/cv_FPS
                    last_time = time.perf_counter()

                    # cv fps 계산
                    time_per_frame = time.perf_counter() - last_time
                    time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
                    time.sleep(time_sleep_frame)

                    real_fps = 1/(time_per_frame)
                    last_time = time.perf_counter()


                    cv.imshow('QR Code Detection', processed_frame)

                    print(f"FPS : {FPS} fps")
                    print(f"cv FPS : {real_fps} fps")
                    print("\n")

                else:
                    cv.imshow('QR Code Detection', frame)
        
        except Exception as e:
            print(f"오류 발생 : {e}")

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    realtime_qr_detection()

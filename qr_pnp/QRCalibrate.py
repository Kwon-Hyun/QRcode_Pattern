import cv2
import numpy as np
import glob
import QRPoints

# 이미지 읽기
images = glob.glob('markerImages/*.JPG')

# 종료 조건 설정 (코너 추출의 종료 조건)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 점 배열 초기화
objp = np.zeros((6, 3), dtype=np.float32)

# 패턴 이미지에 있는 3D 좌표 목록. 실제 3D 공간에서의 좌표를 cm 단위로 지정
pointList = [[0, 0, 0], [2.464, 2.464, 0], [0, 8.8, 0], [2.464, 6.336, 0], [8.8, 0, 0], [6.336, 2.464, 0]]

# 3D 좌표 배열에 pointList 값을 채웁니다.
for i, x in zip(pointList, range(0, len(pointList))):
    objp[x] = i

# 이미지 좌표와 3D 좌표를 저장할 리스트
imgPointsList = []
objPointsList = []

# QR 코드 코너를 추출하고, 2D 및 3D 좌표를 리스트에 저장
for fname in images:
    print(fname)
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # QR 코드 코너 추출
    points, success = QRPoints.getPoints(fname)

    if not success:
        # 코너가 제대로 감지되지 않았을 경우
        points, success = QRPoints.getPoints(fname, 7, 3)
        cv2.cornerSubPix(gray, points, (11, 11), (-1, -1), criteria)
        imgPointsList.append(points)
        objPointsList.append(objp)
    else:
        # 코너 보정 및 리스트 추가
        cv2.cornerSubPix(gray, points, (11, 11), (-1, -1), criteria)
        imgPointsList.append(points)
        objPointsList.append(objp)

# 카메라 캘리브레이션
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPointsList, imgPointsList, gray.shape[::-1], None, None)

# 캘리브레이션 결과를 파일에 저장
np.savez('iPhoneCam2.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# 시각적 확인을 위해 이미지에서 왜곡 보정
for fname in images:
    image = cv2.imread(fname)
    h, w = image.shape[:2]

    # 왜곡 보정된 이미지 생성
    undistorted_img = cv2.undistort(image, mtx, dist, None, mtx)

    # 원본 이미지와 보정된 이미지 비교를 위해 나란히 표시
    combined = np.hstack((image, undistorted_img))

    # 이미지 출력 (원본 vs 보정된 이미지)
    cv2.imshow('Original vs Undistorted', combined)
    
    # 'q' 키를 누르면 다음 이미지로 넘어감
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# 프로젝션 오류 계산
mean_error = 0
for i in range(len(objPointsList)):
    imgpoints2, _ = cv2.projectPoints(objPointsList[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgPointsList[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objPointsList))
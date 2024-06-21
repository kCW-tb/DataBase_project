from pymongo import MongoClient
import time
import mediapipe as mp
import numpy as np
import math
import cv2

client = MongoClient('localhost', 27017)
db = client.stock

def update_stock_values():
    stocks = db.stock_collection.find()

    for stock in stocks:
        if stock["Current_Stock"] <= 3:
            new_total_stock = stock["Total_Stock"] + 2
        elif stock["Total_Stock"] / 2 < stock["Current_Stock"]:
            if stock["Total_Stock"] >= 6:
                new_total_stock = stock["Total_Stock"] - 2
        else:
            new_total_stock = stock["Total_Stock"]

        # MongoDB에서 해당 문서의 Total_Stock과 Current_Stock 업데이트
        db.stock_collection.update_one(
            {"_id": stock["_id"]},
            {"$set": {
                "Total_Stock": new_total_stock,
                "Current_Stock": new_total_stock
            }}
        )
def check_distance(hand_landmarks):
    # 8번 노드(x, y) 좌표
    x8, y8 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
    # 12번 노드(x, y) 좌표
    x12, y12 = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y

    # 두 노드 사이의 거리 계산
    distance = math.sqrt((x12 - x8)**2 + (y12 - y8)**2)
    distance = distance * 100
    return distance
def decrease_current_stock(n):
    stocks = db.stock_collection.find()  
    data_to_update = None
    for idx, stock in enumerate(stocks):
        if idx == n-1:  # 인덱스를 고려하여 n-1로 체크
            current_stock = stock["Current_Stock"]
            new_current_stock = current_stock - 1
            data_to_update = stock
            data_to_update["Current_Stock"] = new_current_stock
            break

    if data_to_update:
        db.stock_collection.update_one({"_id": data_to_update["_id"]}, {"$set": {"Current_Stock": data_to_update["Current_Stock"]}})
        print(f"{n}번 재고 하나 소진.")
    else:
        print(f"제품 선택이 되지 않았습니다.")
def show_stock_data():
    stocks = db.stock_collection.find()
    for stock in stocks:
        print(f"총 재고 : {stock['Total_Stock']}, 남은 재고 : {stock['Current_Stock']}, 고정 : {stock['Increase']}")

db.stock_collection.delete_many({})
product_number = -1

if db.stock_collection.count_documents({}) == 0:
    stock_data = [
        {"Total_Stock": 10, "Current_Stock": 10, "Increase": 0},
        {"Total_Stock": 10, "Current_Stock": 10, "Increase": 0},
        {"Total_Stock": 10, "Current_Stock": 10, "Increase": 0},
        {"Total_Stock": 10, "Current_Stock": 10, "Increase": 0},
        {"Total_Stock": 10, "Current_Stock": 10, "Increase": 0},
        {"Total_Stock": 10, "Current_Stock": 10, "Increase": 0}
    ]
    result = db.stock_collection.insert_many(stock_data)
    print(result)
else:
    print("데이터가 이미 존재합니다. 삽입을 수행하지 않습니다.")

#이미지 데이터 호출
produrt_img1 = cv2.imread("./tri_kimbap/001.jpg")
produrt_img2 = cv2.imread("./tri_kimbap/002.jpg")
produrt_img3 = cv2.imread("./tri_kimbap/003.jpg")
produrt_img4 = cv2.imread("./tri_kimbap/004.jpg")
produrt_img5 = cv2.imread("./tri_kimbap/005.jpg")
produrt_img6 = cv2.imread("./tri_kimbap/006.jpg")
#이미지 데이터 전처리
produrt_img1 = cv2.resize(produrt_img1, (200,200))
produrt_img2 = cv2.resize(produrt_img2, (200,200))
produrt_img3 = cv2.resize(produrt_img3, (200,200))
produrt_img4 = cv2.resize(produrt_img4, (200,200))
produrt_img5 = cv2.resize(produrt_img5, (200,200))
produrt_img6 = cv2.resize(produrt_img6, (200,200))
#배경 생성
background = np.ones((720,1280, 3), dtype=np.uint8) * 255
# 사각형 그리기
cv2.rectangle(background, (25, 25), (225, 125), (200, 200, 200), -1)
cv2.rectangle(background, (25, 300), (225, 400), (200, 200, 200), -1)
cv2.rectangle(background, (1070, 620), (1270, 715), (200, 200, 200), -1)

# 텍스트 추가
cv2.putText(background, "BUY", (65, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
cv2.putText(background, "CANCEL", (35, 365), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
cv2.putText(background, "EXIT", (1110, 695), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

background[80:280, 300:500] = produrt_img1.copy()
background[80:280, 550:750] = produrt_img2.copy()
background[80:280, 800:1000] = produrt_img3.copy()
background[320:520, 300:500] = produrt_img4.copy()
background[320:520, 550:750] = produrt_img5.copy()
background[320:520, 800:1000] = produrt_img6.copy()

# load hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils # drawing_utils를 이용하여 랜드마크 드로잉

#카메라에 대한 호출
cap = cv2.VideoCapture(0)

last_update_time = time.time()

while cap.isOpened():
    current_time = time.time()
    if current_time - last_update_time >= 70:  # 70초마다 업데이트
        print("재고 주문 전 현 상황")
        show_stock_data()
        update_stock_values() #재고 업데이트
        print("재고 주문 후 현 상황")
        show_stock_data()
        last_update_time = current_time  #시간을 갱신합니다.
        
    success, image = cap.read()
    if not success:
        continue
    image = cv2.resize(image, (1280, 720))

    # 스켈레톤을 추출
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image = background[:,:].copy()

    # 손이 검출된 경우
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 노드끼리의 점을 연결하여 표시
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            break
        distance = check_distance(hand_landmarks)
        cv2.putText(image, "Distance : " + str(distance), (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if distance < 10:    #거리가 10 미만일 떄 (검지와 중지가 겹쳐있을 때)
            x = int(hand_landmarks.landmark[8].x * 1280)
            y = int(hand_landmarks.landmark[8].y * 720)
            print("x : ", str(x) ,"y : "+ str(y))
            if 300 < x < 500 and 80 < y < 280:
                product_number = 1
                print("물품 1 선택")
            elif 550 < x < 750 and 80 < y < 280:
                product_number = 2
                print("물품 2 선택")
            elif 800 < x < 1000 and 80 < y < 280:
                product_number = 3
                print("물품 3 선택")
            elif 300 < x < 500 and 320 < y < 520:
                product_number = 4
                print("물품 4 선택")
            elif 550 < x < 750 and 320 < y < 520:
                product_number = 5
                print("물품 5 선택")
            elif 800 < x < 1000 and 320 < y < 520:
                product_number = 6
                print("물품 6 선택")
            elif 25 < x < 225 and 25 < y < 125:
                decrease_current_stock(product_number)
                product_number = -1
            elif 25 < x < 225 and 300 < y < 400:
                print("선택 초기화")
                product_number = -1
            elif 1070 < x < 1270 and 620 < y < 715:
                exit()
            else:
                pass
            print(product_number)

    
    # 화면에 출력
    cv2.imshow('Hand Tracking', image)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

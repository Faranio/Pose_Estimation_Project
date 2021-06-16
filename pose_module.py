import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, mode=False, upper_body_only=False, smooth_landmarks=True, detection_confidence=0.5,
                 tracking_confidence=0.5):
        self.mode = mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upper_body_only, self.smooth_landmarks, self.detection_confidence,
                                     self.tracking_confidence)

        self.results = None

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def get_position(self, img, draw=True):
        h, w, c = img.shape

        if draw and self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()

        img = detector.find_pose(img)

        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

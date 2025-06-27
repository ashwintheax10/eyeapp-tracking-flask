
import cv2, time, mediapipe as mp, numpy as np, sys, winsound
from collections import deque

BLINK_EAR_THR, BLINK_CONSEC_FR = 0.21, 2
GAZE_L_THR,  GAZE_R_THR        = 1.25, 0.75        # ratio → RIGHT / LEFT
AWAY_GRACE_SEC  = 1.5                              # warn after this
EXIT_DELAY_SEC  = 2.0                              # then exit
FPS_BUF = 30

LEFT=[33,160,158,133,153,144];  RIGHT=[362,385,387,263,373,380]
LEFT_IRIS=[468,469,470,471];     RIGHT_IRIS=[473,474,475,476]
# ───────────────────────────────────────────────────────────

def ear(p):
    p1,p2,p3,p4,p5,p6=p; return (np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5))/(2*np.linalg.norm(p1-p4))
def gaze_ratio(iris, left_corner, right_corner):
    dl=np.linalg.norm(iris-left_corner); dr=np.linalg.norm(iris-right_corner)
    return dr/dl if dl else 1
def direction(r):  # mirrored webcam
    return "RIGHT" if r>GAZE_L_THR else "LEFT" if r<GAZE_R_THR else "CENTER"

mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True,
                                          max_num_faces=1,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
cam = cv2.VideoCapture(0)
fps_hist, blink_cntr, blinks = deque(maxlen=FPS_BUF), 0, 0
last_center = time.time(); warned=False

while cam.isOpened():
    t0=time.time()
    ok, frame = cam.read();  h,w = frame.shape[:2]
    if not ok: break
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res=mp_face.process(rgb)

    gazeL=gazeR="--"; ear_disp=0
    if res.multi_face_landmarks:
        lm=res.multi_face_landmarks[0].landmark
        eyeL=np.array([[lm[i].x*w, lm[i].y*h] for i in LEFT])
        eyeR=np.array([[lm[i].x*w, lm[i].y*h] for i in RIGHT])
        irisL=np.array([[lm[i].x*w, lm[i].y*h] for i in LEFT_IRIS]).mean(0)
        irisR=np.array([[lm[i].x*w, lm[i].y*h] for i in RIGHT_IRIS]).mean(0)

        # 1) blink
        ear_disp=round((ear(eyeL)+ear(eyeR))/2,3)
        if ear_disp<BLINK_EAR_THR: blink_cntr+=1
        else:
            if blink_cntr>=BLINK_CONSEC_FR: blinks+=1
            blink_cntr=0

        # 2) gaze dir
        gazeL, gazeR = direction(gaze_ratio(irisL,eyeL[0],eyeL[3])), \
                       direction(gaze_ratio(irisR,eyeR[0],eyeR[3]))

        # 3) draw contours
        for poly in (eyeL,eyeR): cv2.polylines(frame,[poly.astype(int)],True,(0,255,0),1)
        cv2.circle(frame,tuple(irisL.astype(int)),2,(0,255,0),-1)
        cv2.circle(frame,tuple(irisR.astype(int)),2,(0,255,0),-1)

    # ─── focus guard ───────────────────────────────────────
    if gazeL==gazeR=="CENTER":
        last_center=time.time(); warned=False
    else:
        away=time.time()-last_center
        if away>AWAY_GRACE_SEC:
            cv2.putText(frame,"LOOK BACK OR EXAM WILL CLOSE!",
                        (40,60),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),3)
            if not warned:
                warned=True
                warn_t=time.time()
                winsound.Beep(1000, 500)  # frequency, duration

            if time.time()-warn_t > EXIT_DELAY_SEC:
                print("Focus lost too long – closing exam window now.")
                cam.release(); cv2.destroyAllWindows(); sys.exit(0)

    # ─── HUD text ─────────────────────────────────────────
    fps_hist.append(time.time()-t0); fps=1/np.mean(fps_hist)
    cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame,f"Ratio:{ear_disp:.2f}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(frame,f"Total Blinks:{blinks}",(10,110),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    cv2.putText(frame,f"R:{gazeR}",(10,180),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    cv2.putText(frame,f"L:{gazeL}",(10,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

    cv2.imshow("Eye-Tracker w/ Guard",frame)
    if cv2.waitKey(1)&0xFF==27: break   # ESC
cam.release(); cv2.destroyAllWindows()

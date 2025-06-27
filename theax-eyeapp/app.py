# app.py  ───────────────────────────────────────────────────
import cv2, time, mediapipe as mp, numpy as np, winsound
from collections import deque
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

# ─── tracker parameters ───────────────────────────────────
BLINK_EAR_THR, BLINK_CONSEC     = 0.21, 2
GAZE_L_THR,  GAZE_R_THR         = 1.25, 0.75
AWAY_GRACE_SEC, EXIT_DELAY_SEC  = 1.5, 2.0
LEFT=[33,160,158,133,153,144]; RIGHT=[362,385,387,263,373,380]
L_IRIS=[468,469,470,471];       R_IRIS=[473,474,475,476]
# ----------------------------------------------------------

def ear(p):
    p1,p2,p3,p4,p5,p6=p
    return (np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5))/(2*np.linalg.norm(p1-p4))

def gaze_ratio(iris,l,r):
    dl=np.linalg.norm(iris-l); dr=np.linalg.norm(iris-r)
    return dr/dl if dl else 1

def direction(r):
    return "RIGHT" if r>GAZE_L_THR else "LEFT" if r<GAZE_R_THR else "CENTER"

# ─── Flask setup ───────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    mp_face = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cam = cv2.VideoCapture(0)
    fps_hist, blink_cntr, blinks = deque(maxlen=30), 0, 0
    last_center, warned = time.time(), False

    while cam.isOpened():
        t0 = time.time()
        ok, frame = cam.read();  h,w = frame.shape[:2]
        if not ok: break

        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gazeL=gazeR="--"; ear_disp=0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            eyeL = np.array([[lm[i].x*w,lm[i].y*h] for i in LEFT])
            eyeR = np.array([[lm[i].x*w,lm[i].y*h] for i in RIGHT])
            irisL = np.array([[lm[i].x*w,lm[i].y*h] for i in L_IRIS])
            irisR = np.array([[lm[i].x*w,lm[i].y*h] for i in R_IRIS])

            # --- EAR / blink
            ear_disp = round((ear(eyeL)+ear(eyeR))/2, 3)
            if ear_disp < BLINK_EAR_THR: blink_cntr += 1
            else:
                if blink_cntr >= BLINK_CONSEC: blinks += 1
                blink_cntr = 0

            # --- gaze direction
            rL = gaze_ratio(irisL.mean(0), eyeL[0], eyeL[3])
            rR = gaze_ratio(irisR.mean(0), eyeR[0], eyeR[3])
            gazeL, gazeR = direction(rL), direction(rR)

            # --- draw contours
            for poly in (eyeL, eyeR):
                cv2.polylines(frame,[poly.astype(int)],True,(0,255,0),1)
            cv2.polylines(frame,[irisL.astype(int)],True,(0,255,0),1)
            cv2.polylines(frame,[irisR.astype(int)],True,(0,255,0),1)
            cv2.circle(frame,tuple(irisL.mean(0).astype(int)),2,(0,255,0),-1)
            cv2.circle(frame,tuple(irisR.mean(0).astype(int)),2,(0,255,0),-1)

        # ─── focus guard  (red banner + beep + socket events) ─
        if gazeL==gazeR=="CENTER":
            last_center=time.time(); warned=False
        else:
            away=time.time()-last_center
            if away > AWAY_GRACE_SEC:
                cv2.putText(frame, "LOOK BACK OR EXAM WILL CLOSE!",
                            (40,60), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 3)
                if not warned:
                    warned=True; warn_t=time.time()
                    winsound.Beep(1000, 500)          # local speaker beep
                    socketio.emit("warn")             # client banner + beep

            if warned and time.time()-warn_t > EXIT_DELAY_SEC:
                socketio.emit("focus_lost")
                break

        # --- HUD overlay
        fps_hist.append(time.time()-t0); fps = 1/np.mean(fps_hist)
        cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame,f"EAR:{ear_disp:.2f}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.putText(frame,f"Blinks:{blinks}",(10,110),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        cv2.putText(frame,f"R:{gazeR}",(10,180),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
        cv2.putText(frame,f"L:{gazeL}",(10,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

        # --- MJPEG encode
        ok, jpg = cv2.imencode('.jpg', frame)
        if not ok: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + jpg.tobytes() + b'\r\n')
    cam.release()

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

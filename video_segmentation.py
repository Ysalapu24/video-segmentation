import cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, csv, os, sys

HARD_CUT_THRESHOLD=0.85; GRADUAL_THRESHOLD=0.95; WINDOW_SIZE=10
SMOOTH_KERNEL=5; MIN_DISTANCE=15; RESIZE_DIM=(320,240)
OUTPUT_DIR="segments"; LOG_FILE="transitions.csv"; CAMERA_DURATION=60

def capture_from_camera():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise IOError("Cannot access camera.")
    fps=cap.get(cv2.CAP_PROP_FPS) or 30; frames=[]
    print(f"[CAMERA] Recording for {CAMERA_DURATION}s — press Q to stop early.")
    total=int(fps*CAMERA_DURATION)
    while len(frames)<total:
        ret,frame=cap.read()
        if not ret: break
        frames.append(frame)
        cv2.imshow("Recording... Press Q to stop",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"[CAMERA] Captured {len(frames)} frames.")
    return frames,fps

def extract_frames(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise IOError(f"Cannot open: {path}")
    fps=cap.get(cv2.CAP_PROP_FPS); frames=[]
    while True:
        ret,f=cap.read()
        if not ret: break
        frames.append(f)
    cap.release(); print(f"Loaded {len(frames)} frames"); return frames,fps

def compute_histogram(f):
    h=cv2.cvtColor(cv2.resize(f,RESIZE_DIM),cv2.COLOR_BGR2HSV)
    h1=cv2.normalize(cv2.calcHist([h],[0],None,[64],[0,180]),None).flatten()
    s1=cv2.normalize(cv2.calcHist([h],[1],None,[64],[0,256]),None).flatten()
    return np.concatenate([h1,s1])

def compare(a,b):
    return cv2.compareHist(a.reshape(-1,1).astype(np.float32),b.reshape(-1,1).astype(np.float32),cv2.HISTCMP_CORREL)

def run(frames,fps):
    print("[INFO] Processing frames...")
    hists=[compute_histogram(f) for f in frames]
    raw=np.array([compare(hists[i],hists[i+1]) for i in range(len(hists)-1)])
    smoothed=np.convolve(raw,np.ones(SMOOTH_KERNEL)/SMOOTH_KERNEL,mode='same')
    transitions=[]
    for i,s in enumerate(raw):
        if s<HARD_CUT_THRESHOLD: transitions.append((i+1,"hard_cut",float(s)))
    hcf={t[0] for t in transitions}
    for i in range(WINDOW_SIZE,len(smoothed)-WINDOW_SIZE):
        if not any(abs(i-hf)<WINDOW_SIZE for hf in hcf):
            if np.mean(smoothed[i:i+WINDOW_SIZE])<GRADUAL_THRESHOLD:
                transitions.append((i+WINDOW_SIZE//2,"gradual",float(np.mean(smoothed[i:i+WINDOW_SIZE]))))
    transitions.sort(key=lambda x:x[2]); kept=[]
    for t in transitions:
        if all(abs(t[0]-k[0])>=MIN_DISTANCE for k in kept): kept.append(t)
    transitions=sorted(kept,key=lambda x:x[0])
    print(f"[INFO] Detected {len(transitions)} transitions")
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    bounds=[0]+[t[0] for t in transitions]+[len(frames)]
    h2,w2=frames[0].shape[:2]; fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    for i in range(len(bounds)-1):
        s,e=bounds[i],bounds[i+1]; p=f"{OUTPUT_DIR}/segment_{i+1:03d}.mp4"
        wr=cv2.VideoWriter(p,fourcc,fps,(w2,h2))
        for fr in frames[s:e]: wr.write(fr)
        wr.release(); print(f"  Segment {i+1}: frames {s}-{e} → {p}")
    with open(LOG_FILE,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Frame","Type","Score"])
        for t in transitions: w.writerow(t)
    print(f"\n[DONE] Segments saved to '{OUTPUT_DIR}/' | Log: '{LOG_FILE}'")

if __name__=="__main__":
    print("\n=== Video Segmentation System ===")
    print("1. Use laptop camera")
    print("2. Use a video file")
    choice=input("\nEnter choice (1 or 2): ").strip()
    if choice=="1":
        dur=input(f"Seconds to record? (default {CAMERA_DURATION}): ").strip()
        if dur.isdigit(): CAMERA_DURATION=int(dur)
        frames,fps=capture_from_camera(); run(frames,fps)
    elif choice=="2":
        path=input("Enter video file path: ").strip()
        if not os.path.exists(path): print("File not found"); sys.exit(1)
        frames,fps=extract_frames(path); run(frames,fps)
    else:
        print("Invalid choice."); sys.exit(1)

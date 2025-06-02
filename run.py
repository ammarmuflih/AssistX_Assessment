import cv2
import time
from ultralytics import YOLO
import pandas as pd
import os

model_path = "runs\\detect\\train5_ok\\weights\\best.pt"
video_path = r'6.mp4'
output_folder = "result"
class_name = ["Beyblade"]

counter = [0, 0, 0, 0]
beyblade_data = {}
state = [False, False]
previous = [False, False]
time_start = [0, 0]
time_end = [0, 0]

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

class main():
    def __init__(self):
        pass

    def process(self):
        global cap, model, state, previous, time_start, time_end, process_start, process_end
        motionDetec = motionDetection()
        process_start = time.time()

        output_path = output_folder+"/"+os.path.splitext(video_path)[0]+"_result.mp4"
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            start_t = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(source=frame, persist=True, verbose=False, tracker='bytetrack.yaml')[0]

            if results.boxes.id is not None and len(results.boxes.id) == 2:
                
                for i in state:
                    if i == True:
                        for i, box in enumerate(results.boxes.xyxy):
                            x1, y1, x2, y2 = map(int, box)
                            cls_id = int(results.boxes.cls[i].item())
                            conf = results.boxes.conf[i].item()
                            track_id = int(results.boxes.id[i].item()) if results.boxes.id is not None else -1
                            label = f"{class_name[cls_id]}:{conf:.2f} ID:{track_id}"

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                motionDetec.detect_motion_by_blur(results)

                for i, state_bool in enumerate(state):
                    if state_bool and not previous[i]: 
                        time_start[i] = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

                    elif not state_bool and previous[i]: 
                        time_end[i] = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

                    previous[i] = state_bool  
            
            out.write(frame)
        
        process_end = time.time() 
        cap.release()
        final_data = self.jury(results)
        data_handle.save_data(final_data)

    def jury(self, results):
        global beyblade_data
        final_data = []
        for i, ids in enumerate(results.boxes.id):
            track_id = int(ids.item())
            data = {
                "id": track_id,
                "duration": time_end[i] - time_start[i],
                "picture": beyblade_data[track_id]['latest_crop']  # ✅ Akses by ID
            }
            final_data.append(data)
        
        final_data = sorted(final_data, key=lambda x: x["duration"], reverse=True)
        print(len(beyblade_data))
        return final_data

class motionDetection():
    def __init__(self):
        pass

    def calculate_blur_score(self, cropped_frame):
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var       
    
    def filter(self, is_spin):
        global state
        counter_limit = 6

        if is_spin[0]:
            counter[0] += 1
            counter[1] = 0
        else:
            counter[0] = 0
            counter[1] += 1

        if is_spin[1]:
            counter[2] += 1
            counter[3] = 0
        else:
            counter[2] = 0
            counter[3] += 1

        if counter[0] >= counter_limit:
            state[0] = True
        elif counter[1] >= counter_limit:
            state[0] = False

        if counter[2] >= counter_limit:
            state[1] = True
        elif counter[3] >= counter_limit:
            state[1] = False

    def detect_motion_by_blur(self, detection_results):
        global beyblade_data
        motion_threshold = 5500
        boxes_data = detection_results.boxes
        blur_scores = {}
        is_spin = {}
        
        for j, (box, track_id) in enumerate(zip(boxes_data.xyxy.tolist(), boxes_data.id.tolist())):
            track_id = int(track_id)
            
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(detection_results.orig_shape[1], x2)
            y2 = min(detection_results.orig_shape[0], y2)
            
            object_crop = detection_results.orig_img[y1:y2, x1:x2]
            
            if object_crop.size == 0:
                continue
                
            # Simpan crop berdasarkan track_id
            if track_id not in beyblade_data:
                beyblade_data[track_id] = {}
            beyblade_data[track_id]['latest_crop'] = object_crop
            
            # Deteksi blur
            blur_score = self.calculate_blur_score(object_crop)
            blur_scores[track_id] = blur_score
            
            # Tentukan apakah objek sedang berputar
            if blur_score < motion_threshold:
                is_spin[track_id] = True 
            else:
                is_spin[track_id] = False
        
        # Konversi ke format array untuk filter()
        track_ids = sorted(list(is_spin.keys())) 
        is_spin_array = []
        
        for i in range(min(2, len(track_ids))):
            is_spin_array.append(is_spin[track_ids[i]])
        
        # Pad jika kurang dari 2
        while len(is_spin_array) < 2:
            is_spin_array.append(False)
        
        self.filter(is_spin_array)

class data_handle():
    def __init__(self):
        pass

    def save_data(final_data):
        df = pd.DataFrame(columns=["win_id", "lose_id", "win_dur", "lose_dur", "battle_dur", "dur_difference"])
        if output_folder not in os.listdir():
            os.makedirs(output_folder)

        df.loc[len(df)] = {
            "win_id":final_data[0]["id"], 
            "lose_id":final_data[1]["id"], 
            "win_dur": final_data[0]["duration"], 
            "lose_dur": final_data[1]["duration"],

            "battle_dur": final_data[1]["duration"],
            "dur_difference": final_data[0]["duration"]- final_data[1]["duration"],
            }

        name_file = os.path.splitext(video_path)[0]
        if name_file not in os.listdir(output_folder):
            path = output_folder+"/"+name_file+".csv"
            df.to_csv(path, index=False)

if __name__ == '__main__':
    Main = main()
    Main.process()

    print(f"Total processing time: {process_end - process_start:.2f} seconds")
    print(f"Duration spin 0: {time_end[0] - time_start[0]:.2f} seconds")
    print(f"Duration spin 1: {time_end[1] - time_start[1]:.2f} seconds")
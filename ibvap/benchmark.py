import time
import psutil
import os
import cv2
import threading
import numpy as np
from ultralytics import YOLO
import config

def measure_model_load():
    print("\n[Benchmark] Measuring YOLO26n load & warmup time...")
    start = time.time()
    
    # Check if model exists, ultralytics will download it if not (if officially available)
    try:
        model = YOLO(config.MODEL_PATH)
    except Exception as e:
        print(f"Failed to load {config.MODEL_PATH}. Exception: {e}")
        return None, None
        
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    # Warmup
    model.predict(source=dummy, verbose=False)
    load_time = time.time() - start
    print(f"Model Load & Warmup Time: {load_time:.2f} seconds")
    return model, dummy

def run_benchmark(num_cameras):
    print(f"\n=============================================")
    print(f"[Benchmark] Running {num_cameras}-Camera Simulation...")
    print(f"=============================================")
    
    model, dummy_frame = measure_model_load()
    if not model:
        return
        
    # IBVAP runs 1 node for 3 cameras, 2 nodes for 6 cameras
    num_nodes = 2 if num_cameras > 3 else 1
    cams_per_node = [num_cameras // num_nodes + (1 if x < num_cameras % num_nodes else 0) for x in range(num_nodes)]
    
    print(f"Simulating {num_nodes} AI Nodes processing {num_cameras} cameras.")
    
    stop_event = threading.Event()
    stats = {"frames": 0, "detections": 0, "inference_times": []}
    lock = threading.Lock()
    
    def mock_node_worker(node_id, num_cams):
        local_model = YOLO(config.MODEL_PATH)
        while not stop_event.is_set():
            cycle_start = time.time()
            for _ in range(num_cams):
                inf_start = time.time()
                res = local_model.predict(source=dummy_frame, verbose=False, conf=config.CONF_THRESHOLD)[0]
                inf_time = time.time() - inf_start
                
                with lock:
                    stats["frames"] += 1
                    stats["detections"] += len(res.boxes)
                    stats["inference_times"].append(inf_time)
            
            # Simulate adaptive AI throttling (ACTIVE_FPS = 8 by default)
            spent = time.time() - cycle_start
            sleep_time = max(0, (1.0 / config.ACTIVE_FPS) - spent)
            time.sleep(sleep_time)
            
    threads = []
    for i in range(num_nodes):
        t = threading.Thread(target=mock_node_worker, args=(i, cams_per_node[i]))
        t.start()
        threads.append(t)
        
    # Monitor for 10 seconds
    start_time = time.time()
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None) # Initialize CPU monitor
    
    cpu_measurements = []
    ram_measurements = []
    
    for _ in range(10):
        time.sleep(1)
        cpu_measurements.append(process.cpu_percent(interval=None))
        ram_measurements.append(process.memory_info().rss / 1024 / 1024)
        
    stop_event.set()
    for t in threads:
        t.join()
        
    total_time = time.time() - start_time
    total_frames = stats["frames"]
    fps = total_frames / total_time
    avg_inf_time = (sum(stats["inference_times"]) / len(stats["inference_times"])) * 1000 if stats["inference_times"] else 0
    avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
    avg_ram = sum(ram_measurements) / len(ram_measurements)
    
    print(f"\n--- Results for {num_cameras} Cameras ---")
    print(f"Total processed frames: {total_frames}")
    print(f"Average Inference Time per frame: {avg_inf_time:.2f} ms")
    print(f"Overall System AI FPS: {fps:.2f}")
    print(f"Average CPU Usage: {avg_cpu:.1f}%")
    print(f"Average RAM Usage: {avg_ram:.1f} MB")
    
if __name__ == "__main__":
    print("=== IBVAP YOLO26n Benchmark ===")
    run_benchmark(1)
    run_benchmark(3)
    run_benchmark(6)
    print("\nBenchmark complete. Please update the README.md table with these actual measured values.")

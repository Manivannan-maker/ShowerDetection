import requests
import sys
import numpy as np
import scipy.signal
import pyaudio
from edge_impulse_linux.audio import AudioImpulseRunner

import time
import threading

CHECK_DELAY = 60  # 1 minute

CHECK_FAN_DELAY=1200 #20 minutes
timer_active = False
timer_label = None
timer_active_Fan_On = False
timer_Fan_On_Label=None

#Home assistant configurations
HA_URL = "http://192.168.1.110:8123"  # change to your URL

TOKEN="Your token"
# replace with your token
SERVICE = "mobile_app_phone"  # from step 2
url = f"{HA_URL}/api/services/notify/{SERVICE}"
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}
HA_data = {
    "title": "Switch ON Extractor Fan for next 15-20 Mins",
    "message": "Detected Shower ON , Switch ON Extractor Fan for next 20 Mins to reduce the humidity in the bathroom",
    "data": {
        "clickAction": "/lovelace/home"  # optional
    }
}

HA_data2 = {
    "title": "Switch Off the Extractor Fan ",
    "message": "Switch off the Extractor Fan, Humidity level reduced",
    "data": {
        "clickAction": "/lovelace/home"  # optional
    }
}

# Initialize counters for each label
label_counters = {
    "Shower-ON-Extractor-Fan-Off": 0,
    "Shower-ON-Extractor-Fan-On": 0,
    "Shower-Off-Extractor-Fan-Off": 0,
    "Shower-Off-Extractor-Fan-On": 0
}

THRESHOLD_SCORE = 0.7
DETECTION_COUNT_SHOWER_ON = 5
DETECTION_COUNT_FAN_ON=15

def delayed_check(label):
    global timer_active

    print(f"Timer started for {label}. Checking again in X minutes...")
    time.sleep(CHECK_DELAY)
    #print(label_counters["Shower-Off-Extractor-Fan-On"])

    # After X  minutes, check if label is still detected
    if label_counters["Shower-Off-Extractor-Fan-On"] ==0:
        print(f"Label STILL zero  after X minutes: {label}")
        r = requests.post(url, headers=headers, json=HA_data)
        print(r.status_code, r.text)
    else:
        print(f"Label NOT detected after X minutes: {label}")

    timer_active = False  # allow future timers

def delayed_check_fan_on(label):
    global timer_active_Fan_On
    Fan_on_count_Prev=label_counters["Shower-Off-Extractor-Fan-On"]
    print(f"Timer started for {label}. Checking again in 5 minutes...")
    time.sleep(CHECK_FAN_DELAY)

    # After X  minutes, check if label is still detected
    if label_counters["Shower-Off-Extractor-Fan-On"] >7:
        print(f"Fan is still ON  after X minutes: {label}")
        r = requests.post(url, headers=headers, json=HA_data2)
        print(r.status_code, r.text)
    else:
        print(f"Label NOT detected after X minutes: {label}")
        label_counters["Shower-Off-Extractor-Fan-On"]=0

    timer_active_Fan_On = False  # allow future timers


def main(args):
    global timer_active, timer_label,timer_active_Fan_On,timer_Fan_On_Label
    if len(args) < 1:
        print("Usage: python3 classify_showerdetection.py <model.eim>")
        return

    modelfile = args[0]

    with AudioImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']
            model_rate = model_info['model_parameters']['frequency']
            window_size = model_info['model_parameters']['input_features_count']

            print(f'Loaded runner for "{model_info["project"]["owner"]} / {model_info["project"]["name"]}"')
            print(f"Model expects {model_rate} Hz audio, window size {window_size} samples")

            # Microphone setup
            MIC_RATE = 44100
            CHUNK = 1024

            pa = pyaudio.PyAudio()
            selected_device_id = 0
            print("Using audio device ID:", selected_device_id)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=MIC_RATE,
                input=True,
                input_device_index=selected_device_id,
                frames_per_buffer=CHUNK
            )

            print("Listening...")

            # Buffer for accumulating audio
            buffer_16k = np.zeros(0, dtype=np.int16)

            while True:
                # 1. Read audio from microphone
                data = stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)

                # 2. Resample to model rate
                samples_16k = scipy.signal.resample_poly(samples, model_rate, MIC_RATE).astype(np.int16)


                # 3. Append to buffer
                buffer_16k = np.concatenate((buffer_16k, samples_16k))

                # 4. If we have enough samples, classify
                if len(buffer_16k) >= window_size:
                    window = buffer_16k[:window_size]
                    buffer_16k = buffer_16k[window_size:]  # remove used samples

                    res = runner.classify(window)

                    if "classification" in res["result"]:
                        print("Result:", end=" ")
                        for label in labels:
                            score = res["result"]["classification"][label]
                            print(f"{label}: {score:.2f}", end="  ")
                            # Update counters based on score threshold
                            if score >= THRESHOLD_SCORE:
                                label_counters[label] += 1
                                print(f"\n{label}: {label_counters[label]}\n",end="  ")
                           # else:
                            #    label_counters[label] = 0
                            print(f"\n {label} : {label_counters[label]} \n")
        # Check if counter reached required count
                            if label_counters[label] >= DETECTION_COUNT_SHOWER_ON and (label=="Shower-ON-Extractor-Fan-Off" or label=="Shower-ON-Extractor-Fan-On"):
                                print(f"\nDetected - real scenario: {label}")
                               # r = requests.post(url, headers=headers, json=HA_data)
                               # print(r.status_code, r.text)
                                label_counters[label] = 0  # reset after detection
                                label_counters["Shower-Off-Extractor-Fan-On"]=0
                                # Start timer only once
                                if not timer_active:
                                    timer_active = True
                                    timer_label = label
                                    threading.Thread(
                                    target=delayed_check, args=(label,), daemon=True
                                    ).start()
                                print()
                            if label_counters[label] >= DETECTION_COUNT_FAN_ON and label=="Shower-Off-Extractor-Fan-On":
                                print(f"\n Detected - Fan ON ")
                                if not timer_active_Fan_On:
                                    timer_active_Fan_On=True
                                    timer_Fan_On_Label=label
                                    label_counters[label]=0
                                    threading.Thread(
                                    target=delayed_check_fan_on,args=(label,),daemon=True).start()
                                    print()

                            else:
                                print(res["result"])

        finally:
            runner.stop()


if __name__ == "__main__":
    main(sys.argv[1:])

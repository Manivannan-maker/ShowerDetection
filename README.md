# AntiMould Shower Sentinel
![Heading](/ImageData/image1.jpg)

**Smart device detects shower sound and warns via Home Assistant when the extractor fan is off, preventing steam build-up and stopping mould growth.**

- **Author:** Manivannan
- **Published:** June 25, 2026
- **License:** CC BY-NC-ND
- **Difficulty:** Intermediate
- **Build time:** ~20 hours
- **Project link:** [hackster.io/manivannan/antimould-shower-sentinel-9b0d87](https://www.hackster.io/manivannan/antimould-shower-sentinel-9b0d87)

---

## Table of Contents

1. [Overview](#overview)
2. [Things Used in This Project](#things-used-in-this-project)
3. [Problem Statement](#problem-statement)
4. [Solution](#solution)
5. [Step 1: Set Up Home Assistant on the Raspberry Pi 5](#step-1-set-up-home-assistant-on-the-raspberry-pi-5)
6. [Step 2: Connect the Home Assistant Mobile App](#step-2-connect-the-home-assistant-mobile-app)
7. [Python Code to Run on the Arduino UNO Q](#python-code-to-run-on-the-arduino-uno-q)
8. [Step 3: Train the Edge Impulse ML Model](#step-3-train-the-edge-impulse-ml-model)
9. [Step 4: Deploy the Trained Model to the Arduino UNO Q](#step-4-deploy-the-trained-model-to-the-arduino-uno-q)
10. [Step 5: Integrate the Edge Impulse Model with Home Assistant](#step-5-integrate-the-edge-impulse-model-with-home-assistant)
11. [Hardware Setup](#hardware-setup)
12. [Future Enhancements](#future-enhancements)
13. [Custom Parts and Enclosures](#custom-parts-and-enclosures)
14. [Code Repository](#code-repository)
15. [Credits](#credits)

---

## Overview

**AntiMould Shower Sentinel** is a smart, ML-powered device that listens for the sound of a running shower and instantly detects when the extractor fan hasn't been switched on. By sending real-time alerts to Home Assistant, it prompts the user to activate ventilation before humidity builds up. The result is a low-cost, automated way to prevent mould growth, protect bathroom surfaces, and keep the home healthier — all without installing new sensors or modifying existing wiring.

---

## Things Used in This Project

### Hardware Components
| Component | Quantity |
|---|---|
| Arduino UNO Q | × 1 |
| Raspberry Pi 5 | × 1 |
| Type-C Microphone | × 1 |
| USB-C Hub | × 1 |
| Weather-Resistant Enclosure | × 1 |

### Software Apps and Online Services
- **Arduino IDE** — [arduino.cc/en/main/software](https://www.arduino.cc/en/main/software)
- **Home Assistant** — [home-assistant.io](https://home-assistant.io)
- **Edge Impulse** (ML training/deployment platform)

---

## Problem Statement

In the UK, **mould risk is significantly elevated** because showers generate high moisture loads in small, enclosed bathrooms. When the extractor fan is off, **relative humidity quickly exceeds 70–80%**, pushing surfaces toward the **dew point**, where condensation forms.

UK homes are typically **well-insulated with limited passive airflow**, so this moisture cannot dissipate naturally. The combination of:
- High humidity
- Cold external walls
- Poor ventilation

...creates ideal conditions for rapid mould growth, especially on grout, plasterboard, and silicone seals.

---

## Solution

When a user begins showering, the **Edge Impulse machine-learning model running on the Arduino UNO Q** identifies the acoustic signature of the shower. Once shower activity is confirmed, the system checks whether the **extractor fan is running** and continues monitoring for the next **20 minutes** to ensure proper post-shower ventilation.

If the fan is **not detected** during or after the shower, the Arduino UNO Q sends a signal to **Home Assistant**, running locally on a **Raspberry Pi 5**, which then triggers a mobile notification reminding the user to switch the fan on.

The system also handles the reverse scenario: if the user **forgets to turn the fan off**, Home Assistant sends a notification prompting them to switch it off, reducing unnecessary electricity use.

### Build Steps Summary

1. **Set up Home Assistant** on the Raspberry Pi 5 — the central automation hub.
2. **Connect the Home Assistant mobile app** — enables secure local notifications.
3. **Train an Edge Impulse ML model** — detect shower activity from audio samples.
4. **Deploy the trained model** to the Arduino UNO Q for local edge inference.
5. **Integrate the model with Home Assistant** — automated alerts and energy-saving reminders.

---

## Step 1: Set Up Home Assistant on the Raspberry Pi 5

In this setup, the **Raspberry Pi 5 acts as the Home Assistant server**, while the **Arduino UNO Q functions as the client** running the ML model. The Arduino UNO Q detects both shower and extractor-fan sounds and sends events to Home Assistant, which triggers mobile notifications.

### Prerequisites
- **128 GB SD Card** — Class 10 or better recommended
- **Raspberry Pi 5** — board and case (optional but recommended)
- **Keyboard and Monitor** — HDMI-compatible display
- **Official Raspberry Pi 5 Power Supply** — 27W USB-C PD supply

### Installation

Flash **Home Assistant OS** onto the SD card and boot up the Raspberry Pi. Full instructions:

👉 https://www.home-assistant.io/installation/raspberrypi/

### Connecting to Wi-Fi (if no Ethernet available)

After installing Home Assistant OS:

1. **Open the HA OS Terminal** (via keyboard/monitor connected to the Pi, using the local console).
2. **Scan for available Wi-Fi networks:**
   ```bash
   nmcli device wifi list
   ```
3. **Connect to your Wi-Fi network** (replace with your actual SSID/password):
   ```bash
   nmcli device wifi connect "YOUR_SSID" password "YOUR_PASSWORD"
   ```
   A confirmation message will appear once the Pi joins the network.

---

## Step 2: Connect the Home Assistant Mobile App

With Home Assistant configured and connected to Wi-Fi, the next step is enabling communication between Home Assistant and the mobile app for real-time notifications.

### Install the App
1. Download the **Home Assistant** app (iOS App Store or Android Play Store).
2. Ensure your phone is on the **same Wi-Fi network** as the Raspberry Pi 5.
3. Launch the app.
4. Wait for it to detect `http://homeassistant.local:8123` or the Pi's IP address.
5. **Log in** with your Home Assistant account — this links your phone to your HA user profile.

After login, the device is registered automatically and appears under:
**Settings → Devices & Services → Devices**

### Enable Notifications
- **In the app:** go to **App Configuration → Notifications** and enable them.
- **On your phone:** allow notifications for the Home Assistant app in system settings.

### Test a Notification
1. In Home Assistant, go to **Developer Tools → Services**.
2. Select `notify.mobile_app_<your_phone_name>`.
3. Send a test message:
   ```json
   {"message": "Test notification"}
   ```
4. Confirm the notification appears on your phone.

### Long-Lived Access Token

To connect the Arduino UNO Q, Raspberry Pi (server), and mobile phone — all on the **same Wi-Fi network** — you need a **Long-Lived Access Token**, generated via:

**Profile → Long-Lived Access Tokens → Create Token**

This token securely links the Arduino UNO Q with Home Assistant and the mobile device.

---

## Python Code to Run on the Arduino UNO Q

Before running Python on the Arduino UNO Q, complete the board setup steps described here:

👉 https://docs.edgeimpulse.com/hardware/boards/arduino-uno-q

Test script used to verify end-to-end notification connectivity:

```python
import requests

# Home Assistant URL (use your local IP or homeassistant.local)
HA_URL = "http://homeassistant.local:8123/api/services/notify/mobile_app_your_phone"
# Replace mobile_app_your_phone with your mobile device name in Home Assistant

# Replace with your long-lived access token
TOKEN = "YOUR_LONG_LIVED_ACCESS_TOKEN"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

data = {
    "message": "This was sent via the Home Assistant REST API",
    "title": "Hello from Python"
}

response = requests.post(HA_URL, headers=headers, json=data)

print("Status:", response.status_code)
print("Response:", response.text)
```

Running this script produced a live notification on the author's iPhone, sent directly from the Arduino UNO Q — confirming the end-to-end communication path before building the ML model.

---

## Step 3: Train the Edge Impulse ML Model

### Dataset & Labelling

The workflow uses a structured Edge Impulse pipeline. Audio samples are labelled into **four target classes**:

1. `Shower-ON-Extractor-Fan-Off`
2. `Shower-ON-Extractor-Fan-On`
3. `Shower-Off-Extractor-Fan-Off`
4. `Shower-Off-Extractor-Fan-On`

### Device Connection

Connect the Arduino UNO Q board to your Edge Impulse account:
👉 https://docs.edgeimpulse.com/hardware/boards/arduino-uno-q

> **Note:** The Arduino UNO Q has no built-in microphone, so a **USB Type-C microphone** was connected via a **USB hub** to provide audio input.

After installing the firmware, start the daemon:

```bash
edge-impulse-linux
```

Once connected, navigate to the **Data Acquisition** section and record audio samples for each of the four labelled classes (shower sound, extractor fan sound, and combinations of both).

### Model Training

**Project link:** https://studio.edgeimpulse.com/public/982849/live

1. In **Create Impulse**, set the preprocessing block to **MFE** (Mel Filterbank Energy) and the learning block to **Classification**.
2. Generate features and visualize them to get a high-level overview of each label.

### Neural Network Settings

- **Training cycles:** 100
- **Learning rate:** 0.005
- **Architecture:** Reshape layer (converts audio to a 1D array) → 1D Convolution layer → Dropout layer (for improved accuracy/generalization)

**Result:** The model achieved **100% accuracy** during training.

### Model Testing

The model was evaluated against a held-out test dataset not used during training, achieving **100% accuracy** — sufficient for hardware deployment.

---

## Step 4: Deploy the Trained Model to the Arduino UNO Q

### Download the Trained Model

```bash
edge-impulse-linux-runner --download modelfile.eim
```

### Install the Linux SDK

Full docs: https://docs.edgeimpulse.com/tools/libraries/sdks/inference/linux/python

1. Install **Python 3** (>= 3.7).
2. Install the SDK:
   ```bash
   pip3 install edge_impulse_linux
   ```
3. Clone the examples repository:
   ```bash
   git clone https://github.com/edgeimpulse/linux-sdk-python
   ```
4. (Optional) Install dependencies for camera/microphone examples:
   ```bash
   cd linux-sdk-python
   pip install -r requirements.txt
   ```

### Custom Inference Script

The stock `classify.py` example (in `linux-sdk-python/examples/audio`) is **not directly compatible** with the Arduino UNO Q, since it can't read audio data at the required frequency. A modified version — `classify_EI.py` — was created that loads audio buffers based on time-based sampling and resamples them to match the model's expected rate. It's available in the author's GitHub repository.

Place `classify_EI.py` and `modelfile.eim` into `linux-sdk-python/examples/audio` and run:

```bash
python3 classify_EI.py modelfile.eim
```

---

## Step 5: Integrate the Edge Impulse Model with Home Assistant

### Audio Classification Pipeline

The system continuously receives audio data, buffers it, and passes it into the customized **`classify_EI.py`** script. The embedded **EIM model** generates a classification result for each audio window.

A **classification label counter** inside `classify_EI.py` tracks how many consecutive times each label appears:

- When the label is `Shower-ON-Extractor-Fan-Off` **or** `Shower-ON-Extractor-Fan-On`, and the count exceeds **`DETECTION_COUNT_SHOWER_ON`** (15 seconds) → the system concludes the **shower is active** and starts a timer to verify the extractor fan is still ON after **5 minutes** (configurable). If the fan condition isn't met, a notification prompts the user to switch it on.
- When the label is `Shower-Off-Extractor-Fan-On` **or** `Shower-ON-Extractor-Fan-On`, and the count exceeds **`DETECTION_COUNT_FAN_ON`** (1 minute) → the system concludes the **fan is active** and starts a second timer to check whether the fan is still active after **20 minutes** (configurable).

### Shower Active Timer Callback

- When the **shower-active timer** expires, the system checks whether the label count for `Shower-Off-Extractor-Fan-On` is **zero**.
- If zero → the extractor fan is **not running** while the shower is active.
- The system **sends a notification to Home Assistant**, prompting activation of the extractor fan for **15–20 minutes**.
- Home Assistant relays this to the user's smartwatch or preferred device.

### Fan Active Timer Callback

- When the **fan-active timer** expires, the system checks whether the label count for `Shower-Off-Extractor-Fan-On` is **greater than 7** (configurable threshold).
- If true → the fan has run long enough to reduce humidity.
- The system **sends a notification to Home Assistant** instructing it to switch off the fan.
- Home Assistant relays this to the user's smartwatch, confirming humidity has reduced.

---

## Hardware Setup

In this prototype, a **weather-resistant enclosure** houses the Arduino UNO Q, which is powered by a **portable power bank**. For long-term deployment, a dedicated **wired 5V power connection** should replace the portable power source.

---

## Future Enhancements

The project can be extended toward **full autonomy** by adding **Home Assistant automations** that automatically switch the extractor fan **on** or **off** directly, based on shower-detection events — removing the need for manual user action entirely.

---

## Custom Parts and Enclosures

- Weather-resistant enclosure housing the Arduino UNO Q and microphone assembly (see project screenshot on Hackster.io).

---

## Code Repository

**Python Code:** https://github.com/Manivannan-maker/ShowerDetection.git

---

## Credits

**Manivannan** — Engineer by profession, solving real-world problems by passion.
25 projects • 136 followers on Hackster.io
🔗 https://www.hackster.io/manivannan

---

*Source project page: [AntiMould Shower Sentinel on Hackster.io](https://www.hackster.io/manivannan/antimould-shower-sentinel-9b0d87) — Published June 25, 2026, licensed under CC BY-NC-ND.*

# ☁️ How to Run Rakshak AI on Google Colab

Since Rakshak AI is typically a local application (using Audio/Webcam), running it on Colab requires a few special steps to "stream" the interface to your browser.

## Step 1: Prepare the Files
1.  **Zip the entire folder** (`car_detection`).
2.  Go to [Google Colab](https://colab.research.google.com/).
3.  Create a **New Notebook**.
4.  Open the **Files** sidebar (folder icon on the left).
5.  **Upload** your zip file (`car_detection.zip`).
6.  Run this command to unzip:
    ```python
    !unzip car_detection.zip
    %cd car_detection
    ```

## Step 2: Install Dependencies
Copy and paste this code into a new code cell and run it:

```python
# Install system dependencies for OpenCV
!apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python libraries
!pip install streamlit pyngrok ultralytics opencv-python-headless geopy pygame pyttsx3

# (Optional) Install localtunnel to expose the port
!npm install -g localtunnel
```

## Step 3: Run the Application
Google Colab runs on a remote server, so you cannot see the "localhost:8501" window directly. We use a tunnel.

Copy and paste this into a cell:

```python
import urllib
print("Password/Endpoint IP for Tunnel:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

# Run Streamlit in the background
!streamlit run main.py &>/content/logs.txt &

# Expose the port using localtunnel
!npx localtunnel --port 8501
```

## Step 4: Access the App
1.  The output will show a link: `your-url.loca.lt`.
2.  **Click the link.**
3.  It might ask for a **Tunnel Password** or **Endpoint IP**.
4.  Copy the IP address printed at the top of Step 3 output (e.g., `34.12.145.2`) and paste it there.
5.  **Rakshak AI will open!**

## ⚠️ Notes for Colab Users
*   **No Audio**: The beep sounds will not play (Colab servers don't have speakers). The logs will show `[BEEP]` instead.
*   **No Webcam**: Changing mode to "Live Camera" will likely fail or show a black screen. **Please use "Video File" mode** and upload a test video.

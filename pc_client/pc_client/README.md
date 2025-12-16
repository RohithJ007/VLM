# PC Client for PiStream

This folder contains the client-side application to receive the live camera stream from your Raspberry Pi and run your VLM (Vision Language Model) or other ML models.

## Structure

*   `run_client.py`: The main application that displays the video feed and runs the model.
*   `vlm/model.py`: A dedicated file for your model logic. **Edit this file to add your actual model.**
*   `requirements.txt`: Python dependencies.

## Setup

1.  **Install Dependencies**
    Open a terminal in this folder and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure the Stream URL**
    *   Get the Pinggy URL from your Raspberry Pi (it looks like `https://xxxx-xxxx.pinggy.link`).
    *   Open `run_client.py` in a text editor.
    *   Find the line:
        ```python
        STREAM_URL = "http://<YOUR_PINGGY_URL>/video_feed"
        ```
    *   Replace `<YOUR_PINGGY_URL>` with your actual URL. Ensure it ends with `/video_feed`.

## Usage

Run the client:

```bash
python run_client.py
```

A window will open showing the live stream. The output from the model in `vlm/model.py` will be displayed on the video overlay.

## Adding Your Model

To integrate your own VLM or ML model:

1.  Open `vlm/model.py`.
2.  Initialize your model in the `__init__` method.
3.  Implement your inference logic in the `analyze(frame)` method.
4.  Return the result as a string (or any format you want to display).

The `run_client.py` script automatically handles the video loop and visualization.

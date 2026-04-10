import cv2
import time
import numpy as np
import threading
from abc import ABC, abstractmethod

class VideoSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray]:
        pass
    
    @abstractmethod
    def release(self):
        pass
    
    @abstractmethod
    def isOpened(self) -> bool:
        pass
    
    @abstractmethod
    def get_source_type(self) -> str:
        pass

class WebcamSource(VideoSource):
    def __init__(self, source):
        if str(source).isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        
    def read(self):
        return self.cap.read()
        
    def release(self):
        self.cap.release()
        
    def isOpened(self):
        return self.cap.isOpened()
        
    def get_source_type(self):
        return 'webcam'

import requests

class ESP32Source(VideoSource):
    def __init__(self, url: str, reconnect_timeout: int = 5):
        # Auto-fix common mistake where users forget the path
        if url.startswith("http://") and url.count("/") == 2:
            url = url + "/video"
        
        self.url = url
        self.reconnect_timeout = reconnect_timeout
        self._is_opened = False
        self._running = False
        self._frame = None
        self._thread = None
        self._connect()

    def _connect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            
        print(f"📡 Connecting to ESP32 stream via HTTP: {self.url}")
        self._running = True
        self._is_opened = True
        self._thread = threading.Thread(target=self._stream_reader, daemon=True)
        self._thread.start()

    def _stream_reader(self):
        bytes_buffer = b''
        while self._running:
            try:
                # stream=True keeps the connection open
                res = requests.get(self.url, stream=True, timeout=10)
                if res.status_code != 200:
                    self._is_opened = False
                    time.sleep(2)
                    continue
                    
                self._is_opened = True
                print("✅ ESP32 Stream connected")
                
                for chunk in res.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    if chunk:
                        bytes_buffer += chunk
                        # JPEG start and end markers
                        a = bytes_buffer.find(b'\xff\xd8')
                        b = bytes_buffer.find(b'\xff\xd9')
                        
                        if a != -1 and b != -1:
                            jpg = bytes_buffer[a:b+2]
                            bytes_buffer = bytes_buffer[b+2:]
                            
                            idx = np.frombuffer(jpg, dtype=np.uint8)
                            frame = cv2.imdecode(idx, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                self._frame = frame
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ ESP32 Connection dropped: {e}")
                self._is_opened = False
                time.sleep(2)

    def read(self) -> tuple[bool, np.ndarray]:
        if not self._is_opened:
            if not self._running:
                self._connect()
            return False, None

        if self._frame is not None:
            return True, self._frame.copy()
            
        return False, None

    def release(self):
        self._running = False
        self._is_opened = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def isOpened(self):
        return self._is_opened

    def get_source_type(self):
        return 'esp32'


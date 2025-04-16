"""
OpenCV compatibility module that provides fallback functionality when OpenCV is not available.
"""

# Try to import cv2, but provide fallback functionality
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) is not available. Video processing features will be limited.")

# Error class to use when OpenCV is required but not available
class OpenCVNotAvailableError(Exception):
    """Exception raised when OpenCV is required but not available."""
    def __init__(self, message="OpenCV (cv2) is required for this operation but is not available"):
        self.message = message
        super().__init__(self.message)

def check_cv2_available():
    """Check if OpenCV is available and raise an error if not."""
    if not CV2_AVAILABLE:
        raise OpenCVNotAvailableError()
    return True 
import cv2
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(img_path: Path | None = None) -> np.ndarray:
    """
    Load an image from file path or use a sample image if no path provided.

    Parameters
    ----------
    img_path : Path | None
        Path to image file, by default None (uses sample image from skimage)

    Returns
    -------
    np.ndarray
        Loaded image in BGR format
    """
    if img_path is None:
        img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
        return img

    if not img_path.exists():
        raise FileNotFoundError(f"File '{img_path}' not found")

    if not img_path.is_file():
        raise ValueError(f"'{img_path}' is not a file")

    img = cv2.imread(img_path.as_posix())
    if img is None:
        raise ValueError(f"Could not read image '{img_path}'. "
                         "Check file format and integrity.")

    return img


def prepare_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare image for processing by converting to grayscale if needed.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing (original BGR image, grayscale image)
    """
    img_bgr = img.copy()

    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    return img_bgr, img_gray


def initialize_cascades() -> tuple[cv2.CascadeClassifier | None, cv2.CascadeClassifier | None]:
    """
    Initialize Haar cascade classifiers for face and upper body detection.

    Returns
    -------
    tuple
        Tuple containing (face_cascade, upper_body_cascade) or (None, None) on error

    Raises
    ------
    RuntimeError
        If cascades cannot be loaded
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        return face_cascade, upper_body_cascade
    except Exception as e:
        raise RuntimeError(f"Failed to load cascade classifiers: {e}")


def detect_face(face_cascade: cv2.CascadeClassifier | None, gray_image: np.ndarray) -> np.ndarray:
    """
    Detect faces in grayscale image using Haar cascade.

    Parameters
    ----------
    face_cascade : cv2.CascadeClassifier | None
        Face cascade classifier
    gray_image : np.ndarray
        Grayscale input image

    Returns
    -------
    np.ndarray
        Array of detected faces as (x, y, width, height) rectangles
    """
    if face_cascade is None:
        return np.array([])

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def detect_upper_body(upper_body_cascade: cv2.CascadeClassifier | None,
                      gray_image: np.ndarray,
                      scaleFactor: float,
                      minNeighbors: int) -> np.ndarray:
    """
    Detect upper bodies in grayscale image using Haar cascade.

    Parameters
    ----------
    upper_body_cascade : cv2.CascadeClassifier | None
        Upper body cascade classifier
    gray_image : np.ndarray
        Grayscale input image
    scaleFactor : float
        Scale factor for multi-scale detection
    minNeighbors : int
        Minimum number of neighbors for detection

    Returns
    -------
    np.ndarray
        Array of detected upper bodies as (x, y, width, height) rectangles
    """
    if upper_body_cascade is None:
        return np.array([])

    upper_bodies = upper_body_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(50, 50)
    )
    return upper_bodies


def create_detection_visualization(original_img: np.ndarray,
                                   faces: np.ndarray,
                                   upper_bodies: np.ndarray) -> np.ndarray:
    """
    Create visualization of detected regions (faces and upper bodies).

    Parameters
    ----------
    original_img : np.ndarray
        Original input image
    faces : np.ndarray
        Array of detected faces
    upper_bodies : np.ndarray
        Array of detected upper bodies

    Returns
    -------
    np.ndarray
        Image with detection rectangles drawn
    """
    detection_img = original_img.copy()
    thickness = max(1, original_img.shape[0] // 300)

    for (x, y, w, h) in faces:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (255, 0, 0), thickness)  # Blue - face

    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 255, 0), thickness)  # Green - upper body

    return detection_img


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image from BGR to RGB color space for matplotlib display.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format

    Returns
    -------
    np.ndarray
        Image in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_pose_simplified(image_path: Path | None = None) -> None:
    """
    Main function for simplified pose detection using Haar cascades.

    Parameters
    ----------
    image_path : Path | None
        Path to input image, or None for sample image
    """
    try:
        img = load_image(image_path)
        img_bgr, img_gray = prepare_image(img)

        face_cascade, upper_body_cascade = initialize_cascades()

        faces = detect_face(face_cascade, img_gray)

        # Try different parameters for upper body detection
        for minNeighbors in range(2, 10):
            for scaleFactor in range(1, 500):
                coef = 1.0 + scaleFactor / 100
                upper_bodies = detect_upper_body(upper_body_cascade, img_gray, coef, minNeighbors)
                print(coef)
                if len(upper_bodies) != 0:
                    detection_img = create_detection_visualization(img_bgr, faces, upper_bodies)
                    Path(f"coef_finding/{"astronaut" if image_path is None else image_path.stem}").mkdir(parents=True, exist_ok=True)
                    filename = f"coef_finding/{"astronaut" if image_path is None else image_path.stem}/scaleFactor={coef:.3f}_minNeighbors={minNeighbors}.jpg"
                    cv2.imwrite(filename, detection_img)

    except Exception as e:
        print(f"Error in pose detection: {e}")


def test() -> None:
    """
    Test function to run pose detection on sample images.
    """
    detect_pose_simplified(Path("img.png"))
    detect_pose_simplified()


if __name__ == "__main__":
    test()
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from skimage import data


def load_image(img_path: Path | None = None) -> np.ndarray:
    """
    Load an image from file path or use a sample image if no path provided.

    Parameters
    ----------
    img_path : Path or None
        Path to image file, by default None (uses sample image from skimage)

    Returns
    -------
    np.ndarray
        Loaded image in BGR format

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If the path is not a file or image cannot be read
    """
    if img_path is None:
        img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
        return img

    if not img_path.exists():
        raise FileNotFoundError(f"File '{img_path}' not found")

    if not img_path.is_file():
        raise ValueError(f"'{img_path}' is not a file")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    if img_path.suffix.lower() not in image_extensions:
        raise ValueError(f"'{img_path}' has wrong file extension")

    img = cv2.imread(img_path.as_posix())
    if img is None:
        raise ValueError(f"Could not read image '{img_path}'. "
                         "Check file format and integrity.")

    return img


def get_output_path(input_path: Path | None = None) -> Path:
    """
    Determine the output path for the detected image based on input path.

    Parameters
    ----------
    input_path : Path or None
        Path to input image or None for sample image

    Returns
    -------
    Path
        Output path for the detected image
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    if input_path is None:
        script_dir = Path(__file__).parent
        return script_dir / "sample_result/astronaut_detected.png"

    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            return input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        else:
            return input_path.parent / f"{input_path.stem}_detected.png"

    return Path(f"{input_path.stem}_detected.png")


def initialize_pose_detector() -> tuple:
    """
    Initialize MediaPipe Pose detector with custom configuration.

    Returns
    -------
    tuple
        Contains pose detector object and drawing utilities
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return pose, mp_drawing, mp_pose


def create_custom_connections() -> list[tuple[int, int]]:
    """
    Create custom connections only for head, shoulders, elbows and wrists.

    Returns
    -------
    list
        List of connection tuples for specified body parts
    """
    connections = [
        # Shoulders to elbows
        (11, 13), (12, 14),
        # Elbows to wrists
        (13, 15), (14, 16),
        # Shoulders connection
        (11, 12)
    ]
    return connections


def create_green_drawing_spec() -> tuple:
    """
    Create green drawing specifications for landmarks and connections.

    Returns
    -------
    tuple
        Contains landmark and connection drawing specs in green color
    """
    green_color = (0, 255, 0)

    landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
        color=green_color, thickness=2, circle_radius=3
    )

    connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
        color=green_color, thickness=3
    )

    return landmark_drawing_spec, connection_drawing_spec


def detect_pose_landmarks(pose_detector, image: np.ndarray):
    """
    Detect pose landmarks in the given image.

    Parameters
    ----------
    pose_detector : mediapipe.solutions.pose.Pose
        Initialized pose detector object
    image : np.ndarray
        Input image for detection

    Returns
    -------
    mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Detected pose landmarks or None if no detection
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_image)
    return results.pose_landmarks


def calculate_face_center_point(landmarks) -> tuple[float, float] | None:
    """
    Calculate center point of face from multiple facial landmarks.

    Parameters
    ----------
    landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Detected pose landmarks

    Returns
    -------
    tuple
        (x, y) coordinates of face center point
    """
    # Use key facial landmarks: nose, eyes, ears, mouth
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Face landmarks indices

    x_coords = []
    y_coords = []

    for idx in face_indices:
        if idx < len(landmarks.landmark):
            landmark = landmarks.landmark[idx]
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)

    if x_coords and y_coords:
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return center_x, center_y

    return None


def calculate_shoulders_center(landmarks) -> tuple[float, float] | None:
    """
    Calculate center point between shoulders.

    Parameters
    ----------
    landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Detected pose landmarks

    Returns
    -------
    tuple
        (x, y) coordinates of shoulders center point
    """
    if len(landmarks.landmark) > 12:
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]

        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2

        return center_x, center_y

    return None


def draw_custom_pose_landmarks(image: np.ndarray, landmarks,
                               connections: list[tuple[int, int]],
                               drawing_specs: tuple) -> None:
    """
    Draw only specified landmarks and connections on the image.

    Parameters
    ----------
    image : np.ndarray
        Input image to draw on
    landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Detected pose landmarks
    connections : list
        List of connection tuples to draw
    drawing_specs : tuple
        Contains landmark and connection drawing specifications
    """
    landmark_drawing_spec, connection_drawing_spec = drawing_specs

    if landmarks:
        image_height, image_width = image.shape[:2]

        face_center = calculate_face_center_point(landmarks)
        if face_center:
            face_px = (int(face_center[0] * image_width), int(face_center[1] * image_height))
            cv2.circle(image, face_px, 8, landmark_drawing_spec.color, -1)

        shoulders_center = calculate_shoulders_center(landmarks)
        if shoulders_center and face_center:
            shoulders_px = (int(shoulders_center[0] * image_width),
                            int(shoulders_center[1] * image_height))
            face_px = (int(face_center[0] * image_width), int(face_center[1] * image_height))

            cv2.line(image, face_px, shoulders_px, connection_drawing_spec.color,
                     connection_drawing_spec.thickness)

            cv2.circle(image, shoulders_px, 5, landmark_drawing_spec.color, -1)

        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                start_point = landmarks.landmark[start_idx]
                end_point = landmarks.landmark[end_idx]

                start_px = (int(start_point.x * image_width), int(start_point.y * image_height))
                end_px = (int(end_point.x * image_width), int(end_point.y * image_height))

                cv2.line(image, start_px, end_px, connection_drawing_spec.color,
                         connection_drawing_spec.thickness)

        relevant_indices = set()
        for connection in connections:
            relevant_indices.add(connection[0])
            relevant_indices.add(connection[1])

        for idx in relevant_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                landmark_px = (int(landmark.x * image_width), int(landmark.y * image_height))
                cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
                           landmark_drawing_spec.color, landmark_drawing_spec.thickness)


def save_image(image: np.ndarray, output_path: Path) -> bool:
    """
    Save image to the specified path.

    Parameters
    ----------
    image : np.ndarray
        Image to save
    output_path : Path
        Path where to save the image

    Returns
    -------
    bool
        True if image was saved successfully, False otherwise
    """
    if isinstance(output_path, Path):
        output_path_str = output_path.as_posix()
    else:
        output_path_str = str(output_path)

    success = cv2.imwrite(output_path_str, image)
    return success


def process_image_with_pose_detection(pose_detector : mp.solutions.pose.Pose,
                                      image_src_path: Path | None = None,
                                      image_out_path: Path | None = None) -> np.ndarray:
    """
    Main function to process image and detect upper body pose.

    Parameters
    ----------
    pose_detector : mediapipe.solutions.pose.Pose
        Initialized pose detector object
    image_src_path : Path or None
        Path to input image or None for sample image
    image_out_path : Path or None
        Path to output image or None for auto-generated path

    Returns
    -------
    np.ndarray
        Processed image with pose detection
    """
    if image_src_path is not None and not image_src_path.exists():
        raise FileNotFoundError(f"Image file '{image_src_path}' does not exist")

    if image_src_path is not None and not isinstance(image_src_path, Path):
        image_src_path = Path(image_src_path)

    if image_out_path is not None and not isinstance(image_out_path, Path):
        image_out_path = Path(image_out_path)

    image = load_image(image_src_path)

    custom_connections = create_custom_connections()
    drawing_specs = create_green_drawing_spec()

    landmarks = detect_pose_landmarks(pose_detector, image)

    if landmarks is None:
        return image

    draw_custom_pose_landmarks(image, landmarks, custom_connections, drawing_specs)

    if image_out_path is None:
        image_out_path = get_output_path(image_src_path)

    if image_out_path.is_dir():
        image_out_path = image_out_path / f"{image_src_path.stem}_detected.png"

    image_out_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(image, image_out_path)

    return image


def test():
    pose_detector, mp_drawing, mp_pose = initialize_pose_detector()
    with pose_detector:
        process_image_with_pose_detection(pose_detector, None, None)
        process_image_with_pose_detection(pose_detector, Path("sample_pictures_raw/img.png"), Path("sample_result"))
        process_image_with_pose_detection(pose_detector, Path("sample_pictures_raw/img_1.png"), Path("sample_result/1.jpg"))
        process_image_with_pose_detection(pose_detector, Path("sample_pictures_raw/img_2.png"), Path("sample_result/2.jpg"))


if __name__ == "__main__":
    test()
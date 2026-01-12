import os
import uuid
import numpy as np
import cv2
import pydicom
from PIL import Image


class DicomProcessor:
    def __init__(self, upload_dir):
        self.upload_dir = upload_dir

    def _get_file_path(self, filename):
        return os.path.join(self.upload_dir, filename)

    def _generate_output_filename(self, prefix):
        return f"{prefix}_{uuid.uuid4().hex[:8]}.png"

    def _save_image(self, image_array, output_filename):
        output_path = self._get_file_path(output_filename)
        cv2.imwrite(output_path, image_array)
        return output_filename

    def _load_image(self, filename):
        file_path = self._get_file_path(filename)
        if filename.lower().endswith('.dcm') or self._is_dicom(file_path):
            return self._load_dicom(file_path)
        else:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Cannot load image: {filename}")
            return img

    def _is_dicom(self, file_path):
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except:
            return False

    def _load_dicom(self, file_path):
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(float)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        return pixel_array, ds

    def _normalize_to_uint8(self, img):
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img, dtype=np.uint8)
        return normalized

    def extract_metadata(self, filename):
        file_path = self._get_file_path(filename)
        ds = pydicom.dcmread(file_path)
        return {
            'patientId': str(getattr(ds, 'PatientID', 'N/A')),
            'patientName': str(getattr(ds, 'PatientName', 'N/A')),
            'studyDate': str(getattr(ds, 'StudyDate', 'N/A')),
            'modality': str(getattr(ds, 'Modality', 'N/A')),
            'studyDescription': str(getattr(ds, 'StudyDescription', 'N/A')),
            'rows': int(getattr(ds, 'Rows', 0)),
            'columns': int(getattr(ds, 'Columns', 0)),
            'bitsAllocated': int(getattr(ds, 'BitsAllocated', 0)),
            'windowCenter': float(getattr(ds, 'WindowCenter', [40])[0]) if hasattr(ds, 'WindowCenter') else 40,
            'windowWidth': float(getattr(ds, 'WindowWidth', [400])[0]) if hasattr(ds, 'WindowWidth') else 400,
        }

    def convert_to_png(self, filename, window_center=None, window_width=None):
        file_path = self._get_file_path(filename)
        pixel_array, ds = self._load_dicom(file_path)
        
        if window_center is None:
            window_center = float(getattr(ds, 'WindowCenter', [40])[0]) if hasattr(ds, 'WindowCenter') else 40
        if window_width is None:
            window_width = float(getattr(ds, 'WindowWidth', [400])[0]) if hasattr(ds, 'WindowWidth') else 400
        
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = np.clip(pixel_array, img_min, img_max)
        pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        output_filename = self._generate_output_filename('converted')
        return self._save_image(pixel_array, output_filename)

    def apply_grayscale(self, filename):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = img[0]
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = self._normalize_to_uint8(img)
        output_filename = self._generate_output_filename('grayscale')
        return self._save_image(gray, output_filename)

    def apply_blur(self, filename, kernel_size=5):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        output_filename = self._generate_output_filename('blur')
        return self._save_image(blurred, output_filename)

    def apply_resize(self, filename, width, height):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        output_filename = self._generate_output_filename('resized')
        return self._save_image(resized, output_filename)

    def apply_edge_detection(self, filename, method='canny'):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method == 'canny':
            edges = cv2.Canny(img, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = self._normalize_to_uint8(edges)
        elif method == 'laplacian':
            edges = cv2.Laplacian(img, cv2.CV_64F)
            edges = self._normalize_to_uint8(np.abs(edges))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        output_filename = self._generate_output_filename(f'edge_{method}')
        return self._save_image(edges, output_filename)

    def calculate_histogram(self, filename):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        return hist.flatten().tolist()

    def apply_histogram_equalization(self, filename):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(img)
        output_filename = self._generate_output_filename('hist_eq')
        return self._save_image(equalized, output_filename)

    def apply_filter(self, filename, filter_type='sharpen'):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        
        if filter_type == 'sharpen':
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(img, -1, kernel)
        elif filter_type == 'emboss':
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            filtered = cv2.filter2D(img, -1, kernel)
        elif filter_type == 'median':
            filtered = cv2.medianBlur(img, 5)
        elif filter_type == 'bilateral':
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            filtered = cv2.bilateralFilter(img, 9, 75, 75)
            if len(filtered.shape) == 3:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unknown filter: {filter_type}")
        
        output_filename = self._generate_output_filename(f'filter_{filter_type}')
        return self._save_image(filtered, output_filename)

    def apply_windowing(self, filename, window_center, window_width):
        file_path = self._get_file_path(filename)
        if self._is_dicom(file_path):
            pixel_array, _ = self._load_dicom(file_path)
        else:
            pixel_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(float)
        
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed = np.clip(pixel_array, img_min, img_max)
        windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        output_filename = self._generate_output_filename('windowed')
        return self._save_image(windowed, output_filename)

    def apply_invert(self, filename):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        inverted = cv2.bitwise_not(img)
        output_filename = self._generate_output_filename('inverted')
        return self._save_image(inverted, output_filename)

    def apply_rotate(self, filename, angle):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        
        if angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h))
        
        output_filename = self._generate_output_filename('rotated')
        return self._save_image(rotated, output_filename)

    def apply_flip(self, filename, direction='horizontal'):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        
        if direction == 'horizontal':
            flipped = cv2.flip(img, 1)
        elif direction == 'vertical':
            flipped = cv2.flip(img, 0)
        else:
            flipped = cv2.flip(img, -1)
        
        output_filename = self._generate_output_filename('flipped')
        return self._save_image(flipped, output_filename)

    def apply_clahe(self, filename, clip_limit=2.0, tile_size=8):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(img)
        output_filename = self._generate_output_filename('clahe')
        return self._save_image(enhanced, output_filename)

    def apply_brightness_contrast(self, filename, brightness=0, contrast=1.0):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        output_filename = self._generate_output_filename('brightness_contrast')
        return self._save_image(adjusted, output_filename)

    def apply_gamma_correction(self, filename, gamma=1.0):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(img, table)
        output_filename = self._generate_output_filename('gamma')
        return self._save_image(corrected, output_filename)

    def apply_denoise(self, filename, strength=10):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        if len(img.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
        output_filename = self._generate_output_filename('denoised')
        return self._save_image(denoised, output_filename)

    def apply_unsharp_mask(self, filename, sigma=1.0, strength=1.5):
        img = self._load_image(filename)
        if isinstance(img, tuple):
            img = self._normalize_to_uint8(img[0])
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
        output_filename = self._generate_output_filename('unsharp')
        return self._save_image(sharpened, output_filename)

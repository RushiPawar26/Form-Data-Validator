import os
import cv2
import numpy as np
import pytesseract

# --- Base path for templates ---
TEMPLATES_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'templates')
print(f"📁 Templates dir: {TEMPLATES_DIR}")

# --- Per doc type config ---
# Regions as (x%, y%, w%, h%) of image dimensions
DOC_CONFIG = {
    'college id': {
        'alignment_threshold_valid': 0.60,
        'alignment_threshold_suspicious': 0.35,
        'logo_region': (0.0, 0.0, 0.30, 0.20),
        'logo_text': ['pict','pune institue of computer technology'],  # varies by college, rely on visual match
        'photo_region': (0.38, 0.22, 0.24, 0.32),
        'photo_min_area_ratio': 0.02,
    },
    'aadhaar card': {
        'alignment_threshold_valid': 0.35,
        'alignment_threshold_suspicious': 0.15,
        'logo_region': (0.0, 0.0, 0.25, 0.20),
        'logo_text': ['uidai', 'unique identification', 'भारत सरकार', 'government of india'],
        'photo_region': (0.72, 0.30, 0.25, 0.45),
        'photo_min_area_ratio': 0.08,
    },
    'adhaar card': {
        'skip_alignment': False,
        'skip_logo': True,
        'skip_photo': False,
        'alignment_threshold_valid': 0.30,
        'alignment_threshold_suspicious': 0.15,
        'logo_region': (0.0, 0.0, 0.25, 0.20),
        'logo_text': ['uidai', 'unique identification', 'भारत सरकार', 'government of india'],
        'photo_region': (0.02, 0.25, 0.22, 0.55),  # left side
        'photo_min_area_ratio': 0.02,
    },
    'pan card': {
        'alignment_threshold_valid': 0.35,
        'alignment_threshold_suspicious': 0.15,
        'logo_region': (0.0, 0.0, 0.30, 0.25),
        'logo_text': ['income tax', 'government of india', 'भारत सरकार'],
        'photo_region': (0.65, 0.35, 0.30, 0.50),
        'photo_min_area_ratio': 0.08,
    },
    'marksheet': {
        'skip_alignment': True,  # marksheet content varies per student, alignment not reliable
        'alignment_threshold_valid': 0.35,
        'alignment_threshold_suspicious': 0.15,
        'logo_region': (0.35, 0.0, 0.30, 0.15),
        'logo_text': [],  # visual match handles this
        'photo_region': None,
        'photo_min_area_ratio': 0.08,
    },
}


def _get_template_path(doc_type, filename):
    # normalize spelling variations (e.g. adhaar vs aadhaar)
    folder = doc_type.lower().replace(' ', '_').replace('adhaar', 'aadhaar')
    return os.path.join(TEMPLATES_DIR, folder, filename)


def _crop_region(img, region):
    """Crop image using percentage-based region (x%, y%, w%, h%)."""
    h, w = img.shape[:2]
    x = int(region[0] * w)
    y = int(region[1] * h)
    cw = int(region[2] * w)
    ch = int(region[3] * h)
    return img[y:y+ch, x:x+cw]


def _pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV grayscale numpy array."""
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def check_alignment(doc_pil, doc_type):
    config = DOC_CONFIG.get(doc_type, {})

    if config.get('skip_alignment'):
        return {'alignment_score': None, 'alignment_status': 'skip', 'alignment_note': 'Skipped for this doc type'}

    template_path = _get_template_path(doc_type, 'template.png')
    if not os.path.exists(template_path):
        return {'alignment_score': None, 'alignment_status': 'skip', 'alignment_note': 'No template found'}

    threshold_valid = config.get('alignment_threshold_valid', 0.40)
    threshold_suspicious = config.get('alignment_threshold_suspicious', 0.20)

    # load template and uploaded doc, resize to same size
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    doc_img = _pil_to_cv2(doc_pil)
    doc_img = cv2.resize(doc_img, (template_img.shape[1], template_img.shape[0]))

    # find keypoints in both images
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(doc_img, None)

    if des1 is None or des2 is None:
        return {'alignment_score': 0.0, 'alignment_status': 'suspicious', 'alignment_note': 'No keypoints detected'}

    # match keypoints between template and uploaded doc
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # keep only close matches
    good_matches = [m for m in matches if m.distance < 50]
    score = round(len(good_matches) / max(len(kp1), 1), 3)

    if score >= threshold_valid:
        status = 'valid'
    else:
        status = 'suspicious'

    print(f"Alignment score: {score} → {status}")
    return {'alignment_score': score, 'alignment_status': status, 'alignment_note': f'{len(good_matches)} keypoints matched'}


def check_logo(doc_pil, doc_type):
    """
    Crop logo region from both template and uploaded doc.
    1. Histogram correlation for visual match
    2. OCR text check for known logo keywords
    Returns: dict with score, matched bool, text_matched bool
    """
    template_path = _get_template_path(doc_type, 'template.png')
    config = DOC_CONFIG.get(doc_type, {})
    logo_region = config.get('logo_region')
    logo_texts = list(config.get('logo_text', []))

    if not logo_region:
        return {'logo_score': None, 'logo_matched': None, 'logo_text_matched': None, 'logo_note': 'No logo region defined for this doc type'}

    doc_img = _pil_to_cv2(doc_pil)

    # --- OCR text check on logo region ---
    logo_crop_pil = doc_pil.crop((
        int(logo_region[0] * doc_pil.width),
        int(logo_region[1] * doc_pil.height),
        int((logo_region[0] + logo_region[2]) * doc_pil.width),
        int((logo_region[1] + logo_region[3]) * doc_pil.height),
    ))
    logo_ocr_text = pytesseract.image_to_string(logo_crop_pil, lang='eng+hin').lower()
    print(f"🏷️ Logo OCR text: {logo_ocr_text.strip()[:100]}")

    text_matched = None
    if logo_texts:
        text_matched = any(kw in logo_ocr_text for kw in logo_texts)
        print(f"🏷️ Logo keywords checked: {logo_texts} → {text_matched}")

    # --- Visual histogram match (needs template) ---
    if not os.path.exists(template_path):
        return {
            'logo_score': None,
            'logo_matched': None,
            'logo_text_matched': text_matched,
            'logo_note': f'No template found. Text match: {text_matched}'
        }

    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    doc_img_resized = cv2.resize(doc_img, (template_img.shape[1], template_img.shape[0]))

    template_logo = _crop_region(template_img, logo_region)
    doc_logo = _crop_region(doc_img_resized, logo_region)

    if template_logo.size == 0 or doc_logo.size == 0:
        return {'logo_score': None, 'logo_matched': None, 'logo_text_matched': text_matched, 'logo_note': 'Logo crop failed'}

    hist_template = cv2.calcHist([template_logo], [0], None, [256], [0, 256])
    hist_doc = cv2.calcHist([doc_logo], [0], None, [256], [0, 256])
    cv2.normalize(hist_template, hist_template)
    cv2.normalize(hist_doc, hist_doc)

    score = round(cv2.compareHist(hist_template, hist_doc, cv2.HISTCMP_CORREL), 3)
    visual_matched = score >= 0.50

    # Overall: pass if either visual OR text matches (both being None means skip)
    logo_matched = visual_matched or (text_matched is True)

    print(f"🏷️ Logo visual score: {score}, text_matched: {text_matched} → {'matched' if logo_matched else 'no match'}")
    return {
        'logo_score': score,
        'logo_matched': logo_matched,
        'logo_text_matched': text_matched,
        'logo_note': f'Visual: {score}, Text: {text_matched}'
    }


def check_photo(doc_pil, doc_type):
    """
    Detect if a photo (large rectangle) exists in the expected region.
    Returns: dict with photo_present bool
    """
    config = DOC_CONFIG.get(doc_type, {})
    photo_region = config.get('photo_region')
    min_area_ratio = config.get('photo_min_area_ratio', 0.08)

    if not photo_region:
        return {'photo_present': None, 'photo_note': 'No photo region defined for this doc type'}

    doc_img = _pil_to_cv2(doc_pil)
    region_crop = _crop_region(doc_img, photo_region)

    if region_crop.size == 0:
        return {'photo_present': False, 'photo_note': 'Photo region crop failed'}

    # Save debug crop to see what region is being checked
    debug_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'photo_debug.png')
    cv2.imwrite(debug_path, region_crop)
    print(f"📷 Debug crop saved to: {debug_path}")

    # Edge detection + contour finding — lower thresholds for blurry/scanned photos
    blurred = cv2.GaussianBlur(region_crop, (5, 5), 0)
    edges = cv2.Canny(blurred, 10, 50)
    # Dilate edges to close gaps from blur
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_area = region_crop.shape[0] * region_crop.shape[1]
    photo_found = False
    best_ratio = 0.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h  # use bounding box area instead of contour area
        ratio = area / region_area
        if ratio > best_ratio:
            best_ratio = ratio
        if ratio >= min_area_ratio:
            aspect = w / max(h, 1)
            if 0.3 <= aspect <= 3.0:
                photo_found = True
                break

    print(f"📷 Photo region size: {region_crop.shape}, contours: {len(contours)}, best area ratio: {round(best_ratio, 4)}, found: {photo_found}")
    return {'photo_present': photo_found, 'photo_note': 'Contour detection in expected region'}


def _normalize_doc_type(doc_type: str) -> str:
    """Normalize doc type string to match DOC_CONFIG keys."""
    import re
    d = doc_type.lower().strip()
    if re.search(r'adh?a+r', d):
        return 'aadhaar card'
    if 'marksheet' in d or 'mark sheet' in d:
        return 'marksheet'
    if 'college' in d or 'student id' in d:
        return 'college id'
    if 'pan' in d:
        return 'pan card'
    return d


def run_template_checks(doc_pil, doc_type):
    doc_type = _normalize_doc_type(doc_type)
    config = DOC_CONFIG.get(doc_type, {})
    results = {}

    if config.get('skip_alignment'):
        results.update({'alignment_score': None, 'alignment_status': 'skip', 'alignment_note': 'Skipped for this doc type'})
    else:
        results.update(check_alignment(doc_pil, doc_type))

    if config.get('skip_logo'):
        results.update({'logo_score': None, 'logo_matched': None, 'logo_text_matched': None, 'logo_note': 'Skipped for this doc type'})
    else:
        results.update(check_logo(doc_pil, doc_type))

    if config.get('skip_photo'):
        results.update({'photo_present': None, 'photo_note': 'Skipped for this doc type'})
    else:
        results.update(check_photo(doc_pil, doc_type))

    return results

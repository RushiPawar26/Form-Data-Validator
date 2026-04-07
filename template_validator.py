import os
import re
import cv2
import numpy as np
import pytesseract

IS_PROD = os.environ.get("PRODUCTION", "").lower() in ("1", "true", "yes")

def _log(msg):
    if not IS_PROD:
        print(msg)

TEMPLATES_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'templates')
_log(f"📁 Templates dir: {TEMPLATES_DIR}")

# photo_region can be a single tuple OR list of tuples (tries all, passes if any found)
DOC_CONFIG = {
    'college id': {
        'alignment_threshold_valid': 0.60,
        'alignment_threshold_suspicious': 0.35,
        'logo_region': (0.0, 0.0, 0.30, 0.20),
        'logo_text': ['pict', 'pune institute of computer technology', 'institute'],
        'photo_region': [
            (0.38, 0.22, 0.24, 0.32),
            (0.65, 0.20, 0.30, 0.40),
            (0.02, 0.20, 0.25, 0.40),
        ],
        'photo_min_area_ratio': 0.02,
    },
    'aadhaar card': {
        'skip_alignment': True,
        'skip_logo': True,
        'alignment_threshold_valid': 0.35,
        'alignment_threshold_suspicious': 0.15,
        'logo_region': (0.0, 0.0, 0.25, 0.20),
        'logo_text': ['uidai', 'unique identification', 'भारत सरकार', 'government of india'],
        'photo_region': [
            (0.72, 0.25, 0.25, 0.50),
            (0.02, 0.25, 0.22, 0.55),
            (0.60, 0.20, 0.35, 0.55),
        ],
        'photo_min_area_ratio': 0.02,
    },
    'pan card': {
        'alignment_threshold_valid': 0.35,
        'alignment_threshold_suspicious': 0.15,
        'logo_region': (0.0, 0.0, 0.30, 0.25),
        'logo_text': ['income tax', 'government of india', 'भारत सरकार'],
        'photo_region': [
            (0.65, 0.35, 0.30, 0.50),
            (0.60, 0.30, 0.35, 0.55),
        ],
        'photo_min_area_ratio': 0.05,
    },
    'marksheet': {
        'skip_alignment': True,
        'skip_photo': True,
        'alignment_threshold_valid': 0.35,
        'alignment_threshold_suspicious': 0.15,
        'logo_region': (0.30, 0.0, 0.40, 0.18),
        'logo_text': [],
        'photo_region': None,
        'photo_min_area_ratio': 0.08,
    },
}


def _normalize_doc_type(doc_type: str) -> str:
    d = doc_type.lower().strip()
    if re.search(r'adh?a+r|aadhar|aadhaar', d):
        return 'aadhaar card'
    if 'marksheet' in d or 'mark sheet' in d:
        return 'marksheet'
    if 'college' in d or 'student id' in d:
        return 'college id'
    if 'pan' in d:
        return 'pan card'
    return d


def _get_template_path(doc_type, filename):
    folder = doc_type.lower().replace(' ', '_')
    return os.path.join(TEMPLATES_DIR, folder, filename)


def _crop_region(img, region):
    h, w = img.shape[:2]
    x, y = int(region[0] * w), int(region[1] * h)
    cw, ch = int(region[2] * w), int(region[3] * h)
    return img[y:y+ch, x:x+cw]


def _pil_to_cv2(pil_image):
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def check_alignment(doc_pil, doc_type):
    config = DOC_CONFIG.get(doc_type, {})
    if config.get('skip_alignment'):
        return {'alignment_score': None, 'alignment_status': 'skip', 'alignment_note': 'Skipped'}

    template_path = _get_template_path(doc_type, 'template.png')
    if not os.path.exists(template_path):
        return {'alignment_score': None, 'alignment_status': 'skip', 'alignment_note': 'No template found'}

    threshold_valid = config.get('alignment_threshold_valid', 0.40)
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    doc_img = cv2.resize(_pil_to_cv2(doc_pil), (template_img.shape[1], template_img.shape[0]))

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(doc_img, None)

    if des1 is None or des2 is None:
        return {'alignment_score': 0.0, 'alignment_status': 'suspicious', 'alignment_note': 'No keypoints'}

    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    good = [m for m in matches if m.distance < 50]
    score = round(len(good) / max(len(kp1), 1), 3)
    status = 'valid' if score >= threshold_valid else 'suspicious'
    _log(f"Alignment: {score} → {status}")
    return {'alignment_score': score, 'alignment_status': status, 'alignment_note': f'{len(good)} keypoints matched'}


def check_logo(doc_pil, doc_type):
    config = DOC_CONFIG.get(doc_type, {})
    logo_region = config.get('logo_region')
    logo_texts = list(config.get('logo_text', []))

    if not logo_region:
        return {'logo_score': None, 'logo_matched': None, 'logo_text_matched': None, 'logo_note': 'No logo region'}

    logo_crop_pil = doc_pil.crop((
        int(logo_region[0] * doc_pil.width),
        int(logo_region[1] * doc_pil.height),
        int((logo_region[0] + logo_region[2]) * doc_pil.width),
        int((logo_region[1] + logo_region[3]) * doc_pil.height),
    ))
    # OCR on logo region — not logged in production
    logo_ocr_text = pytesseract.image_to_string(logo_crop_pil, lang='eng+hin').lower()

    text_matched = None
    if logo_texts:
        text_matched = any(kw in logo_ocr_text for kw in logo_texts)
        _log(f"🏷️ Logo text match: {text_matched}")

    template_path = _get_template_path(doc_type, 'template.png')
    if not os.path.exists(template_path):
        return {'logo_score': None, 'logo_matched': None, 'logo_text_matched': text_matched,
                'logo_note': f'No template. Text: {text_matched}'}

    doc_img = _pil_to_cv2(doc_pil)
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    doc_img_r = cv2.resize(doc_img, (template_img.shape[1], template_img.shape[0]))

    tl = _crop_region(template_img, logo_region)
    dl = _crop_region(doc_img_r, logo_region)
    if tl.size == 0 or dl.size == 0:
        return {'logo_score': None, 'logo_matched': None, 'logo_text_matched': text_matched, 'logo_note': 'Crop failed'}

    ht = cv2.calcHist([tl], [0], None, [256], [0, 256])
    hd = cv2.calcHist([dl], [0], None, [256], [0, 256])
    cv2.normalize(ht, ht); cv2.normalize(hd, hd)
    score = round(cv2.compareHist(ht, hd, cv2.HISTCMP_CORREL), 3)
    logo_matched = score >= 0.50 or (text_matched is True)
    _log(f"🏷️ Logo visual: {score}, text: {text_matched} → {logo_matched}")
    return {'logo_score': score, 'logo_matched': logo_matched, 'logo_text_matched': text_matched,
            'logo_note': f'Visual:{score} Text:{text_matched}'}


def check_photo(doc_pil, doc_type):
    config = DOC_CONFIG.get(doc_type, {})
    photo_region = config.get('photo_region')
    min_area_ratio = config.get('photo_min_area_ratio', 0.05)

    if not photo_region:
        return {'photo_present': None, 'photo_note': 'No photo region defined'}

    regions = photo_region if isinstance(photo_region, list) else [photo_region]
    doc_img = _pil_to_cv2(doc_pil)

    for idx, region in enumerate(regions):
        crop = _crop_region(doc_img, region)
        if crop.size == 0:
            continue
        blurred = cv2.GaussianBlur(crop, (5, 5), 0)
        edges = cv2.dilate(cv2.Canny(blurred, 10, 50), np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_total = crop.shape[0] * crop.shape[1]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (w * h) / area_total >= min_area_ratio and 0.3 <= w / max(h, 1) <= 3.0:
                _log(f"📷 Photo found in region {idx+1}")
                return {'photo_present': True, 'photo_note': f'Found in region {idx+1}'}

    return {'photo_present': False, 'photo_note': f'Not found in {len(regions)} regions'}


def run_template_checks(doc_pil, doc_type):
    doc_type = _normalize_doc_type(doc_type)
    config = DOC_CONFIG.get(doc_type, {})
    results = {}

    if config.get('skip_alignment'):
        results.update({'alignment_score': None, 'alignment_status': 'skip', 'alignment_note': 'Skipped'})
    else:
        results.update(check_alignment(doc_pil, doc_type))

    if config.get('skip_logo'):
        results.update({'logo_score': None, 'logo_matched': None, 'logo_text_matched': None, 'logo_note': 'Skipped'})
    else:
        results.update(check_logo(doc_pil, doc_type))

    if config.get('skip_photo'):
        results.update({'photo_present': None, 'photo_note': 'Skipped'})
    else:
        results.update(check_photo(doc_pil, doc_type))

    return results

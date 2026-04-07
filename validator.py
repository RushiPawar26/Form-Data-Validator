import os
import re
import cv2
import numpy as np
from difflib import SequenceMatcher
from pdf2image import convert_from_path
from PIL import Image as PILImage
import pytesseract
from template_validator import run_template_checks


def normalize_doc_type(raw: str) -> str:
    d = raw.lower().strip()
    if re.search(r'adh?a+r|aadhar|aadhaar', d):
        return 'aadhaar card'
    if 'pan' in d:
        return 'pan card'
    if 'marksheet' in d or 'mark sheet' in d or 'result' in d:
        return 'marksheet'
    if 'college' in d or 'student id' in d or 'institute' in d:
        return 'college id'
    if 'birth' in d:
        return 'birth certificate'
    return d


def fuzzy_match(a, b, threshold=0.60):
    return SequenceMatcher(None, a, b).ratio() >= threshold


def ocr_image(pil_img):
    """Try 4 preprocessing strategies, return the one with most text."""
    img_np = np.array(pil_img)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    results = []

    # 1. original
    results.append(pytesseract.image_to_string(pil_img, lang='eng+hin', config='--psm 3'))

    # 2. histogram equalization
    eq = cv2.equalizeHist(img_np)
    results.append(pytesseract.image_to_string(PILImage.fromarray(eq), lang='eng+hin', config='--psm 3'))

    # 3. adaptive threshold — best for low-contrast scans
    thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    results.append(pytesseract.image_to_string(PILImage.fromarray(thresh), lang='eng+hin', config='--psm 3'))

    # 4. 2x upscale + adaptive threshold — helps small/dense text
    up = cv2.resize(img_np, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thresh2 = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    results.append(pytesseract.image_to_string(PILImage.fromarray(thresh2), lang='eng+hin', config='--psm 3'))

    best = max(results, key=lambda t: len(t.strip()))
    print(f"� OCR lengths: {[len(r.strip()) for r in results]} → using {len(best.strip())} chars")
    return best


def render_pdf(pdf_path):
    poppler_path = os.environ.get("POPPLER_PATH") or None
    for dpi in [300, 200, 150]:
        try:
            kwargs = dict(first_page=1, last_page=1, dpi=dpi, grayscale=True)
            if poppler_path:
                kwargs["poppler_path"] = poppler_path
            images = convert_from_path(pdf_path, **kwargs)
            if images:
                return images
        except Exception as e:
            print(f"DPI {dpi} failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Confidence scoring — 0 to 100. Pass >= 50, suspicious 50-69, valid >= 70
# ---------------------------------------------------------------------------

SCORE_WEIGHTS = {
    'aadhaar card': {
        'doc_keywords': 20,
        'name':         35,
        'dob':          20,
        'gender':       10,
        'aadhaar_num':  15,
    },
    'pan card': {
        'doc_keywords': 25,
        'name':         40,
        'id_number':    35,
    },
    'marksheet': {
        'doc_keywords': 20,
        'name':         40,
        'department':   25,
        'logo':         15,
    },
    'college id': {
        'doc_keywords': 15,
        'name':         35,
        'id_number':    25,
        'department':   15,
        'photo':        10,
    },
    'birth certificate': {
        'doc_keywords': 30,
        'name':         40,
        'dob':          30,
    },
}

DOC_KEYWORDS = {
    'aadhaar card': [
        'aadhaar', 'आधार', 'uidai', 'unique identification',
        'government of india', 'govt of india', 'भारत सरकार',
        'enrollment', 'vid:', 'dob', 'date of birth'
    ],
    'pan card': [
        'income tax', 'permanent account', 'pan',
        'भारत सरकार', 'government of india'
    ],
    'marksheet': [
        'university', 'board', 'examination', 'result', 'marks',
        'grade', 'semester', 'annual', 'certificate', 'seat no',
        'subject', 'total'
    ],
    'college id': [
        'college', 'university', 'institute', 'student',
        'id card', 'identity', 'department', 'branch'
    ],
    'birth certificate': [
        'birth certificate', 'date of birth', 'born',
        'municipal', 'registrar'
    ],
}


def score_document(full_text, form_data, doc_type, template_results):
    weights = SCORE_WEIGHTS.get(doc_type, {'doc_keywords': 25, 'name': 50, 'id_number': 25})
    score = 0
    breakdown = {}
    reasons = []

    name       = form_data.get('name', '').lower().strip()
    id_field   = str(form_data.get('id_number') or form_data.get('roll_number') or
                     form_data.get('registration_number') or '').lower().replace(' ', '')
    department = form_data.get('department', '').lower().strip()
    aadhaar_num = str(form_data.get('aadhaar_number', '')).strip()

    # doc_keywords
    if 'doc_keywords' in weights:
        keywords = DOC_KEYWORDS.get(doc_type, [])
        matched = [k for k in keywords if k in full_text]
        if matched:
            score += weights['doc_keywords']
            breakdown['doc_keywords'] = f"PASS ({', '.join(matched[:3])})"
        else:
            reasons.append(f"No {doc_type} keywords found")
            breakdown['doc_keywords'] = 'FAIL'

    # name
    if 'name' in weights and name:
        name_found = name in full_text or any(
            fuzzy_match(name, full_text[i:i+len(name)])
            for i in range(len(full_text) - len(name) + 1)
        )
        if name_found:
            score += weights['name']
            breakdown['name'] = 'PASS'
        else:
            reasons.append(f"Name '{name}' not found")
            breakdown['name'] = 'FAIL'
    elif 'name' in weights:
        score += weights['name'] // 2
        breakdown['name'] = 'SKIP'

    # dob
    if 'dob' in weights:
        dob_found = (
            any(k in full_text for k in ['dob', 'date of birth', 'जन्म', 'year of birth', 'yob']) or
            bool(re.search(r'\d{2}[\/\-]\d{2}[\/\-]\d{4}', full_text)) or
            bool(re.search(r'\d{4}[\/\-]\d{2}[\/\-]\d{2}', full_text))
        )
        if dob_found:
            score += weights['dob']
            breakdown['dob'] = 'PASS'
        else:
            reasons.append('Date of birth not found')
            breakdown['dob'] = 'FAIL'

    # gender (minor — doesn't add to reasons)
    if 'gender' in weights:
        if any(k in full_text for k in ['female', 'male', 'महिला', 'पुरुष']):
            score += weights['gender']
            breakdown['gender'] = 'PASS'
        else:
            breakdown['gender'] = 'FAIL (minor)'

    # aadhaar last 4
    if 'aadhaar_num' in weights:
        if aadhaar_num:
            last4 = aadhaar_num[-4:]
            if last4 in full_text.replace(' ', ''):
                score += weights['aadhaar_num']
                breakdown['aadhaar_num'] = f'PASS (last4={last4})'
            else:
                reasons.append(f"Aadhaar last 4 '{last4}' not found")
                breakdown['aadhaar_num'] = 'FAIL'
        else:
            score += weights['aadhaar_num'] // 2
            breakdown['aadhaar_num'] = 'SKIP'

    # id_number
    if 'id_number' in weights:
        if id_field:
            clean = full_text.replace(' ', '')
            id_found = id_field in clean or any(
                fuzzy_match(id_field, clean[i:i+len(id_field)], threshold=0.80)
                for i in range(max(len(clean) - len(id_field) + 1, 1))
            )
            if id_found:
                score += weights['id_number']
                breakdown['id_number'] = 'PASS'
            else:
                reasons.append(f"ID '{id_field}' not found")
                breakdown['id_number'] = 'FAIL'
        else:
            score += weights['id_number'] // 2
            breakdown['id_number'] = 'SKIP'

    # department
    if 'department' in weights:
        if department:
            dept_first = department.split()[0]
            dept_found = (
                department in full_text or
                dept_first in full_text or
                any(fuzzy_match(department, full_text[i:i+len(department)])
                    for i in range(len(full_text) - len(department) + 1))
            )
            if dept_found:
                score += weights['department']
                breakdown['department'] = 'PASS'
            else:
                reasons.append(f"Department '{department}' not found")
                breakdown['department'] = 'FAIL'
        else:
            score += weights['department'] // 2
            breakdown['department'] = 'SKIP'

    # logo
    if 'logo' in weights:
        lm = template_results.get('logo_matched')
        if lm is True:
            score += weights['logo']
            breakdown['logo'] = 'PASS'
        elif lm is False:
            breakdown['logo'] = 'FAIL (minor)'
        else:
            score += weights['logo'] // 2
            breakdown['logo'] = 'SKIP'

    # photo
    if 'photo' in weights:
        pp = template_results.get('photo_present')
        if pp is True:
            score += weights['photo']
            breakdown['photo'] = 'PASS'
        elif pp is False:
            breakdown['photo'] = 'FAIL (minor)'
        else:
            score += weights['photo'] // 2
            breakdown['photo'] = 'SKIP'

    return min(score, 100), breakdown, reasons


def process_document(pdf_path, form_data):
    raw_doc_type = form_data.get('doc_type', '')
    doc_type = normalize_doc_type(raw_doc_type)
    print(f"\n📄 '{raw_doc_type}' → '{doc_type}' for {form_data.get('name')}")

    images = render_pdf(pdf_path)
    if not images:
        return {'doc_type': doc_type, 'status': 'Error: Could not render PDF.', 'validation_passed': False}

    full_text = ocr_image(images[0]).lower()
    print(f"🔍 OCR ({len(full_text)} chars): {full_text[:300]}")

    template_results = run_template_checks(images[0], doc_type)
    score, breakdown, reasons = score_document(full_text, form_data, doc_type, template_results)

    print(f"📊 Score: {score}/100 | {breakdown}")

    if score >= 50:
        suspicious = score < 70
        return {
            'doc_type': doc_type,
            'validation_passed': True,
            'confidence_score': score,
            'status': f"Valid ({score}/100)" + (" — low confidence, review recommended" if suspicious else ""),
            'suspicious': suspicious,
            'breakdown': breakdown,
            **template_results,
        }
    else:
        return {
            'doc_type': doc_type,
            'validation_passed': False,
            'confidence_score': score,
            'status': (f"Invalid ({score}/100): " + '; '.join(reasons)) if reasons else f"Invalid ({score}/100)",
            'breakdown': breakdown,
            **template_results,
        }


def validate_multiple_documents(form_data):
    results = []
    for doc in form_data.get("documents", []):
        pdf_path = doc.get("pdf_path")
        if not pdf_path:
            results.append({'doc_type': doc.get("doc_type"), 'status': "No file path.", 'validation_passed': False})
            continue
        results.append(process_document(pdf_path, doc))
    return results

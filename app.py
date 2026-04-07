import os
import re
import sys
import uuid
import logging
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from validator import validate_multiple_documents, process_document
from auth import init_db, require_api_key, register, login, regenerate_key

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Upload dir (works on Render, Docker, local) ---
UPLOAD_FOLDER = '/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, force=True)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.INFO)

# --- Rate limiter: 60 requests/minute per IP ---
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri=os.environ.get("REDIS_URL", "memory://"),
)

# --- Init DB on startup ---
with app.app_context():
    init_db()


def _temp_path(suffix="pdf"):
    """Generate a unique temp file path to avoid collisions across concurrent requests."""
    return os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{suffix}")


def download_from_drive(file_id, save_path):
    session = http_requests.Session()
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(download_url, timeout=(10, 60))

    if 'text/html' in resp.headers.get('Content-Type', ''):
        token_match = re.search(r'confirm=([0-9A-Za-z_\-]+)', resp.text)
        if token_match:
            resp = session.get(f"{download_url}&confirm={token_match.group(1)}", timeout=(10, 60))
        else:
            raise Exception('File is not publicly accessible on Google Drive')

    if resp.status_code != 200:
        raise Exception(f'Failed to download file: {resp.status_code}')

    with open(save_path, 'wb') as f:
        f.write(resp.content)


@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "Validator API is live"}), 200


# --- Auth routes ---
@app.route('/auth/register', methods=['POST'])
@limiter.limit("10 per hour")
def register_route():
    return register()


@app.route('/auth/login', methods=['POST'])
@limiter.limit("20 per minute")
def login_route():
    return login()


@app.route('/auth/regenerate-key', methods=['POST'])
@limiter.limit("5 per hour")
def regenerate_key_route():
    return regenerate_key()


@app.route('/validate', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def validate_document_endpoint():
    app.logger.info("=== /validate hit ===")

    form_data = (
        request.form.to_dict()
        if request.content_type and 'multipart' in request.content_type
        else (request.json or request.form.to_dict())
    )
    form_data = dict(form_data) or {}

    temp_filepath = _temp_path("pdf")

    try:
        file_id = form_data.pop('file_id', None)

        if file_id:
            app.logger.info(f"Downloading file from Drive, id={file_id}")
            download_from_drive(file_id, temp_filepath)

        elif 'file' in request.files:
            file = request.files['file']
            mime_type = file.content_type or ''
            filename = file.filename or ''
            if not filename.lower().endswith('.pdf') and mime_type != 'application/pdf':
                return jsonify({'error': 'Only PDF files allowed'}), 400
            file.save(temp_filepath)

        else:
            return jsonify({'error': 'No file or file_id provided'}), 400

        app.logger.info(f"Saved temp file: {temp_filepath}")
        validation_result = process_document(temp_filepath, form_data)
        return jsonify(validation_result)

    except Exception as e:
        app.logger.exception("Processing error")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


@app.route('/validate-multiple', methods=['POST'])
@require_api_key
@limiter.limit("20 per minute")
def validate_multiple_endpoint():
    app.logger.info("=== /validate-multiple hit ===")

    form_data = (
        request.form.to_dict()
        if request.content_type and 'multipart' in request.content_type
        else (request.get_json(silent=True) or request.form.to_dict())
    )
    form_data = dict(form_data) or {}
    # log only keys, never values (PII protection)
    app.logger.info(f"Received fields: {list(form_data.keys())}")

    name = form_data.get('name', '')
    id_number = form_data.get('id_number', '')
    department = form_data.get('department', '')
    aadhaar_number = form_data.get('aadhaar_number', '')

    # Build list of docs to process
    docs = []
    for i in range(1, 4):
        file_id = form_data.get(f'file_id_{i}')
        doc_type = form_data.get(f'doc_type_{i}', '')
        if file_id and doc_type:
            docs.append((file_id, doc_type))

    if not docs:
        return jsonify({'validation_passed': False, 'results': [], 'error': 'No documents provided'}), 400

    def process_one(file_id, doc_type):
        temp_path = _temp_path("pdf")
        try:
            download_from_drive(file_id, temp_path)
            return process_document(temp_path, {
                'name': name,
                'id_number': id_number,
                'department': department,
                'doc_type': doc_type,
                'aadhaar_number': aadhaar_number,
            })
        except Exception as e:
            return {'doc_type': doc_type, 'status': str(e), 'validation_passed': False}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Process all docs in parallel — each on its own thread
    results = [None] * len(docs)
    with ThreadPoolExecutor(max_workers=len(docs)) as executor:
        future_to_idx = {executor.submit(process_one, fid, dt): idx
                         for idx, (fid, dt) in enumerate(docs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {'doc_type': docs[idx][1], 'status': str(e), 'validation_passed': False}

    all_passed = all(r.get('validation_passed') for r in results)
    return jsonify({'validation_passed': all_passed, 'results': results})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

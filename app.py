from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import jwt
import os
import uuid
from datetime import datetime, timedelta
from processors.dicom_processor import DicomProcessor

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dicom-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/image_processing_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Fix for Render PostgreSQL URL
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)

UPLOAD_DIR = os.environ.get('UPLOAD_DIR', './uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

db = SQLAlchemy(app)
processor = DicomProcessor(UPLOAD_DIR)

# ==================== MODELS ====================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.BigInteger, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100))
    role = db.Column(db.String(20), default='USER')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class Image(db.Model):
    __tablename__ = 'images'
    id = db.Column(db.BigInteger, primary_key=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    stored_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.BigInteger)
    mime_type = db.Column(db.String(50))
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    patient_id = db.Column(db.String(100))
    patient_name = db.Column(db.String(255))
    study_date = db.Column(db.String(20))
    modality = db.Column(db.String(20))
    study_description = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ProcessedImage(db.Model):
    __tablename__ = 'processed_images'
    id = db.Column(db.BigInteger, primary_key=True)
    original_image_id = db.Column(db.BigInteger, db.ForeignKey('images.id'), nullable=False)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)
    operation = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.JSON)
    stored_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==================== AUTH MIDDLEWARE ====================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'error': 'Token không hợp lệ'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['userId'])
            if not current_user:
                return jsonify({'error': 'User không tồn tại'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token đã hết hạn'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token không hợp lệ'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

# ==================== AUTH ROUTES ====================
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Tên đăng nhập đã tồn tại'}), 400
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email đã tồn tại'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password=generate_password_hash(data['password']),
            full_name=data.get('fullName', '')
        )
        db.session.add(user)
        db.session.commit()
        
        token = jwt.encode({
            'userId': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + timedelta(days=7)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'userId': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not check_password_hash(user.password, data['password']):
            return jsonify({'error': 'Tên đăng nhập hoặc mật khẩu không đúng'}), 401
        
        token = jwt.encode({
            'userId': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + timedelta(days=7)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'userId': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ==================== IMAGE ROUTES ====================
@app.route('/api/images/upload', methods=['POST'])
@token_required
def upload_image(current_user):
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        original_name = secure_filename(file.filename)
        ext = os.path.splitext(original_name)[1]
        stored_name = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_DIR, stored_name)
        
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        # Extract DICOM metadata if applicable
        patient_id = None
        patient_name = None
        study_date = None
        modality = None
        
        if ext.lower() in ['.dcm', '.dicom']:
            try:
                metadata = processor.extract_metadata(stored_name)
                patient_id = metadata.get('patientId')
                patient_name = metadata.get('patientName')
                study_date = metadata.get('studyDate')
                modality = metadata.get('modality')
            except:
                pass
        
        image = Image(
            user_id=current_user.id,
            original_name=original_name,
            stored_name=stored_name,
            file_path=file_path,
            file_size=file_size,
            patient_id=patient_id,
            patient_name=patient_name,
            study_date=study_date,
            modality=modality
        )
        db.session.add(image)
        db.session.commit()
        
        return jsonify({
            'id': image.id,
            'originalName': image.original_name,
            'storedName': image.stored_name,
            'fileSize': image.file_size
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/images', methods=['GET'])
@token_required
def get_images(current_user):
    try:
        images = Image.query.filter_by(user_id=current_user.id).order_by(Image.created_at.desc()).all()
        return jsonify([{
            'id': img.id,
            'originalName': img.original_name,
            'storedName': img.stored_name,
            'fileSize': img.file_size,
            'patientId': img.patient_id,
            'modality': img.modality,
            'createdAt': img.created_at.isoformat() if img.created_at else None
        } for img in images])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/images/<int:image_id>', methods=['GET'])
@token_required
def get_image(current_user, image_id):
    try:
        image = Image.query.filter_by(id=image_id, user_id=current_user.id).first()
        if not image:
            return jsonify({'error': 'Không tìm thấy ảnh'}), 404
        
        return jsonify({
            'id': image.id,
            'originalName': image.original_name,
            'storedName': image.stored_name,
            'fileSize': image.file_size,
            'patientId': image.patient_id,
            'modality': image.modality,
            'createdAt': image.created_at.isoformat() if image.created_at else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/images/<int:image_id>/file', methods=['GET'])
@token_required
def get_image_file(current_user, image_id):
    try:
        image = Image.query.filter_by(id=image_id, user_id=current_user.id).first()
        if not image:
            return jsonify({'error': 'Không tìm thấy ảnh'}), 404
        
        file_path = os.path.join(UPLOAD_DIR, image.stored_name)
        if os.path.exists(file_path):
            return send_file(file_path)
        return jsonify({'error': 'File không tồn tại'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/images/<int:image_id>', methods=['DELETE'])
@token_required
def delete_image(current_user, image_id):
    try:
        image = Image.query.filter_by(id=image_id, user_id=current_user.id).first()
        if not image:
            return jsonify({'error': 'Không tìm thấy ảnh'}), 404
        
        # Delete file
        file_path = os.path.join(UPLOAD_DIR, image.stored_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        db.session.delete(image)
        db.session.commit()
        
        return jsonify({'message': 'Đã xóa ảnh'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


# ==================== PROCESSING ROUTES ====================
@app.route('/api/process/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'dicom-processing-api'})

@app.route('/api/process/file/<filename>', methods=['GET'])
@token_required
def get_processed_file(current_user, filename):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/dicom/metadata/<filename>', methods=['GET'])
@token_required
def get_dicom_metadata(current_user, filename):
    try:
        metadata = processor.extract_metadata(filename)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/dicom/convert/<filename>', methods=['POST'])
@token_required
def convert_dicom_to_png(current_user, filename):
    try:
        window_center = request.json.get('windowCenter') if request.json else None
        window_width = request.json.get('windowWidth') if request.json else None
        output_filename = processor.convert_to_png(filename, window_center, window_width)
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/grayscale', methods=['POST'])
@token_required
def apply_grayscale(current_user):
    try:
        data = request.json
        output_filename = processor.apply_grayscale(data['filename'])
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/blur', methods=['POST'])
@token_required
def apply_blur(current_user):
    try:
        data = request.json
        output_filename = processor.apply_blur(data['filename'], data.get('kernelSize', 5))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/edge-detection', methods=['POST'])
@token_required
def apply_edge_detection(current_user):
    try:
        data = request.json
        output_filename = processor.apply_edge_detection(data['filename'], data.get('method', 'canny'))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/histogram', methods=['POST'])
@token_required
def get_histogram(current_user):
    try:
        data = request.json
        histogram_data = processor.calculate_histogram(data['filename'])
        return jsonify({'success': True, 'histogram': histogram_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/histogram-equalization', methods=['POST'])
@token_required
def apply_histogram_equalization(current_user):
    try:
        data = request.json
        output_filename = processor.apply_histogram_equalization(data['filename'])
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/filter', methods=['POST'])
@token_required
def apply_filter(current_user):
    try:
        data = request.json
        output_filename = processor.apply_filter(data['filename'], data.get('filterType', 'sharpen'))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/windowing', methods=['POST'])
@token_required
def apply_windowing(current_user):
    try:
        data = request.json
        output_filename = processor.apply_windowing(data['filename'], data.get('windowCenter', 40), data.get('windowWidth', 400))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/invert', methods=['POST'])
@token_required
def apply_invert(current_user):
    try:
        data = request.json
        output_filename = processor.apply_invert(data['filename'])
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/rotate', methods=['POST'])
@token_required
def apply_rotate(current_user):
    try:
        data = request.json
        output_filename = processor.apply_rotate(data['filename'], data.get('angle', 90))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/flip', methods=['POST'])
@token_required
def apply_flip(current_user):
    try:
        data = request.json
        output_filename = processor.apply_flip(data['filename'], data.get('direction', 'horizontal'))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/clahe', methods=['POST'])
@token_required
def apply_clahe(current_user):
    try:
        data = request.json
        output_filename = processor.apply_clahe(data['filename'], data.get('clipLimit', 2.0), data.get('tileSize', 8))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/brightness-contrast', methods=['POST'])
@token_required
def apply_brightness_contrast(current_user):
    try:
        data = request.json
        output_filename = processor.apply_brightness_contrast(data['filename'], data.get('brightness', 0), data.get('contrast', 1.0))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/gamma', methods=['POST'])
@token_required
def apply_gamma(current_user):
    try:
        data = request.json
        output_filename = processor.apply_gamma_correction(data['filename'], data.get('gamma', 1.0))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/denoise', methods=['POST'])
@token_required
def apply_denoise(current_user):
    try:
        data = request.json
        output_filename = processor.apply_denoise(data['filename'], data.get('strength', 10))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/process/unsharp-mask', methods=['POST'])
@token_required
def apply_unsharp_mask(current_user):
    try:
        data = request.json
        output_filename = processor.apply_unsharp_mask(data['filename'], data.get('sigma', 1.0), data.get('strength', 1.5))
        return jsonify({'success': True, 'outputFile': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ==================== INIT DB ====================
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

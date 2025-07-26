from flask import Flask, render_template, request, jsonify, make_response, send_file
import openai
import os
import logging
from dotenv import load_dotenv
import requests
import uuid
from flask_cors import CORS  # 添加 CORS 支持
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime

# 文件监控缓存
file_cache = {}

# 加载环境变量
load_dotenv()

# 创建日志配置
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB文件上传限制
CORS(app)  # 启用 CORS 支持

# 默认配置
DEFAULT_CONFIG = {
    "api_base": "https://api.deepseek.com/v1",  # DeepSeek 默认端点
    "api_key": "",
    "default_model": "deepseek-chat",  # DeepSeek 默认模型
    "timeout": 300.0,
    "retries": 3,      # 增加重试次数
    "backoff_factor": 0.5
}

# 初始化配置
app.config.update(DEFAULT_CONFIG)

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/api/config', methods=['POST'])
def save_config():
    """保存API配置"""
    try:
        config = request.json
        logger.info(f"收到配置请求: {config}")
        
        # 验证配置
        if not config.get('api_base') or not config.get('api_key'):
            return jsonify({
                "status": "error",
                "message": "API基础地址和API密钥是必需的"
            }), 400
        
        # 更新配置
        app.config['api_base'] = config.get('api_base', DEFAULT_CONFIG['api_base'])
        app.config['api_key'] = config.get('api_key', '')
        app.config['default_model'] = config.get('default_model', DEFAULT_CONFIG['default_model'])
        
        logger.info(f"配置已更新: {app.config}")
        return jsonify({
            "status": "success",
            "config": {
                "api_base": app.config['api_base'],
                "default_model": app.config['default_model']
            }
        })
        
    except Exception as e:
        logger.error(f"保存配置时出错: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"服务器错误: {str(e)}"
        }), 500

@app.route('/api/validate', methods=['GET'])
def validate_config():
    """验证API配置有效性"""
    try:
        logger.info("验证API配置...")
        
        # 检查必要配置
        if not app.config['api_key']:
            return jsonify({
                "status": "error",
                "message": "API密钥未配置"
            }), 400
        
        # DeepSeek 验证端点
        headers = {
            "Authorization": f"Bearer {app.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # 使用 DeepSeek 的验证端点
        response = requests.get(
            f"{app.config['api_base']}/models",
            headers=headers,
            timeout=app.config['timeout']
        )
        
        # 检查响应状态
        if response.status_code == 200:
            models = [model['id'] for model in response.json().get('data', [])]
            logger.info(f"API验证成功，找到 {len(models)} 个模型")
            return jsonify({
                "status": "success",
                "message": "API配置验证成功",
                "models": models
            })
        else:
            error_msg = f"API验证失败: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 400
            
    except Exception as e:
        error_msg = f"验证配置时出错: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.json
        message = data.get('message')
        model = data.get('model', app.config['default_model'])
        
        logger.info(f"收到聊天请求 - 模型: {model}, 消息: {message[:50]}...")
        
        if not message:
            return jsonify({"status": "error", "message": "消息内容不能为空"}), 400
        
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {app.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体 - 使用 DeepSeek 兼容格式
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个AI助手。"},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        # 发送请求到 DeepSeek API
        response = requests.post(
            f"{app.config['api_base']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=app.config['timeout']
        )
        
        # 检查响应
        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                ai_response = response_data['choices'][0]['message']['content']
                logger.info(f"成功生成响应: {ai_response[:50]}...")
                return jsonify({
                    "status": "success",
                    "response": ai_response
                })
            else:
                error_msg = "响应中没有找到有效内容"
                logger.error(error_msg)
                return jsonify({
                    "status": "error",
                    "message": error_msg
                }), 500
        else:
            error_msg = f"API请求失败: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg
            }), response.status_code
            
    except Exception as e:
        error_msg = f"处理聊天请求时出错: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

@app.route('/api/check_file_update', methods=['POST'])
def check_file_update():
    """检查文件是否更新"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        logger.info(f"检查文件更新: {file_path}")

        if not file_path:
            return jsonify({'updated': False, 'error': 'Path parameter is required'})

        # 处理相对路径和绝对路径
        if not os.path.isabs(file_path):
            # 相对路径，相对于应用根目录
            file_path = os.path.join(os.getcwd(), file_path)

        # 规范化路径
        file_path = os.path.normpath(file_path)
        logger.info(f"规范化后的文件路径: {file_path}")

        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return jsonify({'updated': False, 'error': 'File not found'})

        # 获取文件修改时间
        current_mtime = os.path.getmtime(file_path)

        # 检查缓存中的修改时间
        cached_mtime = file_cache.get(file_path, 0)

        if current_mtime > cached_mtime:
            file_cache[file_path] = current_mtime
            logger.info(f"文件已更新: {file_path}, 新修改时间: {current_mtime}")
            return jsonify({'updated': True, 'mtime': current_mtime})

        return jsonify({'updated': False, 'mtime': current_mtime})

    except Exception as e:
        logger.error(f"检查文件更新失败: {str(e)}")
        return jsonify({'updated': False, 'error': str(e)})

@app.route('/api/serve_image')
def serve_image():
    """提供图片文件服务"""
    try:
        file_path = request.args.get('path', '')
        logger.info(f"请求图片路径: {file_path}")

        if not file_path:
            logger.error("图片路径为空")
            return jsonify({'error': 'Path parameter is required'}), 400

        # 处理相对路径和绝对路径
        if not os.path.isabs(file_path):
            # 相对路径，相对于应用根目录
            file_path = os.path.join(os.getcwd(), file_path)

        # 规范化路径
        file_path = os.path.normpath(file_path)
        logger.info(f"规范化后的图片路径: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"图片文件不存在: {file_path}")
            return jsonify({'error': f'File not found: {file_path}'}), 404

        if not os.path.isfile(file_path):
            logger.error(f"路径不是文件: {file_path}")
            return jsonify({'error': 'Path is not a file'}), 400

        # 检查文件扩展名
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            logger.error(f"不支持的图片格式: {file_ext}")
            return jsonify({'error': f'Unsupported image format: {file_ext}'}), 400

        logger.info(f"成功提供图片: {file_path}")
        return send_file(file_path)

    except Exception as e:
        logger.error(f"提供图片服务失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """上传图片文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        # 检查文件类型
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': f'Unsupported file type: {file_ext}'})

        # 创建上传目录
        upload_dir = 'uploaded_images'
        os.makedirs(upload_dir, exist_ok=True)

        # 生成唯一文件名
        import uuid
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)

        # 保存文件
        file.save(file_path)

        # 获取文件信息
        file_size = os.path.getsize(file_path)

        # 获取图片尺寸
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception as e:
            logger.warning(f"无法获取图片尺寸: {e}")
            width, height = None, None

        logger.info(f"图片上传成功: {file_path}, 大小: {file_size} bytes")

        return jsonify({
            'success': True,
            'file_path': file_path,
            'file_name': file.filename,
            'file_size': file_size,
            'width': width,
            'height': height,
            'url': f'/api/serve_image?path={file_path}'
        })

    except Exception as e:
        logger.error(f"图片上传失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_test_image', methods=['POST'])
def update_test_image():
    """更新测试图片（用于演示监控功能）"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        color = data.get('color', 'red')

        if not file_path:
            return jsonify({'success': False, 'error': 'Path parameter is required'})

        # 处理相对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 创建新颜色的图片
        try:
            from PIL import Image
            img = Image.new('RGB', (200, 200), color=color)
            img.save(file_path)

            logger.info(f"测试图片已更新: {file_path}, 颜色: {color}")
            return jsonify({'success': True, 'message': f'图片已更新为{color}色'})

        except ImportError:
            return jsonify({'success': False, 'error': 'PIL not available'})

    except Exception as e:
        logger.error(f"更新测试图片失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    logger.info(f"启动服务在 {host}:{port}")
    app.run(host=host, port=port)

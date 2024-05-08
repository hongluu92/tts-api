# FastAPI Authentication APIs

### Installation & Configuration 
- Cài đặt MySQL Worckbench tạo 1 schema mới có tên là test(MYSQL_DB)
- Bên dưới sẽ là chi tiết kết nối mysql 
```bash
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DB=test
MYSQL_PORT=3306
```
Nếu muốn thay đổi tên người dùng, mật khẩu hoặc tên cơ sở dữ liệu, có thể sửa đổi nó trong tệp `.env` 
Cài đặt môi trường
- Mở terminal trong project
```bash
conda create -n tts_api python=3.9
conda activate tts_api
pip install -r requirements.txt
```

### Installation speexdsp (Python bindings of speexdsp noise suppression library)
Build
```bash
sudo apt install libspeexdsp-dev
sudo apt install swig
git clone https://github.com/TeaPoly/speexdsp-ns-python.git
cd speexdsp-ns-python
python setup.py install
```

### Create DB
Khởi chạy DB lần đầu tiên
```bash
alembic revision --autogenerate -m "create my table"
alembic upgrade head
```

### Building the Project
- We can start building our projects by running `python main.py`
- To stop the services you can press `Ctrl + C` - (Control + C)

# Accessing 
- FastAPI Application Status [http://localhost:8000](http://localhost:8000)
- API Documentation [http://localhost:8000/docs](http://localhost:8000/docs)

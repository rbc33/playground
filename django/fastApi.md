# 15 Problemas de FastAPI al Estilo LeetCode

Aquí tienes 15 problemas de FastAPI con sus soluciones, organizados en formato similar a LeetCode:

## 1. Implementación de Autenticación JWT

**Problema:** Implementa un sistema de autenticación JWT para una API FastAPI que permita el registro de usuarios, inicio de sesión y protección de rutas.

**Solución:**

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional

# Configuración
SECRET_KEY = "tu_clave_secreta_muy_segura"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Modelos
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Base de datos simulada
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}

# Utilidades
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Rutas
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]

@app.post("/register/")
async def register_user(username: str, email: str, full_name: str, password: str):
    if username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    hashed_password = get_password_hash(password)
    fake_users_db[username] = {
        "username": username,
        "email": email,
        "full_name": full_name,
        "hashed_password": hashed_password,
        "disabled": False
    }
    return {"message": "User registered successfully"}
```

## 2. Implementación de Validación de Datos Avanzada

**Problema:** Implementa validadores personalizados para un modelo Pydantic que valide un número de tarjeta de crédito, un correo electrónico con dominio específico y una contraseña segura.

**Solución:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, EmailStr, Field
import re
from typing import Optional
import luhn  # pip install luhn

app = FastAPI()

class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str
    credit_card: Optional[str] = None

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

    @validator('email')
    def email_domain(cls, v):
        allowed_domains = ['gmail.com', 'hotmail.com', 'yahoo.com', 'company.com']
        domain = v.split('@')[1]
        if domain not in allowed_domains:
            raise ValueError(f'Email domain must be one of {allowed_domains}')
        return v

    @validator('password')
    def password_strength(cls, v):
        """
        Validate that the password:
        - Is at least 8 characters long
        - Contains at least one uppercase letter
        - Contains at least one lowercase letter
        - Contains at least one digit
        - Contains at least one special character
        """
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')

        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')

        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')

        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')

        return v

    @validator('credit_card')
    def validate_credit_card(cls, v):
        if v is None:
            return v

        # Remove spaces and dashes
        v = v.replace(' ', '').replace('-', '')

        # Check if it's all digits
        if not v.isdigit():
            raise ValueError('Credit card must contain only digits')

        # Check length (most cards are between 13-19 digits)
        if not (13 <= len(v) <= 19):
            raise ValueError('Credit card must be between 13 and 19 digits')

        # Luhn algorithm check
        if not luhn.verify(v):
            raise ValueError('Invalid credit card number (failed Luhn check)')

        return v

@app.post("/register/")
async def register_user(user: UserRegistration):
    # In a real application, you would save the user to a database
    return {"message": "User registered successfully", "user": user.dict(exclude={"password", "credit_card"})}

@app.get("/")
async def root():
    return {"message": "Welcome to the API with advanced validation"}
```

## 3. Implementación de Rate Limiting

**Problema:** Implementa un sistema de limitación de tasa (rate limiting) para proteger tu API contra abusos, permitiendo un número máximo de solicitudes por usuario en un período de tiempo.

**Solución:**

```python
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta

app = FastAPI()

# Simulación de autenticación
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # En una aplicación real, verificarías el token JWT
    # Aquí simplemente asumimos que el token es el nombre de usuario
    return {"username": token}

# Implementación de rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, List[float]] = {}
        self.cleanup_interval = 60  # Limpiar historial cada 60 segundos
        self.last_cleanup = time.time()

    def is_rate_limited(self, user_id: str) -> Tuple[bool, int]:
        current_time = time.time()

        # Limpiar historial periódicamente
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests(current_time)
            self.last_cleanup = current_time

        # Inicializar historial para nuevo usuario
        if user_id not in self.request_history:
            self.request_history[user_id] = []

        # Filtrar solicitudes más antiguas que 1 minuto
        minute_ago = current_time - 60
        self.request_history[user_id] = [
            timestamp for timestamp in self.request_history[user_id]
            if timestamp > minute_ago
        ]

        # Verificar si se excede el límite
        if len(self.request_history[user_id]) >= self.requests_per_minute:
            # Calcular cuándo se liberará un espacio
            oldest_request = self.request_history[user_id][0]
            retry_after = int(oldest_request + 60 - current_time) + 1
            return True, retry_after

        # Registrar la solicitud actual
        self.request_history[user_id].append(current_time)
        return False, 0

    def _cleanup_old_requests(self, current_time: float):
        """Elimina solicitudes antiguas para liberar memoria"""
        minute_ago = current_time - 60
        for user_id in list(self.request_history.keys()):
            self.request_history[user_id] = [
                timestamp for timestamp in self.request_history[user_id]
                if timestamp > minute_ago
            ]
            # Eliminar usuarios sin solicitudes recientes
            if not self.request_history[user_id]:
                del self.request_history[user_id]

# Crear instancia del limitador
rate_limiter = RateLimiter(requests_per_minute=5)  # 5 solicitudes por minuto

# Middleware para aplicar rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Obtener IP para solicitudes no autenticadas
    client_ip = request.client.host

    # Intentar obtener usuario autenticado
    user_id = client_ip
    try:
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
            user_id = token  # En una app real, decodificarías el token
    except:
        pass

    # Aplicar rate limiting
    is_limited, retry_after = rate_limiter.is_rate_limited(user_id)
    if is_limited:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    # Procesar la solicitud normalmente
    response = await call_next(request)
    return response

# Rutas de ejemplo
@app.get("/public/")
async def public_route():
    return {"message": "This is a public endpoint with rate limiting by IP"}

@app.get("/protected/")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {current_user['username']}! This is a protected endpoint with rate limiting by user."}

# Ruta para simular autenticación
@app.post("/token")
async def get_token(username: str, password: str):
    # En una app real, verificarías las credenciales
    if username and password:
        return {"access_token": username, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Invalid credentials")
```

## 4. Implementación de Caché con Redis

**Problema:** Implementa un sistema de caché utilizando Redis para almacenar resultados de operaciones costosas en una API FastAPI.

**Solución:**

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis
import json
import time
import hashlib
from functools import wraps
from typing import Optional, List, Any, Callable

app = FastAPI()

# Configuración de Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Modelo de datos
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# Base de datos simulada
fake_items_db = {
    1: {"id": 1, "name": "Foo", "description": "This is Foo", "price": 50.2, "tax": 10.5},
    2: {"id": 2, "name": "Bar", "description": "This is Bar", "price": 62.0},
    3: {"id": 3, "name": "Baz", "description": "This is Baz", "price": 35.4, "tax": 5.2},
}

# Decorador para cachear respuestas
def cache_response(expire_time_seconds: int = 60):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Crear una clave única basada en la función y sus argumentos
            key_parts = [func.__name__]

            # Añadir argumentos a la clave
            for arg in args:
                if hasattr(arg, '__dict__'):
                    key_parts.append(str(arg.__dict__))
                else:
                    key_parts.append(str(arg))

            # Añadir kwargs ordenados a la clave
            for k, v in sorted(kwargs.items()):
                if hasattr(v, '__dict__'):
                    key_parts.append(f"{k}:{v.__dict__}")
                else:
                    key_parts.append(f"{k}:{v}")

            # Crear hash de la clave
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Intentar obtener de la caché
            cached_result = redis_client.get(cache_key)

            if cached_result:
                # Devolver resultado cacheado
                return json.loads(cached_result)

            # Ejecutar la función si no hay caché
            result = await func(*args, **kwargs)

            # Guardar resultado en caché
            redis_client.setex(
                cache_key,
                expire_time_seconds,
                json.dumps(result)
            )

            return result
        return wrapper
    return decorator

# Función que simula una operación costosa
def expensive_operation(item_id: int) -> dict:
    """Simula una operación costosa como un cálculo complejo o consulta a BD externa"""
    time.sleep(2)  # Simular operación que toma tiempo
    return fake_items_db.get(item_id)

# Rutas
@app.get("/items/{item_id}", response_model=Item)
@cache_response(expire_time_seconds=30)
async def read_item(item_id: int):
    """Obtiene un item por su ID, con caché de 30 segundos"""
    start_time = time.time()

    item = expensive_operation(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    processing_time = time.time() - start_time

    # Convertir a Item y luego a dict para añadir el tiempo de procesamiento
    result = Item(**item).dict()
    result["processing_time"] = processing_time

    return result

@app.get("/items/", response_model=List[Item])
@cache_response(expire_time_seconds=60)
async def read_items(skip: int = 0, limit: int = 10):
    """Obtiene una lista de items, con caché de 60 segundos"""
    start_time = time.time()

    # Simular operación costosa
    time.sleep(1)

    items = list(fake_items_db.values())[skip : skip + limit]

    processing_time = time.time() - start_time

    # Añadir tiempo de procesamiento a la respuesta
    return {
        "items": items,
        "processing_time": processing_time,
        "count": len(items),
        "skip": skip,
        "limit": limit
    }

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    """Crea un nuevo item y limpia la caché relacionada"""
    if item.id in fake_items_db:
        raise HTTPException(status_code=400, detail="Item already exists")

    fake_items_db[item.id] = item.dict()

    # Limpiar caché relacionada con items
    for key in redis_client.keys("read_items*"):
        redis_client.delete(key)

    return item

@app.delete("/cache/")
async def clear_cache():
    """Limpia toda la caché"""
    redis_client.flushdb()
    return {"message": "Cache cleared successfully"}

@app.get("/")
async def root():
    return {"message": "FastAPI with Redis Cache"}
```

## 5. Implementación de Websockets para Chat en Tiempo Real

**Problema:** Implementa un sistema de chat en tiempo real utilizando WebSockets en FastAPI.

**Solución:**

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import json

app = FastAPI()

# Modelos
class Message(BaseModel):
    sender: str
    content: str
    timestamp: datetime = datetime.now()
    room: str = "general"

# Autenticación simulada
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
fake_users_db = {
    "alice": {"username": "alice", "password": "secret"},
    "bob": {"username": "bob", "password": "secret"},
    "charlie": {"username": "charlie", "password": "secret"},
}

def get_current_user(token: str = Depends(oauth2_scheme)):
    if token not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return fake_users_db[token]

# Gestor de conexiones WebSocket
class ConnectionManager:
    def __init__(self):
        # Estructura: {room: {client_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Historial de mensajes por sala
        self.chat_history: Dict[str, List[Message]] = {}

    async def connect(self, websocket: WebSocket, client_id: str, room: str):
        await websocket.accept()

        # Inicializar sala si no existe
        if room not in self.active_connections:
            self.active_connections[room] = {}
            self.chat_history[room] = []

        # Guardar conexión
        self.active_connections[room][client_id] = websocket

        # Enviar historial de mensajes al nuevo cliente
        if room in self.chat_history:
            for message in self.chat_history[room][-50:]:  # Últimos 50 mensajes
                await websocket.send_text(json.dumps({
                    "sender": message.sender,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "room": message.room
                }))

    def disconnect(self, client_id: str, room: str):
        if room in self.active_connections and client_id in self.active_connections[room]:
            del self.active_connections[room][client_id]
            # Eliminar sala si está vacía
            if not self.active_connections[room]:
                del self.active_connections[room]

    async def broadcast(self, message: Message):
        room = message.room
        if room not in self.active_connections:
            return

        # Guardar mensaje en historial
        if room not in self.chat_history:
            self.chat_history[room] = []
        self.chat_history[room].append(message)

        # Limitar historial a 1000 mensajes por sala
        if len(self.chat_history[room]) > 1000:
            self.chat_history[room] = self.chat_history[room][-1000:]

        # Enviar mensaje a todos los clientes en la sala
        message_json = json.dumps({
            "sender": message.sender,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "room": message.room
        })

        for client_id, connection in self.active_connections[room].items():
            try:
                await connection.send_text(message_json)
            except:
                # Manejar conexiones cerradas
                self.disconnect(client_id, room)

    def get_active_users(self, room: str) -> List[str]:
        if room not in self.active_connections:
            return []
        return list(self.active_connections[room].keys())

# Crear instancia del gestor
manager = ConnectionManager()

# Rutas
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or form_data.password != user["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": user["username"], "token_type": "bearer"}

@app.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str, token: str):
    try:
        # Validar token
        if token not in fake_users_db:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        client_id = token
        await manager.connect(websocket, client_id, room)

        # Notificar a todos que un usuario se ha unido
        join_message = Message(
            sender="system",
            content=f"{client_id} has joined the chat",
            room=room
        )
        await manager.broadcast(join_message)

        try:
            while True:
                data = await websocket.receive_text()
                message = Message(
                    sender=client_id,
                    content=data,
                    room=room
                )
                await manager.broadcast(message)
        except WebSocketDisconnect:
            manager.disconnect(client_id, room)
            leave_message = Message(
                sender="system",
                content=f"{client_id} has left the chat",
                room=room
            )
            await manager.broadcast(leave_message)
    except Exception as e:
        print(f"Error: {e}")
        if websocket.client_state.CONNECTED:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

@app.get("/rooms/{room}/users")
async def get_active_users(room: str, user: dict = Depends(get_current_user)):
    return {"room": room, "active_users": manager.get_active_users(room)}

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>FastAPI Chat</title>
        </head>
        <body>
            <h1>FastAPI WebSocket Chat</h1>
            <form action="" onsubmit="connect(event)">
                <input type="text" id="token" placeholder="Your token" autocomplete="off"/>
                <input type="text" id="room" placeholder="Room name" value="general" autocomplete="off"/>
                <button>Connect</button>
            </form>
            <div id="status"></div>
            <ul id="messages"></ul>
            <form action="" onsubmit="sendMessage(event)">
                <input type="text" id="messageText" autocomplete="off"/>
                <button>Send</button>
            </form>
            <script>
                var ws = null;

                function connect(event) {
                    event.preventDefault();
                    const token = document.getElementById('token').value;
                    const room = document.getElementById('room').value;

                    if (ws) {
                        ws.close();
                    }

                    ws = new WebSocket(`ws://localhost:8000/ws/${room}?token=${token}`);

                    document.getElementById('status').innerHTML = 'Connecting...';

                    ws.onopen = function(event) {
                        document.getElementById('status').innerHTML = 'Connected to ' + room;
                    };

                    ws.onmessage = function(event) {
                        const message = JSON.parse(event.data);
                        const messages = document.getElementById('messages');
                        const messageItem = document.createElement('li');
                        const time = new Date(message.timestamp).toLocaleTimeString();
                        messageItem.innerHTML = `<strong>${message.sender}</strong> (${time}): ${message.content}`;
                        messages.appendChild(messageItem);
                    };

                    ws.onclose = function(event) {
                        document.getElementById('status').innerHTML = 'Disconnected';
                    };
                }

                function sendMessage(event) {
                    event.preventDefault();
                    const messageText = document.getElementById('messageText');
                    if (ws && messageText.value) {
                        ws.send(messageText.value);
                        messageText.value = '';
                    }
                }
            </script>
        </body>
    </html>
    """)
```

## 6. Implementación de Subida de Archivos con Validación

**Problema:** Implementa un sistema de subida de archivos con validación de tipo, tamaño y contenido, que almacene los archivos de forma segura.

**Solución:**

```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import shutil
import os
import uuid
import aiofiles
import magic
import hashlib
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

# Configuración
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {
    "image": ["image/jpeg", "image/png", "image/gif"],
    "document": ["application/pdf", "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    "video": ["video/mp4", "video/mpeg", "video/quicktime"]
}

# Crear directorio de subidas si no existe
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Modelos
class FileInfo(BaseModel):
    id: str
    filename: str
    original_filename: str
    content_type: str
    size: int
    upload_time: datetime
    checksum: str
    category: str

# Base de datos simulada para archivos
file_db = {}

# Funciones de utilidad
def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

def get_file_category(content_type: str) -> str:
    for category, types in ALLOWED_EXTENSIONS.items():
        if content_type in types:
            return category
    return "other"

def is_valid_file_type(content_type: str, allowed_types: Optional[List[str]] = None) -> bool:
    if allowed_types:
        return content_type in allowed_types

    # Si no se especifican tipos, verificar contra todos los tipos permitidos
    for types in ALLOWED_EXTENSIONS.values/fastapi_problems/problem6.py():
        if content_type in types:
            return True
    return False

def calculate_file_hash(file_path: str) -> str:
    """Calcula el hash SHA-256 de un archivo"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Rutas
@app.post("/upload/", response_model=FileInfo)
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form(None)
):
    # Validar tamaño del archivo
    file_size = 0
    contents = b""

    # Leer el archivo en chunks para evitar problemas de memoria
    async for chunk in file.file:
        contents += chunk
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)} MB"
            )

    # Validar tipo de archivo usando python-magic
    content_type = magic.from_buffer(contents, mime=True)

    # Verificar si el tipo de archivo está permitido
    if not is_valid_file_type(content_type):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}"
        )

    # Generar nombre de archivo único
    file_id = str(uuid.uuid4())
    extension = get_file_extension(file.filename)
    filename = f"{file_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Guardar el archivo
    async with aiofiles.open(file_path, "wb") as out_file:
        await out_file.write(contents)

    # Calcular hash del archivo
    file_hash = calculate_file_hash(file_path)

    # Determinar categoría
    if not category:
        category = get_file_category(content_type)

    # Crear y guardar información del archivo
    file_info = FileInfo(
        id=file_id,
        filename=filename,
        original_filename=file.filename,
        content_type=content_type,
        size=file_size,
        upload_time=datetime.now(),
        checksum=file_hash,
        category=category
    )

    # Guardar en la "base de datos"
    file_db[file_id] = file_info.dict()

    return file_info

@app.get("/files/")
async def list_files(category: Optional[str] = None):
    """Lista todos los archivos, opcionalmente filtrados por categoría"""
    if category:
        files = [file for file in file_db.values() if file["category"] == category]
    else:
        files = list(file_db.values())

    return {"files": files, "count": len(files)}

@app.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """Obtiene información de un archivo específico"""
    if file_id not in file_db:
        raise HTTPException(status_code=404, detail="File not found")

    return file_db[file_id]

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Descarga un archivo específico"""
    if file_id not in file_db:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_db[file_id]
    file_path = os.path.join(UPLOAD_DIR, file_info["filename"])

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=file_path,
        filename=file_info["original_filename"],
        media_type=file_info["content_type"]
    )

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Elimina un archivo específico"""
    if file_id not in file_db:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_db[file_id]
    file_path = os.path.join(UPLOAD_DIR, file_info["filename"])

    # Eliminar archivo del disco
    if os.path.exists(file_path):
        os.remove(file_path)

    # Eliminar de la "base de datos"
    del file_db[file_id]

    return {"message": "File deleted successfully"}

# Servir archivos estáticos (opcional, para una interfaz de usuario)
app.mount("/static", StaticFiles(directory="static"), name="static")
```

## 7. Implementación de Tareas en Segundo Plano con Celery

**Problema:** Implementa un sistema de tareas en segundo plano utilizando Celery para procesar operaciones largas sin bloquear la API.

**Solución:**

```python
# Archivo: main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict
import time
from datetime import datetime
import uuid
from celery import Celery
from celery.result import AsyncResult

# Configuración de Celery
# En un entorno real, usarías Redis o RabbitMQ como broker
celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

app = FastAPI()

# Autenticación simulada
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # En una aplicación real, verificarías el token JWT
    return {"username": token}

# Modelos
class EmailNotification(BaseModel):
    email: EmailStr
    subject: str
    body: str

class ReportRequest(BaseModel):
    report_type: str
    parameters: Dict
    email: Optional[EmailStr] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: datetime

# Base de datos simulada para tareas
tasks_db = {}

# Tareas de Celery
@celery_app.task(name="send_email")
def send_email_task(email: str, subject: str, body: str):
    """Tarea que simula el envío de un correo electrónico"""
    # En una aplicación real, usarías una biblioteca como smtplib
    print(f"Sending email to {email}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")

    # Simular tiempo de procesamiento
    time.sleep(5)

    return {"status": "sent", "email": email, "timestamp": datetime.now().isoformat()}

@celery_app.task(name="generate_report")
def generate_report_task(report_type: str, parameters: Dict, email: Optional[str] = None):
    """Tarea que simula la generación de un informe"""
    print(f"Generating {report_type} report with parameters: {parameters}")

    # Simular tiempo de procesamiento según el tipo de informe
    if report_type == "simple":
        time.sleep(10)
    elif report_type == "complex":
        time.sleep(30)
    else:
        time.sleep(5)

    # Generar URL de descarga simulada
    report_id = str(uuid.uuid4())
    download_url = f"http://example.com/reports/{report_id}"

    result = {
        "status": "completed",
        "report_type": report_type,
        "download_url": download_url,
        "timestamp": datetime.now().isoformat()
    }

    # Si se proporcionó un correo, enviar notificación
    if email:
        send_email_task.delay(
            email=email,
            subject=f"Your {report_type} report is ready",
            body=f"Your report is ready for download at {download_url}"
        )

    return result

@celery_app.task(name="process_data")
def process_data_task(data: List[Dict], operation: str):
    """Tarea que simula el procesamiento de datos"""
    print(f"Processing {len(data)} items with operation: {operation}")

    # Simular tiempo de procesamiento
    time.sleep(0.1 * len(data))

    # Simular resultados según la operación
    if operation == "sum":
        result = sum(item.get("value", 0) for item in data)
    elif operation == "average":
        values = [item.get("value", 0) for item in data]
        result = sum(values) / len(values) if values else 0
    else:
        result = len(data)

    return {
        "status": "completed",
        "operation": operation,
        "result": result,
        "items_processed": len(data),
        "timestamp": datetime.now().isoformat()
    }

# Rutas
@app.post("/tasks/email", response_model=TaskResponse)
async def send_email(
    notification: EmailNotification,
    current_user: dict = Depends(get_current_user)
):
    """Envía un correo electrónico en segundo plano"""
    # Iniciar tarea de Celery
    task = send_email_task.delay(
        email=notification.email,
        subject=notification.subject,
        body=notification.body
    )

    # Registrar tarea
    task_info = {
        "task_id": task.id,
        "status": "pending",
        "created_at": datetime.now(),
        "type": "email",
        "user": current_user["username"],
        "details": notification.dict()
    }
    tasks_db[task.id] = task_info

    return TaskResponse(
        task_id=task.id,
        status="pending",
        created_at=task_info["created_at"]
    )

@app.post("/tasks/report", response_model=TaskResponse)
async def generate_report(
    request: ReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """Genera un informe en segundo plano"""
    # Validar tipo de informe
    valid_report_types = ["simple", "complex", "custom"]
    if request.report_type not in valid_report_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type. Must be one of: {valid_report_types}"
        )

    # Iniciar tarea de Celery
    task = generate_report_task.delay(
        report_type=request.report_type,
        parameters=request.parameters,
        email=request.email
    )

    # Registrar tarea
    task_info = {
        "task_id": task.id,
        "status": "pending",
        "created_at": datetime.now(),
        "type": "report",
        "user": current_user["username"],
        "details": request.dict()
    }
    tasks_db[task.id] = task_info

    return TaskResponse(
        task_id=task.id,
        status="pending",
        created_at=task_info["created_at"]
    )

@app.post("/tasks/process-data", response_model=TaskResponse)
async def process_data(
    data: List[Dict],
    operation: str,
    current_user: dict = Depends(get_current_user)
):
    """Procesa datos en segundo plano"""
    # Validar operación
    valid_operations = ["sum", "average", "count"]
    if operation not in valid_operations:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid operation. Must be one of: {valid_operations}"
        )

    # Iniciar tarea de Celery
    task = process_data_task.delay(data=data, operation=operation)

    # Registrar tarea
    task_info = {
        "task_id": task.id,
        "status": "pending",
        "created_at": datetime.now(),
        "type": "data_processing",
        "user": current_user["username"],
        "details": {"operation": operation, "data_count": len(data)}
    }
    tasks_db[task.id] = task_info

    return TaskResponse(
        task_id=task.id,
        status="pending",
        created_at=task_info["created_at"]
    )

@app.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Obtiene el estado de una tarea"""
    # Verificar si la tarea existe en nuestra base de datos
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    # Obtener información de la tarea de Celery
    celery_task = AsyncResult(task_id)

    # Actualizar estado en nuestra base de datos
    tasks_db[task_id]["status"] = celery_task.status

    # Preparar respuesta
    response = {
        "task_id": task_id,
        "status": celery_task.status,
        "created_at": tasks_db[task_id]["created_at"],
        "type": tasks_db[task_id]["type"],
    }

    # Añadir resultado si está disponible
    if celery_task.ready():
        response["result"] = celery_task.result

    return response

@app.get("/tasks/")
async def list_tasks(
    current_user: dict = Depends(get_current_user),
    task_type: Optional[str] = None,
    status: Optional[str] = None
):
    """Lista todas las tareas, opcionalmente filtradas por tipo y estado"""
    # Filtrar tareas por usuario
    user_tasks = {
        task_id: task for task_id, task in tasks_db.items()
        if task["user"] == current_user["username"]
    }

    # Aplicar filtros adicionales
    if task_type:
        user_tasks = {
            task_id: task for task_id, task in user_tasks.items()
            if task["type"] == task_type
        }

    if status:
        # Actualizar estados desde Celery
        for task_id, task in user_tasks.items():
            celery_task = AsyncResult(task_id)
            task["status"] = celery_task.status

        user_tasks = {
            task_id: task for task_id, task in user_tasks.items()
            if task["status"] == status
        }

    return {"tasks": list(user_tasks.values()), "count": len(user_tasks)}

# Archivo: celery_worker.py (separado)
# from main import celery_app
#
# if __name__ == "__main__":
#     celery_app.worker_main(["worker", "--loglevel=info"])
```

## 8. Implementación de GraphQL con Strawberry

**Problema:** Implementa una API GraphQL utilizando Strawberry para consultar y modificar datos de productos y categorías.

**Solución:**

```python
from fastapi import FastAPI
from typing import List, Optional
import strawberry
from strawberry.fastapi import GraphQLRouter
from datetime import datetime
import uuid

# Modelos de datos (simulando una base de datos)
class CategoryModel:
    def __init__(self, id: str, name: str, description: Optional[str] = None):
        self.id = id
        self.name = name
        self.description = description

class ProductModel:
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        price: float,
        category_id: str,
        in_stock: bool = True,
        created_at: datetime = None
    ):
        self.id = id
        self.name = name
        self.description = description
        self.price = price
        self.category_id = category_id
        self.in_stock = in_stock
        self.created_at = created_at or datetime.now()

# Base de datos simulada
categories_db = {
    "cat1": CategoryModel(id="cat1", name="Electronics", description="Electronic devices and gadgets"),
    "cat2": CategoryModel(id="cat2", name="Books", description="Physical and digital books"),
    "cat3": CategoryModel(id="cat3", name="Clothing", description="Apparel and accessories")
}

products_db = {
    "prod1": ProductModel(
        id="prod1",
        name="Smartphone",
        description="Latest model smartphone",
        price=999.99,
        category_id="cat1"
    ),
    "prod2": ProductModel(
        id="prod2",
        name="Laptop",
        description="Powerful laptop for professionals",
        price=1499.99,
        category_id="cat1"
    ),
    "prod3": ProductModel(
        id="prod3",
        name="Python Programming",
        description="Learn Python programming",
        price=39.99,
        category_id="cat2"
    ),
    "prod4": ProductModel(
        id="prod4",
        name="T-shirt",
        description="Cotton t-shirt",
        price=19.99,
        category_id="cat3",
        in_stock=False
    )
}

# Tipos GraphQL
@strawberry.type
class Category:
    id: str
    name: str
    description: Optional[str] = None

    @strawberry.field
    def products(self) -> List["Product"]:
        return [
            Product.from_model(product)
            for product in products_db.values()
            if product.category_id == self.id
        ]

    @classmethod
    def from_model(cls, model: CategoryModel) -> "Category":
        return cls(
            id=model.id,
            name=model.name,
            description=model.description
        )

@strawberry.type
class Product:
    id: str
    name: str
    description: str
    price: float
    in_stock: bool
    created_at: datetime

    @strawberry.field
    def category(self) -> Category:
        category_model = categories_db.get(self.category_id)
        if category_model:
            return Category.from_model(category_model)
        return None

    @classmethod
    def from_model(cls, model: ProductModel) -> "Product":
        return cls(
            id=model.id,
            name=model.name,
            description=model.description,
            price=model.price,
            in_stock=model.in_stock,
            created_at=model.created_at,
            category_id=model.category_id
        )

    # Campo adicional para uso interno
    category_id: str = strawberry.private

# Inputs para mutaciones
@strawberry.input
class CategoryInput:
    name: str
    description: Optional[str] = None

@strawberry.input
class ProductInput:
    name: str
    description: str
    price: float
    category_id: str
    in_stock: bool = True

# Queries
@strawberry.type
class Query:
    @strawberry.field
    def category(self, id: str) -> Optional[Category]:
        category_model = categories_db.get(id)
        if category_model:
            return Category.from_model(category_model)
        return None

    @strawberry.field
    def categories(self) -> List[Category]:
        return [Category.from_model(cat) for cat in categories_db.values()]

    @strawberry.field
    def product(self, id: str) -> Optional[Product]:
        product_model = products_db.get(id)
        if product_model:
            return Product.from_model(product_model)
        return None

    @strawberry.field
    def products(
        self,
        category_id: Optional[str] = None,
        in_stock: Optional[bool] = None
    ) -> List[Product]:
        filtered_products = products_db.values()

        if category_id:
            filtered_products = [p for p in filtered_products if p.category_id == category_id]

        if in_stock is not None:
            filtered_products = [p for p in filtered_products if p.in_stock == in_stock]

        return [Product.from_model(product) for product in filtered_products]

    @strawberry.field
    def search_products(self, query: str) -> List[Product]:
        query = query.lower()
        matching_products = [
            product for product in products_db.values()
            if query in product.name.lower() or query in product.description.lower()
        ]
        return [Product.from_model(product) for product in matching_products]

# Mutations
@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_category(self, input: CategoryInput) -> Category:
        category_id = f"cat{str(uuid.uuid4())[:8]}"
        category = CategoryModel(
            id=category_id,
            name=input.name,
            description=input.description
        )
        categories_db[category_id] = category
        return Category.from_model(category)

    @strawberry.mutation
    def update_category(self, id: str, input: CategoryInput) -> Optional[Category]:
        if id not in categories_db:
            return None

        category = categories_db[id]
        category.name = input.name
        category.description = input.description

        return Category.from_model(category)

    @strawberry.mutation
    def delete_category(self, id: str) -> bool:
        if id not in categories_db:
            return False

        # Verificar si hay productos en esta categoría
        products_in_category = [p for p in products_db.values() if p.category_id == id]
        if products_in_category:
            return False

        del categories_db[id]
        return True

    @strawberry.mutation
    def create_product(self, input: ProductInput) -> Optional[Product]:
        # Verificar si la categoría existe
        if input.category_id not in categories_db:
            return None

        product_id = f"prod{str(uuid.uuid4())[:8]}"
        product = ProductModel(
            id=product_id,
            name=input.name,
            description=input.description,
            price=input.price,
            category_id=input.category_id,
            in_stock=input.in_stock,
            created_at=datetime.now()
        )
        products_db[product_id] = product
        return Product.from_model(product)

    @strawberry.mutation
    def update_product(self, id: str, input: ProductInput) -> Optional[Product]:
        if id not in products_db:
            return None

        # Verificar si la categoría existe
        if input.category_id not in categories_db:
            return None

        product = products_db[id]
        product.name = input.name
        product.description = input.description
        product.price = input.price
        product.category_id = input.category_id
        product.in_stock = input.in_stock

        return Product.from_model(product)

    @strawberry.mutation
    def delete_product(self, id: str) -> bool:
        if id not in products_db:
            return False

        del products_db[id]
        return True

    @strawberry.mutation
    def update_product_stock(self, id: str, in_stock: bool) -> Optional[Product]:
        if id not in products_db:
            return None

        product = products_db[id]
        product.in_stock = in_stock

        return Product.from_model(product)

# Crear esquema GraphQL
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Crear router GraphQL
graphql_app = GraphQLRouter(schema)

# Crear aplicación FastAPI
app = FastAPI()

# Añadir router GraphQL
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
async def root():
    return {"message": "Navigate to /graphql for the GraphQL interface"}
```

## 9. Implementación de Autenticación Multi-Factor

**Problema:** Implementa un sistema de autenticación multi-factor que combine contraseñas con códigos de un solo uso (TOTP) para una API FastAPI.

**Solución:**

```python
from fastapi import FastAPI, Depends, HTTPException, status, Form, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid
import os

app = FastAPI()

# Configuración
SECRET_KEY = "your-secret-key-for-jwt"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MFA_REQUIRED_ROLES = ["admin", "finance"]  # Roles que requieren MFA

# Configuración de plantillas
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modelos
class User(BaseModel):
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = []
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    requires_mfa: bool = False

class TokenData(BaseModel):
    username: Optional[str] = None
    mfa_verified: bool = False

class MFASetup(BaseModel):
    secret: str
    uri: str
    qr_code: str

# Base de datos simulada
fake_users_db = {
    "johndoe": {
        "id": "user1",
        "username": "johndoe",
        "email": "johndoe@example.com",
        "full_name": "John Doe",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "roles": ["user"],
        "mfa_enabled": False,
        "mfa_secret": None
    },
    "alice": {
        "id": "user2",
        "username": "alice",
        "email": "alice@example.com",
        "full_name": "Alice Wonderland",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "roles": ["admin"],
        "mfa_enabled": True,
        "mfa_secret": "JBSWY3DPEHPK3PXP"  # Ejemplo de secreto TOTP
    }
}

# Utilidades
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_mfa_secret():
    """Genera un secreto para TOTP"""
    return pyotp.random_base32()

def generate_mfa_qr_code(username: str, secret: str):
    """Genera un código QR para configurar TOTP"""
    uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=username,
        issuer_name="FastAPI MFA Demo"
    )

    # Generar código QR
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convertir imagen a base64
    buffered = io.BytesIO()
    img.save(buffered)
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "secret": secret,
        "uri": uri,
        "qr_code": f"data:image/png;base64,{img_str}"
    }

def verify_totp(secret: str, token: str):
    """Verifica un token TOTP"""
    totp = pyotp.TOTP(secret)
    return totp.verify(token)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(
            username=username,
            mfa_verified=payload.get("mfa_verified", False)
        )
    except JWTError:
        raise credentials_exception

    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception

    # Verificar si el usuario tiene un rol que requiere MFA
    requires_mfa = any(role in MFA_REQUIRED_ROLES for role in user.roles)

    # Si el usuario requiere MFA pero no está verificado, denegar acceso
    if requires_mfa and user.mfa_enabled and not token_data.mfa_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="MFA verification required",
            headers={"X-MFA-Required": "true"},
        )

    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Rutas
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verificar si el usuario requiere MFA
    requires_mfa = any(role in MFA_REQUIRED_ROLES for role in user.roles)

    # Crear token con información de MFA
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "mfa_verified": False},
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        requires_mfa=requires_mfa and user.mfa_enabled
    )

@app.post("/verify-mfa")
async def verify_mfa_token(
    token: str = Form(...),
    current_token: str = Depends(oauth2_scheme)
):
    try:
        # Decodificar token actual
        payload = jwt.decode(current_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        # Obtener usuario
        user = get_user(fake_users_db, username)
        if not user or not user.mfa_secret:
            raise HTTPException(status_code=400, detail="Invalid user or MFA not set up")

        # Verificar token TOTP
        if not verify_totp(user.mfa_secret, token):
            raise HTTPException(status_code=400, detail="Invalid MFA token")

        # Crear nuevo token con MFA verificado
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "mfa_verified": True},
            expires_delta=access_token_expires,
        )

        return {"access_token": access_token, "token_type": "bearer"}

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/setup-mfa", response_model=MFASetup)
async def setup_mfa(current_user: User = Depends(get_current_active_user)):
    # Generar secreto TOTP
    secret = generate_mfa_secret()

    # Generar código QR
    mfa_setup = generate_mfa_qr_code(current_user.username, secret)

    # En una aplicación real, guardarías el secreto temporalmente hasta que el usuario lo verifique
    # Aquí lo guardamos directamente para simplificar
    fake_users_db[current_user.username]["mfa_secret"] = secret

    return MFASetup(**mfa_setup)

@app.post("/enable-mfa")
async def enable_mfa(
    token: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    # Verificar que el usuario tenga un secreto MFA
    if not current_user.mfa_secret:
        raise HTTPException(status_code=400, detail="MFA not set up")

    # Verificar token TOTP
    if not verify_totp(current_user.mfa_secret, token):
        raise HTTPException(status_code=400, detail="Invalid MFA token")

    # Habilitar MFA para el usuario
    fake_users_db[current_user.username]["mfa_enabled"] = True

    return {"message": "MFA enabled successfully"}

@app.post("/disable-mfa")
async def disable_mfa(
    token: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    # Verificar que el usuario tenga MFA habilitado
    if not current_user.mfa_enabled:
        raise HTTPException(status_code=400, detail="MFA not enabled")

    # Verificar token TOTP
    if not verify_totp(current_user.mfa_secret, token):
        raise HTTPException(status_code=400, detail="Invalid MFA token")

    # Deshabilitar MFA para el usuario
    fake_users_db[current_user.username]["mfa_enabled"] = False

    return {"message": "MFA disabled successfully"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/")
async def root():
    return {"message": "FastAPI MFA Demo"}

@app.get("/login-page", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/mfa-page", response_class=HTMLResponse)
async def get_mfa_page(request: Request):
    return templates.TemplateResponse("mfa.html", {"request": request})

@app.get("/setup-mfa-page", response_class=HTMLResponse)
async def get_setup_mfa_page(request: Request, current_user: User = Depends(get_current_active_user)):
    return templates.TemplateResponse("setup_mfa.html", {"request": request, "user": current_user})

# Archivo: templates/login.html
"""
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input {
            width: 100%;
            padding: 8px;
            box-sizing: border-box;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Login</h1>
    <div id="error" class="error"></div>
    <form id="loginForm">
        <div class="form-group">
            <label for="username">Username:</label>
            <input type="text" id="username" name="username" required>
        </div>
        <div class="form-group">
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" required>
        </div>
        <button type="submit">Login</button>
    </form>

    <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;

            try {
                const response = await fetch('/token', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
                });

                const data = await response.json();

                if (response.ok) {
                    // Guardar token
                    localStorage.setItem('access_token', data.access_token);

                    // Verificar si se requiere MFA
                    if (data.requires_mfa) {
                        window.location.href = '/mfa-page';
                    } else {
                        window.location.href = '/users/me';
                    }
                } else {
                    document.getElementById('error').textContent = data.detail || 'Login failed';
                }
            } catch (error) {
                document.getElementById('error').textContent = 'An error occurred';
                console.error(error);
            }
        });
    </script>
</body>
</html>
"""

# Archivo: templates/mfa.html
"""
<!DOCTYPE html>
<html>
<head>
    <title>MFA Verification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input {
            width: 100%;
            padding: 8px;
            box-sizing: border-box;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>MFA Verification</h1>
    <p>Please enter the verification code from your authenticator app.</p>
    <div id="error" class="error"></div>
    <form id="mfaForm">
        <div class="form-group">
            <label for="token">Verification Code:</label>
            <input type="text" id="token" name="token" required pattern="[0-9]{6}" maxlength="6">
        </div>
        <button type="submit">Verify</button>
    </form>

    <script>
        document.getElementById('mfaForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const token = document.getElementById('token').value;
            const accessToken = localStorage.getItem('access_token');

            if (!accessToken) {
                window.location.href = '/login-page';
                return;
            }

            try {
                const response = await fetch('/verify-mfa', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Authorization': `Bearer ${accessToken}`
                    },
                    body: `token=${encodeURIComponent(token)}`,
                });

                const data = await response.json();

                if (response.ok) {
                    // Actualizar token con MFA verificado
                    localStorage.setItem('access_token', data.access_token);
                    window.location.href = '/users/me';
                } else {
                    document.getElementById('error').textContent = data.detail || 'Verification failed';
                }
            } catch (error) {
                document.getElementById('error').textContent = 'An error occurred';
                console.error(error);
            }
        });
    </script>
</body>
</html>
"""

# Archivo: templates/setup_mfa.html
"""
<!DOCTYPE html>
<html>
<head>
    <title>Setup MFA</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input {
            width: 100%;
            padding: 8px;
            box-sizing: border-box;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
        .qr-code {
            margin: 20px 0;
            text-align: center;
        }
        .secret {
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <h1>Setup MFA</h1>
    <p>Scan the QR code with your authenticator app (like Google Authenticator, Authy, etc.)</p>

    <div id="qrCode" class="qr-code">Loading...</div>

    <p>Or enter this secret key manually:</p>
    <div id="secret" class="secret">Loading...</div>

    <div id="error" class="error"></div>

    <form id="verifyForm" style="display: none;">
        <div class="form-group">
            <label for="token">Verification Code:</label>
            <input type="text" id="token" name="token" required pattern="[0-9]{6}" maxlength="6">
        </div>
        <button type="submit">Verify and Enable MFA</button>
    </form>

    <script>
        const accessToken = localStorage.getItem('access_token');

        if (!accessToken) {
            window.location.href = '/login-page';
        }

        // Obtener configuración MFA
        async function setupMFA() {
            try {
                const response = await fetch('/setup-mfa', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${accessToken}`
                    }
                });

                if (response.ok) {
                    const data = await response.json();

                    // Mostrar código QR
                    document.getElementById('qrCode').innerHTML = `<img src="${data.qr_code}" alt="QR Code">`;

                    // Mostrar secreto
                    document.getElementById('secret').textContent = data.secret;

                    // Mostrar formulario de verificación
                    document.getElementById('verifyForm').style.display = 'block';
                } else {
                    const errorData = await response.json();
                    document.getElementById('error').textContent = errorData.detail || 'Failed to setup MFA';
                }
            } catch (error) {
                document.getElementById('error').textContent = 'An error occurred';
                console.error(error);
            }
        }

        // Verificar y habilitar MFA
        document.getElementById('verifyForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const token = document.getElementById('token').value;

            try {
                const response = await fetch('/enable-mfa', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Authorization': `Bearer ${accessToken}`
                    },
                    body: `token=${encodeURIComponent(token)}`,
                });

                const data = await response.json();

                if (response.ok) {
                    alert('MFA enabled successfully!');
                    window.location.href = '/users/me';
                } else {
                    document.getElementById('error').textContent = data.detail || 'Verification failed';
                }
            } catch (error) {
                document.getElementById('error').textContent = 'An error occurred';
                console.error(error);
            }
        });

        // Iniciar configuración
        setupMFA();
    </script>
</body>
</html>
"""
```

## 10. Implementación de Microservicios con FastAPI

**Problema:** Implementa un sistema de microservicios utilizando FastAPI, donde diferentes servicios se comunican entre sí para proporcionar una funcionalidad completa.

**Solución:**

Vamos a crear tres microservicios:

1. Servicio de Usuarios
2. Servicio de Productos
3. Servicio de Pedidos (que se comunica con los otros dos)

I'll continue with the implementation of the microservices system for Problem 10. Let me complete the implementation of the Orders service and add an API Gateway to coordinate between the services.

First, let's finish the Orders service implementation:

```python
# Servicio de Pedidos (orders_service.py)
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid
from datetime import datetime
import httpx
import asyncio

app = FastAPI(title="Orders Service", description="Microservicio de gestión de pedidos")

# Configuración
USERS_SERVICE_URL = "http://localhost:8001"
PRODUCTS_SERVICE_URL = "http://localhost:8002"

# Autenticación
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{USERS_SERVICE_URL}/token")

# Modelos
class OrderItemCreate(BaseModel):
    product_id: str
    quantity: int = Field(..., gt=0)

class OrderItem(OrderItemCreate):
    id: str
    product_name: str
    unit_price: float
    subtotal: float

class OrderCreate(BaseModel):
    items: List[OrderItemCreate]
    shipping_address: str

class OrderStatus(str):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Order(BaseModel):
    id: str
    user_id: str
    items: List[OrderItem]
    total: float
    status: str
    shipping_address: str
    created_at: datetime
    updated_at: datetime

# Base de datos simulada
orders_db: Dict[str, Dict] = {}

# Cliente HTTP para comunicación entre servicios
async def get_http_client():
    async with httpx.AsyncClient() as client:
        yield client

# Dependencias
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    client: httpx.AsyncClient = Depends(get_http_client)
):
    try:
        # Extraer username del token (simplificado)
        # En una implementación real, decodificarías el JWT
        payload = jwt.decode(token, "users_service_secret_key", algorithms=["HS256"])
        username = payload.get("sub")

        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validar usuario con el servicio de usuarios
        response = await client.get(
            f"{USERS_SERVICE_URL}/users/validate/{username}",
            params={"token": token}
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_data = response.json()
        return {"username": username, "user_id": user_data["user_id"]}

    except (jwt.JWTError, httpx.RequestError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Funciones auxiliares
async def get_product_details(
    product_id: str,
    client: httpx.AsyncClient
):
    """Obtiene detalles de un producto desde el servicio de productos"""
    try:
        response = await client.get(f"{PRODUCTS_SERVICE_URL}/products/{product_id}")

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

        return response.json()

    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Products service unavailable")

async def check_product_stock(
    product_id: str,
    quantity: int,
    client: httpx.AsyncClient
):
    """Verifica si hay suficiente stock de un producto"""
    try:
        response = await client.get(
            f"{PRODUCTS_SERVICE_URL}/products/{product_id}/check-stock",
            params={"quantity": quantity}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

        stock_info = response.json()

        if not stock_info["available"]:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient stock for product {product_id}. Available: {stock_info['current_stock']}, Requested: {quantity}"
            )

        return stock_info

    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Products service unavailable")

async def update_product_stock(
    product_id: str,
    new_stock: int,
    client: httpx.AsyncClient
):
    """Actualiza el stock de un producto"""
    try:
        response = await client.put(
            f"{PRODUCTS_SERVICE_URL}/products/{product_id}/stock",
            params={"stock": new_stock}
        )

        if response.status_code != 200:
            print(f"Failed to update stock for product {product_id}: {response.text}")
            return False

        return True

    except httpx.RequestError as e:
        print(f"Error updating stock for product {product_id}: {str(e)}")
        return False

# Rutas
@app.post("/orders/", response_model=Order)
async def create_order(
    order_create: OrderCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client)
):
    # Verificar stock y obtener detalles de productos
    order_items = []
    total = 0.0
    stock_updates = []

    for item in order_create.items:
        # Verificar stock
        stock_info = await check_product_stock(item.product_id, item.quantity, client)

        # Obtener detalles del producto
        product = await get_product_details(item.product_id, client)

        # Calcular subtotal
        subtotal = product["price"] * item.quantity

        # Añadir a la lista de items
        order_items.append(
            OrderItem(
                id=str(uuid.uuid4()),
                product_id=item.product_id,
                product_name=product["name"],
                quantity=item.quantity,
                unit_price=product["price"],
                subtotal=subtotal
            )
        )

        # Actualizar total
        total += subtotal

        # Preparar actualización de stock
        new_stock = stock_info["current_stock"] - item.quantity
        stock_updates.append((item.product_id, new_stock))

    # Crear pedido
    order_id = str(uuid.uuid4())
    now = datetime.now()

    order_dict = {
        "id": order_id,
        "user_id": current_user["user_id"],
        "items": [item.dict() for item in order_items],
        "total": total,
        "status": OrderStatus.PENDING,
        "shipping_address": order_create.shipping_address,
        "created_at": now,
        "updated_at": now
    }

    orders_db[order_id] = order_dict

    # Actualizar stock en segundo plano
    async def update_stocks():
        async with httpx.AsyncClient() as bg_client:
            for product_id, new_stock in stock_updates:
                await update_product_stock(product_id, new_stock, bg_client)

    background_tasks.add_task(update_stocks)

    return Order(**order_dict)

@app.get("/orders/", response_model=List[Order])
async def read_orders(current_user: dict = Depends(get_current_user)):
    # Filtrar pedidos por usuario
    user_orders = [
        Order(**order) for order in orders_db.values()
        if order["user_id"] == current_user["user_id"]
    ]

    return user_orders

@app.get("/orders/{order_id}", response_model=Order)
async def read_order(
    order_id: str,
    current_user: dict = Depends(get_current_user)
):
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")

    order = orders_db[order_id]

    # Verificar que el pedido pertenece al usuario
    if order["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this order")

    return Order(**order)

@app.put("/orders/{order_id}/status")
async def update_order_status(
    order_id: str,
    status: str,
    current_user: dict = Depends(get_current_user)
):
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")

    order = orders_db[order_id]

    # Verificar que el pedido pertenece al usuario
    if order["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this order")

    # Validar estado
    valid_statuses = [
        OrderStatus.PENDING,
        OrderStatus.PROCESSING,
        OrderStatus.SHIPPED,
        OrderStatus.DELIVERED,
        OrderStatus.CANCELLED
    ]

    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    # Actualizar estado
    order["status"] = status
    order["updated_at"] = datetime.now()

    return {"message": f"Order status updated to {status}"}

@app.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client)
):
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")

    order = orders_db[order_id]

    # Verificar que el pedido pertenece al usuario
    if order["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this order")

    # Solo permitir cancelar pedidos pendientes o en procesamiento
    if order["status"] not in [OrderStatus.PENDING, OrderStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel order with status {order['status']}"
        )

    # Actualizar estado
    order["status"] = OrderStatus.CANCELLED
    order["updated_at"] = datetime.now()

    # Restaurar stock en segundo plano
    async def restore_stocks():
        async with httpx.AsyncClient() as bg_client:
            for item in order["items"]:
                # Obtener stock actual
                try:
                    product_response = await bg_client.get(f"{PRODUCTS_SERVICE_URL}/products/{item['product_id']}")
                    if product_response.status_code == 200:
                        product = product_response.json()
                        new_stock = product["stock"] + item["quantity"]
                        await update_product_stock(item["product_id"], new_stock, bg_client)
                except Exception as e:
                    print(f"Error restoring stock: {str(e)}")

    background_tasks.add_task(restore_stocks)

    return {"message": "Order cancelled successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
```

Now, let's create an API Gateway to coordinate between the services:

```python
# API Gateway (gateway.py)
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import Dict, Any, Optional
import time

app = FastAPI(title="API Gateway", description="Gateway para el sistema de microservicios")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de servicios
SERVICES = {
    "users": "http://localhost:8001",
    "products": "http://localhost:8002",
    "orders": "http://localhost:8003"
}

# Cliente HTTP
async def get_http_client():
    async with httpx.AsyncClient() as client:
        yield client

# Middleware para logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Procesar la solicitud
    response = await call_next(request)

    # Calcular tiempo de procesamiento
    process_time = time.time() - start_time

    # Log
    print(f"Path: {request.url.path}, Method: {request.method}, Time: {process_time:.4f}s, Status: {response.status_code}")

    return response

# Función para enrutar solicitudes
async def route_request(
    service: str,
    path: str,
    request: Request,
    client: httpx.AsyncClient
):
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")

    # Construir URL de destino
    target_url = f"{SERVICES[service]}{path}"

    # Obtener método HTTP
    method = request.method.lower()

    # Obtener headers (excluyendo host)
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    # Obtener parámetros de consulta
    params = dict(request.query_params)

    # Obtener cuerpo de la solicitud
    body = await request.body()

    try:
        # Realizar solicitud al servicio
        response = await getattr(client, method)(
            target_url,
            headers=headers,
            params=params,
            content=body if body else None,
            timeout=30.0  # Timeout de 30 segundos
        )

        # Construir respuesta
        return JSONResponse(
            content=response.json() if response.content else {},
            status_code=response.status_code,
            headers=dict(response.headers)
        )

    except httpx.RequestError as e:
        # Manejar errores de conexión
        print(f"Error connecting to {service} service: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service '{service}' is unavailable")

# Rutas para cada servicio
@app.api_route("/users{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def users_route(
    request: Request,
    path: str,
    client: httpx.AsyncClient = Depends(get_http_client)
):
    return await route_request("users", path, request, client)

@app.api_route("/products{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def products_route(
    request: Request,
    path: str,
    client: httpx.AsyncClient = Depends(get_http_client)
):
    return await route_request("products", path, request, client)

@app.api_route("/orders{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def orders_route(
    request: Request,
    path: str,
    client: httpx.AsyncClient = Depends(get_http_client)
):
    return await route_request("orders", path, request, client)

# Ruta para verificar estado de los servicios
@app.get("/health")
async def health_check(client: httpx.AsyncClient = Depends(get_http_client)):
    results = {}

    for service_name, service_url in SERVICES.items():
        try:
            # Intentar conectar al servicio
            start_time = time.time()
            response = await client.get(f"{service_url}/", timeout=5.0)
            response_time = time.time() - start_time

            results[service_name] = {
                "status": "up" if response.status_code < 500 else "error",
                "response_time": f"{response_time:.4f}s",
                "status_code": response.status_code
            }

        except httpx.RequestError:
            results[service_name] = {
                "status": "down",
                "error": "Connection error"
            }

    # Estado general
    overall_status = "healthy" if all(r["status"] == "up" for r in results.values()) else "degraded"

    return {
        "status": overall_status,
        "timestamp": time.time(),
        "services": results
    }

@app.get("/")
async def root():
    return {
        "message": "API Gateway for Microservices",
        "services": list(SERVICES.keys()),
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Finally, let's update the orchestrator to include the API Gateway:

```python
# Orchestrator para ejecutar todos los servicios
import subprocess
import time
import signal
import sys
import os

# Configuración
services = [
    {
        "name": "Users Service",
        "file": "problem10_users.py",
        "port": 8001
    },
    {
        "name": "Products Service",
        "file": "problem10_products.py",
        "port": 8002
    },
    {
        "name": "Orders Service",
        "file": "problem10_orders.py",
        "port": 8003
    },
    {
        "name": "API Gateway",
        "file": "problem10_gateway.py",
        "port": 8000
    }
]

processes = []

def start_services():
    """Inicia todos los servicios"""
    for service in services:
        print(f"Starting {service['name']} on port {service['port']}...")
        process = subprocess.Popen(
            ["python", service["file"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(process)
        print(f"{service['name']} started with PID {process.pid}")
        # Esperar un poco para que el servicio se inicie
        time.sleep(2)

def stop_services():
    """Detiene todos los servicios"""
    for i, process in enumerate(processes):
        service = services[i]
        print(f"Stopping {service['name']} (PID {process.pid})...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print(f"{service['name']} stopped")
        except subprocess.TimeoutExpired:
            print(f"Forcing {service['name']} to stop...")
            process.kill()

def signal_handler(sig, frame):
    """Maneja señales de interrupción"""
    print("\nShutting down all services...")
    stop_services()
    sys.exit(0)

if __name__ == "__main__":
    # Registrar manejador de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Iniciar servicios
        start_services()

        print("\nAll services are running.")
        print("API Gateway: http://localhost:8000")
        print("Users Service: http://localhost:8001")
        print("Products Service: http://localhost:8002")
        print("Orders Service: http://localhost:8003")
        print("\nPress Ctrl+C to stop all services.")

        # Mantener el script en ejecución
        while True:
            time.sleep(1)

            # Verificar si algún servicio ha terminado
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    service = services[i]
                    stdout, stderr = process.communicate()
                    print(f"\n{service['name']} has stopped unexpectedly!")
                    if stdout:
                        print(f"STDOUT: {stdout}")
                    if stderr:
                        print(f"STDERR: {stderr}")

                    # Reiniciar el servicio
                    print(f"Restarting {service['name']}...")
                    process = subprocess.Popen(
                        ["python", service["file"]],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    processes[i] = process
                    print(f"{service['name']} restarted with PID {process.pid}")

    except Exception as e:
        print(f"Error: {str(e)}")
        stop_services()
        sys.exit(1)
```

Let's also create a simple client script to demonstrate how to use the microservices system:

```python
# Cliente de ejemplo para el sistema de microservicios
import httpx
import asyncio
import json
from typing import Dict, List, Optional
import sys

# Configuración
GATEWAY_URL = "http://localhost:8000"

async def login(username: str, password: str) -> Optional[str]:
    """Inicia sesión y obtiene un token de acceso"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GATEWAY_URL}/users/token",
                data={"username": username, "password": password}
            )

            if response.status_code == 200:
                data = response.json()
                return data["access_token"]
            else:
                print(f"Error de autenticación: {response.status_code}")
                print(response.text)
                return None

    except httpx.RequestError as e:
        print(f"Error de conexión: {str(e)}")
        return None

async def get_categories(token: str) -> List[Dict]:
    """Obtiene todas las categorías"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GATEWAY_URL}/products/categories/",
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error al obtener categorías: {response.status_code}")
            print(response.text)
            return []

async def get_products(token: str, category_id: Optional[str] = None) -> List[Dict]:
    """Obtiene productos, opcionalmente filtrados por categoría"""
    params = {}
    if category_id:
        params["category_id"] = category_id

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GATEWAY_URL}/products/products/",
            params=params,
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error al obtener productos: {response.status_code}")
            print(response.text)
            return []

async def create_order(token: str, items: List[Dict], shipping_address: str) -> Optional[Dict]:
    """Crea un nuevo pedido"""
    order_data = {
        "items": items,
        "shipping_address": shipping_address
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{GATEWAY_URL}/orders/orders/",
            json=order_data,
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error al crear pedido: {response.status_code}")
            print(response.text)
            return None

async def get_orders(token: str) -> List[Dict]:
    """Obtiene todos los pedidos del usuario"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GATEWAY_URL}/orders/orders/",
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error al obtener pedidos: {response.status_code}")
            print(response.text)
            return []

async def main():
    """Función principal"""
    print("=== Cliente de Microservicios ===")

    # Iniciar sesión
    username = input("Usuario (john/jane): ") or "john"
    password = input("Contraseña (password): ") or "password"

    token = await login(username, password)

    if not token:
        print("No se pudo iniciar sesión. Saliendo...")
        return

    print("\n=== Menú Principal ===")
    print("1. Ver categorías")
    print("2. Ver productos")
    print("3. Crear pedido")
    print("4. Ver mis pedidos")
    print("5. Salir")

    option = input("\nSeleccione una opción: ")

    if option == "1":
        # Ver categorías
        categories = await get_categories(token)
        print("\n=== Categorías ===")
        for category in categories:
            print(f"ID: {category['id']}")
            print(f"Nombre: {category['name']}")
            print(f"Descripción: {category['description']}")
            print("-" * 30)

    elif option == "2":
        # Ver productos
        category_id = input("ID de categoría (opcional): ")
        products = await get_products(token, category_id)

        print("\n=== Productos ===")
        for product in products:
            print(f"ID: {product['id']}")
            print(f"Nombre: {product['name']}")
            print(f"Precio: ${product['price']}")
            print(f"Stock: {product['stock']}")
            print("-" * 30)

    elif option == "3":
        # Crear pedido
        products = await get_products(token)

        print("\n=== Productos Disponibles ===")
        for i, product in enumerate(products):
            print(f"{i+1}. {product['name']} - ${product['price']} (Stock: {product['stock']})")

        items = []
        while True:
            product_idx = input("\nSeleccione un producto (número) o 'fin' para terminar: ")

            if product_idx.lower() == "fin":
                break

            try:
                idx = int(product_idx) - 1
                if 0 <= idx < len(products):
                    quantity = int(input(f"Cantidad (máx {products[idx]['stock']}): "))

                    if 0 < quantity <= products[idx]['stock']:
                        items.append({
                            "product_id": products[idx]['id'],
                            "quantity": quantity
                        })
                        print(f"Añadido: {products[idx]['name']} x{quantity}")
                    else:
                        print("Cantidad inválida")
                else:
                    print("Producto inválido")
            except ValueError:
                print("Entrada inválida")

        if items:
            shipping_address = input("\nDirección de envío: ")

            if shipping_address:
                order = await create_order(token, items, shipping_address)

                if order:
                    print("\n=== Pedido Creado ===")
                    print(f"ID: {order['id']}")
                    print(f"Total: ${order['total']}")
                    print(f"Estado: {order['status']}")
                    print("\nItems:")
                    for item in order['items']:
                        print(f"- {item['product_name']} x{item['quantity']} (${item['unit_price']} c/u)")
            else:
                print("La dirección de envío es obligatoria")
        else:
            print("No se han seleccionado productos")

    elif option == "4":
        # Ver pedidos
        orders = await get_orders(token)

        print("\n=== Mis Pedidos ===")
        for order in orders:
            print(f"ID: {order['id']}")
            print(f"Fecha: {order['created_at']}")
            print(f"Total: ${order['total']}")
            print(f"Estado: {order['status']}")
            print("Items:")
            for item in order['items']:
                print(f"- {item['product_name']} x{item['quantity']} (${item['unit_price']} c/u)")
            print("-" * 30)

    elif option == "5":
        print("Saliendo...")

    else:
        print("Opción inválida")

if __name__ == "__main__":
    asyncio.run(main())
```

This completes the implementation of the microservices system for Problem 10. The system includes:

1. **Users Service**: Handles user authentication and management
2. **Products Service**: Manages product catalog and inventory
3. **Orders Service**: Processes customer orders and communicates with other services
4. **API Gateway**: Provides a unified entry point to all services
5. **Orchestrator**: Manages the lifecycle of all services
6. **Client**: Demonstrates how to interact with the system

To run the system, you would execute the orchestrator script:

```bash
python /Users/ric/code/py/playgroung/fastapi_problems/problem10_orchestrator.py
```

Then you can use the client script to interact with the system:

```bash
python /Users/ric/code/py/playgroung/fastapi_problems/problem10_client.py
```

The microservices architecture provides several benefits:

- **Scalability**: Each service can be scaled independently
- **Resilience**: Failure in one service doesn't bring down the entire system
- **Technology diversity**: Different services can use different technologies
- **Team autonomy**: Different teams can work on different services

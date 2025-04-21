# 15 Problemas de Django al Estilo LeetCode

Aquí tienes 15 problemas de Django con sus soluciones, organizados en formato similar a LeetCode:

## 1. Optimización de Consultas N+1

**Problema:** Dado un modelo `Author` con una relación de uno a muchos con `Book`, escribe una vista que muestre todos los autores y sus libros sin caer en el problema N+1.

**Solución:**

```python
# Modelos
class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='books')

# Vista ineficiente (problema N+1)
def inefficient_view(request):
    authors = Author.objects.all()
    # Por cada autor, se realizará una consulta adicional para obtener sus libros
    return render(request, 'authors.html', {'authors': authors})

# Vista optimizada
def efficient_view(request):
    # prefetch_related carga los libros de todos los autores en una sola consulta
    authors = Author.objects.prefetch_related('books').all()
    return render(request, 'authors.html', {'authors': authors})
```

## 2. Middleware de Autenticación Personalizado

**Problema:** Implementa un middleware personalizado que registre todas las solicitudes de usuarios autenticados en la base de datos.

**Solución:**

```python
# models.py
class RequestLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    path = models.CharField(max_length=255)
    method = models.CharField(max_length=10)
    timestamp = models.DateTimeField(auto_now_add=True)

# middleware.py
class RequestLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Solo registrar usuarios autenticados
        if request.user.is_authenticated:
            RequestLog.objects.create(
                user=request.user,
                path=request.path,
                method=request.method
            )

        return response

# settings.py
MIDDLEWARE = [
    # ... otros middlewares
    'myapp.middleware.RequestLoggingMiddleware',
]
```

## 3. Caché de Vistas Basado en Usuario

**Problema:** Implementa un decorador que cachee el resultado de una vista por usuario durante un tiempo específico.

**Solución:**

```python
from django.core.cache import cache
from django.utils.decorators import wraps
from functools import wraps

def cache_per_user(timeout=300):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            # Crear una clave única basada en el usuario y la URL
            if request.user.is_authenticated:
                cache_key = f"user_{request.user.id}_{request.path}"
            else:
                cache_key = f"anonymous_{request.path}"

            # Intentar obtener la respuesta de la caché
            response = cache.get(cache_key)

            if response is None:
                # Si no está en caché, ejecutar la vista
                response = view_func(request, *args, **kwargs)
                # Guardar en caché
                cache.set(cache_key, response, timeout)

            return response
        return _wrapped_view
    return decorator

# Uso
@cache_per_user(timeout=60)
def my_expensive_view(request):
    # Lógica costosa
    return render(request, 'template.html', {'data': expensive_calculation()})
```

## 4. Sistema de Permisos Personalizado

**Problema:** Implementa un sistema de permisos basado en roles que permita acceso a vistas específicas.

**Solución:**

```python
# models.py
class Role(models.Model):
    name = models.CharField(max_length=50, unique=True)
    permissions = models.ManyToManyField('Permission')

class Permission(models.Model):
    name = models.CharField(max_length=50, unique=True)
    codename = models.CharField(max_length=50, unique=True)

class UserRole(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='roles')
    role = models.ForeignKey(Role, on_delete=models.CASCADE)

# decorators.py
def has_permission(permission_codename):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')

            # Verificar si el usuario tiene el permiso
            user_roles = UserRole.objects.filter(user=request.user).select_related('role')
            for user_role in user_roles:
                if user_role.role.permissions.filter(codename=permission_codename).exists():
                    return view_func(request, *args, **kwargs)

            # Si no tiene permiso, mostrar error 403
            return HttpResponseForbidden("No tienes permiso para acceder a esta página")
        return _wrapped_view
    return decorator

# Uso
@has_permission('can_view_reports')
def reports_view(request):
    return render(request, 'reports.html')
```

## 5. Optimización de Consultas con Anotaciones

**Problema:** Dado un modelo `Product` con una relación de muchos a muchos con `Order`, escribe una consulta que devuelva todos los productos con el número de veces que han sido pedidos.

**Solución:**

```python
# models.py
class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)

class Order(models.Model):
    products = models.ManyToManyField(Product, through='OrderItem')
    date = models.DateTimeField(auto_now_add=True)

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)

# views.py
from django.db.models import Count, Sum

def product_stats(request):
    # Obtener productos con conteo de pedidos
    products = Product.objects.annotate(
        order_count=Count('orderitem'),
        total_ordered=Sum('orderitem__quantity')
    )

    return render(request, 'product_stats.html', {'products': products})
```

## 6. Implementación de API REST Paginada

**Problema:** Implementa una vista de API que devuelva una lista paginada de objetos `Article` con filtrado por fecha.

**Solución:**

```python
# models.py
class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    published_date = models.DateField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)

# serializers.py
from rest_framework import serializers

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'published_date', 'author']

# views.py
from rest_framework.pagination import PageNumberPagination
from rest_framework.generics import ListAPIView
from django_filters.rest_framework import DjangoFilterBackend, FilterSet, DateFilter

class ArticleFilter(FilterSet):
    start_date = DateFilter(field_name='published_date', lookup_expr='gte')
    end_date = DateFilter(field_name='published_date', lookup_expr='lte')

    class Meta:
        model = Article
        fields = ['start_date', 'end_date', 'author']

class ArticlePagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100

class ArticleListView(ListAPIView):
    queryset = Article.objects.all().order_by('-published_date')
    serializer_class = ArticleSerializer
    pagination_class = ArticlePagination
    filter_backends = [DjangoFilterBackend]
    filterset_class = ArticleFilter
```

## 7. Sistema de Notificaciones en Tiempo Real

**Problema:** Implementa un sistema de notificaciones en tiempo real utilizando Django Channels.

**Solución:**

```python
# models.py
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    read = models.BooleanField(default=False)

# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class NotificationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]

        if not self.user.is_authenticated:
            await self.close()
            return

        self.room_name = f"user_{self.user.id}_notifications"

        # Unirse al grupo de notificaciones del usuario
        await self.channel_layer.group_add(
            self.room_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Abandonar el grupo
        await self.channel_layer.group_discard(
            self.room_name,
            self.channel_name
        )

    # Recibir mensaje del grupo y enviarlo al WebSocket
    async def notification_message(self, event):
        message = event["message"]

        # Enviar mensaje al WebSocket
        await self.send(text_data=json.dumps({
            "message": message
        }))

# utils.py
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

def send_notification(user_id, message):
    # Crear notificación en la base de datos
    user = User.objects.get(id=user_id)
    notification = Notification.objects.create(user=user, message=message)

    # Enviar notificación a través de WebSocket
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"user_{user_id}_notifications",
        {
            "type": "notification_message",
            "message": message
        }
    )

    return notification
```

## 8. Implementación de Búsqueda Avanzada

**Problema:** Implementa una búsqueda avanzada con múltiples campos y operadores lógicos para el modelo `Product`.

**Solución:**

```python
# models.py
class Category(models.Model):
    name = models.CharField(max_length=100)

class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    in_stock = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

# forms.py
from django import forms

class AdvancedSearchForm(forms.Form):
    query = forms.CharField(required=False, label="Búsqueda general")
    category = forms.ModelChoiceField(
        queryset=Category.objects.all(),
        required=False,
        label="Categoría"
    )
    min_price = forms.DecimalField(required=False, label="Precio mínimo")
    max_price = forms.DecimalField(required=False, label="Precio máximo")
    in_stock_only = forms.BooleanField(required=False, label="Solo en stock")
    sort_by = forms.ChoiceField(
        choices=[
            ('name', 'Nombre'),
            ('price_asc', 'Precio (menor a mayor)'),
            ('price_desc', 'Precio (mayor a menor)'),
            ('newest', 'Más recientes')
        ],
        required=False,
        label="Ordenar por"
    )

# views.py
from django.db.models import Q

def advanced_search(request):
    form = AdvancedSearchForm(request.GET)
    products = Product.objects.all()

    if form.is_valid():
        # Filtrar por búsqueda general
        if query := form.cleaned_data.get('query'):
            products = products.filter(
                Q(name__icontains=query) | Q(description__icontains=query)
            )

        # Filtrar por categoría
        if category := form.cleaned_data.get('category'):
            products = products.filter(category=category)

        # Filtrar por precio
        if min_price := form.cleaned_data.get('min_price'):
            products = products.filter(price__gte=min_price)

        if max_price := form.cleaned_data.get('max_price'):
            products = products.filter(price__lte=max_price)

        # Filtrar por stock
        if form.cleaned_data.get('in_stock_only'):
            products = products.filter(in_stock=True)

        # Ordenar resultados
        if sort_by := form.cleaned_data.get('sort_by'):
            if sort_by == 'price_asc':
                products = products.order_by('price')
            elif sort_by == 'price_desc':
                products = products.order_by('-price')
            elif sort_by == 'newest':
                products = products.order_by('-created_at')
            else:  # 'name'
                products = products.order_by('name')

    return render(request, 'search_results.html', {
        'form': form,
        'products': products
    })
```

## 9. Sistema de Caché Personalizado

**Problema:** Implementa un sistema de caché personalizado que almacene resultados de consultas costosas en Redis con invalidación automática.

**Solución:**

```python
# cache.py
import json
import hashlib
from django.core.cache import cache
from functools import wraps

def cached_query(timeout=3600, prefix='query'):
    """
    Decorador para cachear resultados de consultas de Django.
    Automáticamente invalida la caché cuando los modelos relacionados cambian.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Crear una clave única basada en la función y sus argumentos
            key_parts = [prefix, func.__name__]

            # Añadir argumentos a la clave
            for arg in args:
                key_parts.append(str(arg))

            # Añadir kwargs ordenados a la clave
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")

            # Crear hash de la clave
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Intentar obtener de la caché
            result = cache.get(key)

            if result is None:
                # Si no está en caché, ejecutar la función
                result = func(*args, **kwargs)

                # Guardar en caché
                cache.set(key, result, timeout)

            return result
        return wrapper
    return decorator

# signals.py
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.cache import cache

@receiver([post_save, post_delete])
def invalidate_cache(sender, **kwargs):
    """
    Invalida la caché cuando un modelo es modificado o eliminado.
    """
    # Obtener el nombre del modelo
    model_name = sender.__name__.lower()

    # Patrón para las claves de caché relacionadas con este modelo
    cache_pattern = f"query:*{model_name}*"

    # Obtener todas las claves que coinciden con el patrón
    keys = cache.keys(cache_pattern)

    # Eliminar todas las claves coincidentes
    if keys:
        cache.delete_many(keys)

# views.py
@cached_query(timeout=1800, prefix='expensive_report')
def generate_expensive_report(start_date, end_date, category_id=None):
    # Simulación de una consulta costosa
    query = Order.objects.filter(date__range=[start_date, end_date])

    if category_id:
        query = query.filter(products__category_id=category_id)

    # Realizar agregaciones costosas
    report_data = query.values('products__category__name').annotate(
        total_sales=Sum('orderitem__quantity'),
        revenue=Sum(F('orderitem__quantity') * F('orderitem__product__price'))
    ).order_by('-revenue')

    return list(report_data)
```

## 10. Implementación de Autenticación Multi-Factor

**Problema:** Implementa un sistema de autenticación de dos factores (2FA) utilizando códigos TOTP.

**Solución:**

```python
# models.py
import pyotp
from django.conf import settings

class UserTOTP(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    secret_key = models.CharField(max_length=32)
    is_verified = models.BooleanField(default=False)

    def __str__(self):
        return f"TOTP for {self.user.username}"

    def get_totp_uri(self):
        """Genera la URI para el código QR de TOTP"""
        return pyotp.totp.TOTP(self.secret_key).provisioning_uri(
            name=self.user.email,
            issuer_name=settings.SITE_NAME
        )

    def verify_token(self, token):
        """Verifica si el token TOTP es válido"""
        totp = pyotp.TOTP(self.secret_key)
        return totp.verify(token)

# views.py
import pyotp
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import qrcode
from io import BytesIO
import base64

@login_required
def setup_2fa(request):
    # Verificar si el usuario ya tiene 2FA configurado
    try:
        user_totp = UserTOTP.objects.get(user=request.user)
        already_configured = True
    except UserTOTP.DoesNotExist:
        # Generar una nueva clave secreta
        secret_key = pyotp.random_base32()
        user_totp = UserTOTP.objects.create(
            user=request.user,
            secret_key=secret_key
        )
        already_configured = False

    if request.method == 'POST':
        token = request.POST.get('token')

        if user_totp.verify_token(token):
            user_totp.is_verified = True
            user_totp.save()
            return redirect('2fa_success')
        else:
            error = "Código incorrecto. Inténtalo de nuevo."
            return render(request, 'setup_2fa.html', {'error': error})

    # Generar código QR
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(user_totp.get_totp_uri())
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return render(request, 'setup_2fa.html', {
        'qr_code': img_str,
        'secret_key': user_totp.secret_key,
        'already_configured': already_configured
    })

# Middleware para verificar 2FA
class Require2FAMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # Rutas que no requieren 2FA
        self.exempt_urls = [
            '/login/',
            '/logout/',
            '/setup-2fa/',
            '/verify-2fa/',
        ]

    def __call__(self, request):
        if request.user.is_authenticated:
            # Verificar si el usuario tiene 2FA configurado y verificado
            try:
                user_totp = UserTOTP.objects.get(user=request.user)
                if user_totp.is_verified:
                    # Verificar si la sesión tiene verificación 2FA
                    if not request.session.get('2fa_verified'):
                        # Redirigir a la página de verificación 2FA si no está en una URL exenta
                        if not any(request.path.startswith(url) for url in self.exempt_urls):
                            return redirect('verify_2fa')
            except UserTOTP.DoesNotExist:
                # El usuario no tiene 2FA configurado
                pass

        response = self.get_response(request)
        return response
```

## 11. Implementación de Tareas Asíncronas con Celery

**Problema:** Implementa un sistema para procesar tareas pesadas de forma asíncrona utilizando Celery.

**Solución:**

```python
# settings.py
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

# celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

app = Celery('myproject')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# tasks.py
from celery import shared_task
from django.core.mail import send_mail
import time
from PIL import Image
import os

@shared_task
def process_image(image_path, width, height):
    """Procesa una imagen de forma asíncrona"""
    # Simular procesamiento pesado
    time.sleep(5)

    # Abrir la imagen
    img = Image.open(image_path)

    # Redimensionar
    img = img.resize((width, height), Image.LANCZOS)

    # Guardar la imagen procesada
    filename, ext = os.path.splitext(image_path)
    new_path = f"{filename}_resized{ext}"
    img.save(new_path)

    return new_path

@shared_task
def send_bulk_emails(subject, message, recipient_list):
    """Envía correos electrónicos en masa de forma asíncrona"""
    for recipient in recipient_list:
        send_mail(
            subject,
            message,
            'from@example.com',
            [recipient],
            fail_silently=False,
        )
        # Pequeña pausa para no sobrecargar el servidor de correo
        time.sleep(0.5)

    return len(recipient_list)

# views.py
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .tasks import process_image, send_bulk_emails
from django.contrib import messages

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Guardar la imagen
            image = form.save()

            # Iniciar tarea asíncrona
            task = process_image.delay(
                image.image.path,
                form.cleaned_data['width'],
                form.cleaned_data['height']
            )

            # Guardar el ID de la tarea para seguimiento
            image.task_id = task.id
            image.save()

            messages.success(request, 'Imagen subida. El procesamiento ha comenzado.')
            return redirect('image_status', task_id=task.id)
    else:
        form = ImageUploadForm()

    return render(request, 'upload_image.html', {'form': form})

def image_status(request, task_id):
    # Obtener el estado de la tarea
    task_result = AsyncResult(task_id)

    context = {
        'task_id': task_id,
        'task_status': task_result.status,
        'task_result': task_result.result if task_result.successful() else None,
    }

    return render(request, 'image_status.html', context)
```

## 12. Implementación de Versionado de API

**Problema:** Implementa un sistema de versionado para una API REST que permita mantener múltiples versiones simultáneamente.

**Solución:**

```python
# urls.py
from django.urls import path, include
from rest_framework import routers

# Routers para diferentes versiones
router_v1 = routers.DefaultRouter()
router_v1.register(r'products', views_v1.ProductViewSet)
router_v1.register(r'categories', views_v1.CategoryViewSet)

router_v2 = routers.DefaultRouter()
router_v2.register(r'products', views_v2.ProductViewSet)
router_v2.register(r'categories', views_v2.CategoryViewSet)
router_v2.register(r'reviews', views_v2.ReviewViewSet)  # Nueva en v2

urlpatterns = [
    # API v1
    path('api/v1/', include((router_v1.urls, 'api_v1'))),

    # API v2
    path('api/v2/', include((router_v2.urls, 'api_v2'))),

    # Versión por defecto (redirecciona a la última)
    path('api/', include((router_v2.urls, 'api'))),
]

# serializers_v1.py
from rest_framework import serializers
from .models import Product, Category

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name']

class ProductSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(),
        source='category',
        write_only=True
    )

    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'price', 'category', 'category_id']

# serializers_v2.py
from rest_framework import serializers
from .models import Product, Category, Review

class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = ['id', 'product', 'user', 'rating', 'comment', 'created_at']
        read_only_fields = ['user']

class CategorySerializerV2(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name', 'slug']  # Añadido slug en v2

class ProductSerializerV2(serializers.ModelSerializer):
    category = CategorySerializerV2(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(),
        source='category',
        write_only=True
    )
    reviews = ReviewSerializer(many=True, read_only=True)
    average_rating = serializers.FloatField(read_only=True)

    class Meta:
        model = Product
        fields = [
            'id', 'name', 'description', 'price',
            'category', 'category_id', 'reviews',
            'average_rating', 'in_stock', 'created_at'  # Campos adicionales en v2
        ]

# views_v1.py
from rest_framework import viewsets
from .models import Product, Category
from .serializers_v1 import ProductSerializer, CategorySerializer

class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

# views_v2.py
from rest_framework import viewsets
from django.db.models import Avg
from .models import Product, Category, Review
from .serializers_v2 import ProductSerializerV2, CategorySerializerV2, ReviewSerializer

class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializerV2
    lookup_field = 'slug'  # Cambio en v2: buscar por slug en lugar de id

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.annotate(
        average_rating=Avg('reviews__rating')
    ).all()
    serializer_class = ProductSerializerV2
    filterset_fields = ['category', 'in_stock']  # Filtros adicionales en v2
    search_fields = ['name', 'description']

class ReviewViewSet(viewsets.ModelViewSet):
    queryset = Review.objects.all()
    serializer_class = ReviewSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
```

## 13. Sistema de Auditoría de Cambios

**Problema:** Implementa un sistema que registre automáticamente todos los cambios realizados en los modelos de la base de datos.

**Solución:**

```python
# models.py
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.utils import timezone

class AuditLog(models.Model):
    ACTION_CHOICES = (
        ('CREATE', 'Create'),
        ('UPDATE', 'Update'),
        ('DELETE', 'Delete'),
    )

    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')

    action = models.CharField(max_length=10, choices=ACTION_CHOICES)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    # Para almacenar los cambios en formato JSON
    changes = models.JSONField(default=dict)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.action} on {self.content_type} {self.object_id} at {self.timestamp}"

# middleware.py
from django.utils.deprecation import MiddlewareMixin
from django.contrib.contenttypes.models import ContentType
import threading

# Almacenamiento local de thread para el usuario actual
_thread_locals = threading.local()

class AuditMiddleware(MiddlewareMixin):
    def process_request(self, request):
        _thread_locals.user = request.user if request.user.is_authenticated else None

# utils.py
def get_current_user():
    """Obtiene el usuario actual desde el thread local"""
    return getattr(_thread_locals, 'user', None)

# signals.py
from django.db.models.signals import post_save, post_delete, pre_save
from django.dispatch import receiver
from django.contrib.contenttypes.models import ContentType
from .models import AuditLog
from .utils import get_current_user

@receiver(pre_save)
def pre_save_handler(sender, instance, **kwargs):
    """Guarda el estado anterior del objeto antes de guardarlo"""
    if sender == AuditLog:  # Evitar recursión
        return

    # Solo para objetos existentes (no nuevos)
    if instance.pk:
        try:
            # Guardar el estado anterior en el thread local
            _thread_locals.previous_instance = sender.objects.get(pk=instance.pk)
        except sender.DoesNotExist:
            pass

@receiver(post_save)
def post_save_handler(sender, instance, created, **kwargs):
    """Registra creaciones y actualizaciones"""
    if sender == AuditLog:  # Evitar recursión
        return

    content_type = ContentType.objects.get_for_model(sender)
    user = get_current_user()

    if created:
        # Registrar creación
        changes = {field.name: getattr(instance, field.name) for field in instance._meta.fields}
        AuditLog.objects.create(
            content_type=content_type,
            object_id=instance.pk,
            action='CREATE',
            user=user,
            changes=changes
        )
    else:
        # Registrar actualización
        if hasattr(_thread_locals, 'previous_instance'):
            previous = _thread_locals.previous_instance
            changes = {}

            # Comparar campos para detectar cambios
            for field in instance._meta.fields:
                old_value = getattr(previous, field.name)
                new_value = getattr(instance, field.name)

                if old_value != new_value:
                    changes[field.name] = {
                        'old': old_value,
                        'new': new_value
                    }

            if changes:  # Solo registrar si hay cambios
                AuditLog.objects.create(
                    content_type=content_type,
                    object_id=instance.pk,
                    action='UPDATE',
                    user=user,
                    changes=changes
                )

            # Limpiar el thread local
            delattr(_thread_locals, 'previous_instance')

@receiver(post_delete)
def post_delete_handler(sender, instance, **kwargs):
    """Registra eliminaciones"""
    if sender == AuditLog:  # Evitar recursión
        return

    content_type = ContentType.objects.get_for_model(sender)
    user = get_current_user()

    # Registrar eliminación
    changes = {field.name: getattr(instance, field.name) for field in instance._meta.fields}
    AuditLog.objects.create(
        content_type=content_type,
        object_id=instance.pk,
        action='DELETE',
        user=user,
        changes=changes
    )
```

## 14. Sistema de Gestión de Permisos Basado en Objetos

**Problema:** Implementa un sistema de permisos que permita controlar el acceso a nivel de objeto individual, no solo a nivel de modelo.

**Solución:**

```python
# models.py
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType

class ObjectPermission(models.Model):
    """
    Permisos a nivel de objeto individual.
    Permite asignar permisos específicos a usuarios o grupos
    para objetos individuales en la base de datos.
    """
    PERMISSION_TYPES = (
        ('view', 'Can view'),
        ('change', 'Can change'),
        ('delete', 'Can delete'),
        ('admin', 'Full admin'),
    )

    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')

    permission = models.CharField(max_length=10, choices=PERMISSION_TYPES)

    # El permiso puede ser para un usuario o un grupo
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    group = models.ForeignKey(Group, on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        unique_together = [
            ['content_type', 'object_id', 'permission', 'user'],
            ['content_type', 'object_id', 'permission', 'group'],
        ]

    def __str__(self):
        target = self.user or self.group
        return f"{self.permission} permission for {target} on {self.content_type} {self.object_id}"

# managers.py
class ObjectPermissionManager:
    """
    Gestor de permisos a nivel de objeto.
    Proporciona métodos para verificar y asignar permisos.
    """
    @staticmethod
    def has_permission(user, obj, permission):
        """
        Verifica si un usuario tiene un permiso específico sobre un objeto.
        """
        if user.is_superuser:
            return True

        content_type = ContentType.objects.get_for_model(obj.__class__)

        # Verificar permisos directos del usuario
        direct_perm = ObjectPermission.objects.filter(
            content_type=content_type,
            object_id=obj.pk,
            permission=permission,
            user=user
        ).exists()

        if direct_perm:
            return True

        # Verificar permisos de grupo
        user_groups = user.groups.all()
        group_perm = ObjectPermission.objects.filter(
            content_type=content_type,
            object_id=obj.pk,
            permission=permission,
            group__in=user_groups
        ).exists()

        return group_perm

    @staticmethod
    def grant_permission(user_or_group, obj, permission):
        """
        Otorga un permiso específico a un usuario o grupo sobre un objeto.
        """
        content_type = ContentType.objects.get_for_model(obj.__class__)

        if isinstance(user_or_group, User):
            ObjectPermission.objects.create(
                content_type=content_type,
                object_id=obj.pk,
                permission=permission,
                user=user_or_group
            )
        else:  # Group
            ObjectPermission.objects.create(
                content_type=content_type,
                object_id=obj.pk,
                permission=permission,
                group=user_or_group
            )

    @staticmethod
    def revoke_permission(user_or_group, obj, permission):
        """
        Revoca un permiso específico de un usuario o grupo sobre un objeto.
        """
        content_type = ContentType.objects.get_for_model(obj.__class__)

        if isinstance(user_or_group, User):
            ObjectPermission.objects.filter(
                content_type=content_type,
                object_id=obj.pk,
                permission=permission,
                user=user_or_group
            ).delete()
        else:  # Group
            ObjectPermission.objects.filter(
                content_type=content_type,
                object_id=obj.pk,
                permission=permission,
                group=user_or_group
            ).delete()

    @staticmethod
    def get_objects_for_user(user, model, permission):
        """
        Devuelve todos los objetos de un modelo para los que el usuario tiene un permiso específico.
        """
        content_type = ContentType.objects.get_for_model(model)

        if user.is_superuser:
            return model.objects.all()

        # Obtener IDs de objetos con permisos directos
        user_perms = ObjectPermission.objects.filter(
            content_type=content_type,
            permission=permission,
            user=user
        ).values_list('object_id', flat=True)

        # Obtener IDs de objetos con permisos de grupo
        user_groups = user.groups.all()
        group_perms = ObjectPermission.objects.filter(
            content_type=content_type,
            permission=permission,
            group__in=user_groups
        ).values_list('object_id', flat=True)

        # Combinar IDs y devolver objetos
        object_ids = set(list(user_perms) + list(group_perms))
        return model.objects.filter(pk__in=object_ids)

# decorators.py
from django.core.exceptions import PermissionDenied
from functools import wraps

def object_permission_required(permission):
    """
    Decorador para vistas que verifica si el usuario tiene permiso sobre el objeto.
    El objeto debe ser accesible a través de get_object() en la vista.
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            # La vista debe tener un método get_object
            view_obj = view_func.view_class(
                request=request,
                args=args,
                kwargs=kwargs
            )
            obj = view_obj.get_object()

            if not ObjectPermissionManager.has_permission(request.user, obj, permission):
                raise PermissionDenied("No tienes permiso para realizar esta acción")

            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator

# views.py
from django.views.generic import DetailView
from django.contrib.auth.mixins import LoginRequiredMixin

class ArticleDetailView(LoginRequiredMixin, DetailView):
    model = Article
    template_name = 'article_detail.html'

    def get_object(self, queryset=None):
        obj = super().get_object(queryset)

        # Verificar permiso
        if not ObjectPermissionManager.has_permission(self.request.user, obj, 'view'):
            raise PermissionDenied("No tienes permiso para ver este artículo")

        return obj

# Uso con decorador en vistas basadas en funciones
@object_permission_required('change')
def edit_article(request, pk):
    article = get_object_or_404(Article, pk=pk)
    # Resto de la lógica de la vista
    return render(request, 'edit_article.html', {'article': article})
```

## 15. Implementación de GraphQL con Django

**Problema:** Implementa una API GraphQL para los modelos `Product` y `Category` que permita consultas anidadas y mutaciones.

**Solución:**

```python
# requirements.txt
graphene-django>=2.0
django-filter>=2.0

# settings.py
INSTALLED_APPS = [
    # ... otras apps
    'graphene_django',
]

GRAPHENE = {
    'SCHEMA': 'myapp.schema.schema'
}

# models.py
class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

class Product(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.ForeignKey(Category, related_name='products', on_delete=models.CASCADE)
    in_stock = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

# schema.py
import graphene
from graphene_django import DjangoObjectType
from graphene_django.filter import DjangoFilterConnectionField
from django.db.models import Q

class CategoryType(DjangoObjectType):
    class Meta:
        model = Category
        filter_fields = {
            'name': ['exact', 'icontains', 'istartswith'],
            'description': ['icontains'],
        }
        interfaces = (graphene.relay.Node, )

class ProductType(DjangoObjectType):
    class Meta:
        model = Product
        filter_fields = {
            'name': ['exact', 'icontains', 'istartswith'],
            'description': ['icontains'],
            'price': ['exact', 'lt', 'gt'],
            'category': ['exact'],
            'in_stock': ['exact'],
        }
        interfaces = (graphene.relay.Node, )

class Query(graphene.ObjectType):
    # Consultas individuales
    category = graphene.Field(CategoryType, id=graphene.ID(), name=graphene.String())
    product = graphene.Field(ProductType, id=graphene.ID(), name=graphene.String())

    # Consultas de lista con filtrado
    all_categories = DjangoFilterConnectionField(CategoryType)
    all_products = DjangoFilterConnectionField(ProductType)

    # Búsqueda personalizada
    search_products = graphene.List(
        ProductType,
        search=graphene.String(),
        category_id=graphene.ID(),
        min_price=graphene.Float(),
        max_price=graphene.Float(),
        in_stock=graphene.Boolean()
    )

    def resolve_category(self, info, id=None, name=None):
        if id:
            return Category.objects.get(pk=id)
        if name:
            return Category.objects.get(name=name)
        return None

    def resolve_product(self, info, id=None, name=None):
        if id:
            return Product.objects.get(pk=id)
        if name:
            return Product.objects.get(name=name)
        return None

    def resolve_search_products(self, info, search=None, category_id=None,
                               min_price=None, max_price=None, in_stock=None):
        queryset = Product.objects.all()

        if search:
            filter = Q(name__icontains=search) | Q(description__icontains=search)
            queryset = queryset.filter(filter)

        if category_id:
            queryset = queryset.filter(category_id=category_id)

        if min_price is not None:
            queryset = queryset.filter(price__gte=min_price)

        if max_price is not None:
            queryset = queryset.filter(price__lte=max_price)

        if in_stock is not None:
            queryset = queryset.filter(in_stock=in_stock)

        return queryset

# Mutaciones
class CreateCategoryMutation(graphene.Mutation):
    class Arguments:
        name = graphene.String(required=True)
        description = graphene.String()

    category = graphene.Field(CategoryType)

    def mutate(self, info, name, description=None):
        category = Category.objects.create(
            name=name,
            description=description or ""
        )
        return CreateCategoryMutation(category=category)

class UpdateCategoryMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        name = graphene.String()
        description = graphene.String()

    category = graphene.Field(CategoryType)

    def mutate(self, info, id, name=None, description=None):
        try:
            category = Category.objects.get(pk=id)

            if name is not None:
                category.name = name

            if description is not None:
                category.description = description

            category.save()
            return UpdateCategoryMutation(category=category)
        except Category.DoesNotExist:
            raise Exception(f"Category with ID {id} does not exist")

class CreateProductMutation(graphene.Mutation):
    class Arguments:
        name = graphene.String(required=True)
        description = graphene.String(required=True)
        price = graphene.Float(required=True)
        category_id = graphene.ID(required=True)
        in_stock = graphene.Boolean()

    product = graphene.Field(ProductType)

    def mutate(self, info, name, description, price, category_id, in_stock=True):
        try:
            category = Category.objects.get(pk=category_id)

            product = Product.objects.create(
                name=name,
                description=description,
                price=price,
                category=category,
                in_stock=in_stock
            )

            return CreateProductMutation(product=product)
        except Category.DoesNotExist:
            raise Exception(f"Category with ID {category_id} does not exist")

class UpdateProductMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        name = graphene.String()
        description = graphene.String()
        price = graphene.Float()
        category_id = graphene.ID()
        in_stock = graphene.Boolean()

    product = graphene.Field(ProductType)

    def mutate(self, info, id, name=None, description=None, price=None,
              category_id=None, in_stock=None):
        try:
            product = Product.objects.get(pk=id)

            if name is not None:
                product.name = name

            if description is not None:
                product.description = description

            if price is not None:
                product.price = price

            if category_id is not None:
                try:
                    category = Category.objects.get(pk=category_id)
                    product.category = category
                except Category.DoesNotExist:
                    raise Exception(f"Category with ID {category_id} does not exist")

            if in_stock is not None:
                product.in_stock = in_stock

            product.save()
            return UpdateProductMutation(product=product)
        except Product.DoesNotExist:
            raise Exception(f"Product with ID {id} does not exist")

class DeleteProductMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)

    success = graphene.Boolean()

    def mutate(self, info, id):
        try:
            product = Product.objects.get(pk=id)
            product.delete()
            return DeleteProductMutation(success=True)
        except Product.DoesNotExist:
            return DeleteProductMutation(success=False)

class Mutation(graphene.ObjectType):
    create_category = CreateCategoryMutation.Field()
    update_category = UpdateCategoryMutation.Field()
    create_product = CreateProductMutation.Field()
    update_product = UpdateProductMutation.Field()
    delete_product = DeleteProductMutation.Field()

schema = graphene.Schema(query=Query, mutation=Mutation)

# urls.py
from django.urls import path
from graphene_django.views import GraphQLView
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    # ... otras URLs
    path('graphql/', csrf_exempt(GraphQLView.as_view(graphiql=True))),
]
```

Estos 15 problemas cubren una amplia gama de escenarios y técnicas avanzadas en Django, desde optimización de consultas hasta implementación de APIs GraphQL, sistemas de autenticación multi-factor y gestión de permisos a nivel de objeto.

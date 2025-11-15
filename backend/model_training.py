import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import shutil
from sklearn.utils.class_weight import compute_class_weight

class PlantDiseaseModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        # استخدام Transfer Learning مع EfficientNet مع Fine-tuning
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # تجميد الطبقات الأولى فقط
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # استخدام معدل تعلم منخفض للFine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, class_weights=None, epochs=50, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_accuracy',
                mode='max',
                min_delta=0.01
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=5,
                monitor='val_loss',
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/plant_disease_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
            class_weight=class_weights
        )
        
        return history

def explore_dataset(data_dir):
    """استكشاف هيكل Dataset"""
    print("🔍 جاري استكشاف هيكل البيانات...")
    
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        print(f"❌ المجلد {data_dir} غير موجود!")
        return None
    
    classes = []
    total_images = 0
    
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            # حساب عدد الصور الحقيقي
            image_count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_count += len(list(class_dir.glob(ext)))
            
            classes.append({
                'name': class_name,
                'count': image_count,
                'path': str(class_dir)
            })
            total_images += image_count
            print(f"📁 {class_name}: {image_count} صورة")
    
    print(f"📊 إجمالي الصور: {total_images}")
    print(f"📊 عدد الفئات: {len(classes)}")
    
    return classes

def simple_preprocess_image(image_path, target_size=(224, 224)):
    """معالجة مبسطة للصور - بديل آمن"""
    try:
        # تحميل الصورة
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # التحويل إلى RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # تغيير الحجم
        image = cv2.resize(image, target_size)
        
        # التطبيع البسيط
        image = image.astype(np.float32) / 255.0
        
        return image
        
    except Exception as e:
        print(f"⚠️ خطأ في المعالجة المبسطة {image_path}: {e}")
        return None

def map_to_general_diseases(original_class_name):
    """تحويل أسماء الفئات إلى أمراض عامة للأوراق"""
    
    disease_mapping = {
        # أمراض فطرية
        'Tomato___Early_blight': 'اللفحة_المبكرة',
        'Potato___Early_blight': 'اللفحة_المبكرة',
        'Tomato___Late_blight': 'اللفحة_المتأخرة', 
        'Potato___Late_blight': 'اللفحة_المتأخرة',
        'Tomato___Leaf_Mold': 'عفن_الأوراق',
        'Tomato___Septoria_leaf_spot': 'بقعة_الأوراق',
        
        # أمراض بكتيرية
        'Tomato___Bacterial_spot': 'البقعة_البكتيرية',
        'Pepper___bell___Bacterial_spot': 'البقعة_البكتيرية',
        
        # أمراض فيروسية
        'Tomato___Tomato_mosaic_virus': 'الفيروس_الفسيفسائي',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'التجعد_الاصفر',
        
        # آفات حشرية
        'Tomato___Spider_mites Two-spotted_spider_mite': 'العنكبوت_الأحمر',
        'Tomato___Target_Spot': 'البقعة_الهدف',
        
        # حالات سليمة
        'Tomato___healthy': 'سليم',
        'Pepper___bell___healthy': 'سليم',
        'Potato___healthy': 'سليم'
    }
    
    return disease_mapping.get(original_class_name, 'غير_معروف')

def load_plant_disease_data(data_dir, max_images_per_class=500):
    """تحميل بيانات الأمراض النباتية العامة"""
    images = []
    labels = []
    class_names = []
    class_mapping = {}
    
    print("🔄 جاري تحميل بيانات الأمراض النباتية العامة...")
    
    # استكشاف الفئات
    classes_info = explore_dataset(data_dir)
    if not classes_info:
        return None, None, None
    
    # الفئات المستهدفة (جميع الأمراض المتاحة)
    target_classes = [
        'Tomato___healthy', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Bacterial_spot', 'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Pepper___bell___healthy', 'Pepper___bell___Bacterial_spot',
        'Potato___healthy', 'Potato___Early_blight', 'Potato___Late_blight'
    ]
    
    # إنشاء تعيين الفئات العامة
    for class_info in classes_info:
        original_name = class_info['name']
        
        if original_name not in target_classes:
            continue
            
        # تحويل إلى اسم المرض العام
        general_disease_name = map_to_general_diseases(original_name)
        
        if general_disease_name == 'غير_معروف':
            continue
            
        if general_disease_name not in class_mapping:
            class_mapping[general_disease_name] = len(class_names)
            class_names.append(general_disease_name)
        
        class_idx = class_mapping[general_disease_name]
        print(f"🎯 معالجة: {original_name} → {general_disease_name}")
        
        # تحميل الصور
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(Path(class_info['path']).glob(ext)))
        
        image_files = image_files[:max_images_per_class]
        loaded_count = 0
        
        for image_file in image_files:
            try:
                image = simple_preprocess_image(image_file)
                if image is not None:
                    images.append(image)
                    labels.append(class_idx)
                    loaded_count += 1
                    
                    if loaded_count % 100 == 0:
                        print(f"   📸 تم تحميل {loaded_count} صورة...")
            except:
                continue
        
        print(f"   ✅ تم تحميل {loaded_count} صورة من {general_disease_name}")
    
    if not images:
        print("❌ لم يتم تحميل أي صور!")
        return None, None, None
    
    print(f"🎉 اكتمل التحميل: {len(images)} صورة، {len(class_names)} فئة")
    print(f"📋 الفئات العامة: {class_names}")
    
    X = np.array(images)
    y = np.array(labels)
    
    return X, y, class_names

def check_class_balance(y, class_names):
    """فحص توازن الفئات"""
    unique, counts = np.unique(y, return_counts=True)
    
    print("📊 توزيع الفئات:")
    for cls, count in zip(unique, counts):
        print(f"   {class_names[cls]}: {count} عينة")
    
    # حساب معامل التوازن
    if len(counts) > 0:
        balance_ratio = np.min(counts) / np.max(counts)
        print(f"📈 معامل التوازن: {balance_ratio:.3f}")
        
        if balance_ratio < 0.3:
            print("⚠️ تحذير: الفئات غير متوازنة بشكل خطير!")
            return False, counts
        elif balance_ratio < 0.5:
            print("⚠️ تنبيه: الفئات غير متوازنة بشكل جيد")
            return False, counts
        else:
            print("✅ الفئات متوازنة بشكل جيد")
            return True, counts
    
    return False, counts

def compute_class_weights(y):
    """حساب أوزان الفئات لمعالجة عدم التوازن"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(class_weights))

def augment_data(X, y):
    """زيادة البيانات لتحسين الأداء"""
    print("🔄 جاري زيادة البيانات...")
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    X_augmented = [img for img in X]
    y_augmented = [label for label in y]
    
    # زيادة كل صورة مرتين
    for i in range(len(X)):
        img = X[i]
        img_expanded = np.expand_dims(img, axis=0)
        
        # تطبيق تحويلين مختلفين
        for j, batch in enumerate(datagen.flow(img_expanded, batch_size=1)):
            X_augmented.append(batch[0])
            y_augmented.append(y[i])
            if j == 1:  # صورتين إضافيتين لكل صورة
                break
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    print(f"📈 بعد الزيادة: {len(X_augmented)} صورة")
    
    return X_augmented, y_augmented

def main():
    # إنشاء المجلدات
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # المسارات المحتملة للبيانات
    data_paths = [
        'data/PlantVillage',
        'data/plantvillage', 
        'PlantVillage',
        '../data/PlantVillage'
    ]
    
    data_dir = None
    for path in data_paths:
        if os.path.exists(path):
            data_dir = path
            print(f"✅ تم العثور على البيانات في: {data_dir}")
            break
    
    if not data_dir:
        print("❌ لم يتم العثور على بيانات PlantVillage!")
        return
    
    # تحميل البيانات
    print("🔄 جاري تحميل بيانات الأمراض العامة...")
    X, y, class_names = load_plant_disease_data(data_dir, max_images_per_class=400)
    
    if X is None or len(X) == 0:
        print("❌ فشل في تحميل البيانات!")
        return
    
    print(f"📊 شكل البيانات: {X.shape}")
    print(f"🎯 الفئات النهائية: {class_names}")
    
    # فحص توازن الفئات
    is_balanced, class_counts = check_class_balance(y, class_names)
    
    # حساب أوزان الفئات
    class_weights = compute_class_weights(y)
    print(f"⚖️ أوزان الفئات: {class_weights}")
    
    # زيادة البيانات بشكل مكثف
    print("🔄 تطبيق زيادة البيانات المكثفة...")
    X, y = augment_data(X, y)
    
    # تحويل التسميات
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=0.15, 
        random_state=42, 
        stratify=y,
        shuffle=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.15, 
        random_state=42,
        stratify=np.argmax(y_train, axis=1),
        shuffle=True
    )
    
    print(f"📈 بيانات التدريب: {X_train.shape[0]} عينة")
    print(f"📊 بيانات التحقق: {X_val.shape[0]} عينة") 
    print(f"🧪 بيانات الاختبار: {X_test.shape[0]} عينة")
    
    # بناء النموذج
    print("🔄 جاري بناء النموذج...")
    model_builder = PlantDiseaseModel(
        input_shape=(224, 224, 3), 
        num_classes=len(class_names)
    )
    model = model_builder.build_model()
    
    print("📋 ملخص النموذج:")
    model.summary()
    
    # التدريب
    print("🚀 بدء التدريب...")
    history = model_builder.train(
        X_train, y_train, 
        X_val, y_val, 
        class_weights=class_weights,
        epochs=50,
        batch_size=32
    )
    
    # التقييم النهائي
    print("🧪 جاري تقييم النموذج النهائي...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"🎯 نتائج الاختبار النهائية:")
    print(f"   📊 الدقة: {test_accuracy:.4f}")
    print(f"   🎯 الدقة (Precision): {test_precision:.4f}")
    print(f"   🔍 الاستدعاء (Recall): {test_recall:.4f}")
    print(f"   📉 الخسارة: {test_loss:.4f}")
    
    # تحليل النتائج
    if test_accuracy < 0.6:
        print("❌ النموذج ضعيف الأداء!")
        print("💡 الحلول المقترحة:")
        print("   - زيادة حجم البيانات")
        print("   - استخدام بيانات أكثر توازناً")
        print("   - تجربة بنية نموذج مختلفة")
    elif test_accuracy < 0.8:
        print("⚠️ النموذج متوسط الأداء")
    else:
        print("✅ النموذج ممتاز الأداء!")
    
    # حفظ النموذج
    model.save('models/plant_disease_model.keras')
    print("✅ تم حفظ النموذج في: models/plant_disease_model.keras")
    
    # حفظ معلومات الفئات
    class_info = {
        'class_names': class_names,
        'class_indices': {name: idx for idx, name in enumerate(class_names)},
        'num_classes': len(class_names),
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'training_samples': len(X),
        'class_distribution': {class_names[i]: int(count) for i, count in enumerate(class_counts)},
        'timestamp': str(np.datetime64('now'))
    }
    
    with open('models/class_info.json', 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)
    
    print("✅ تم حفظ معلومات الفئات في: models/class_info.json")
    
    # رسم نتائج التدريب
    plot_training_results(history, test_accuracy)
    
    # اختبار التنبؤ على عينة
    test_prediction(model, X_test, y_test, class_names)

def plot_training_results(history, test_accuracy):
    """رسم نتائج التدريب"""
    try:
        plt.figure(figsize=(15, 5))
        
        # الدقة
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='دقة التدريب', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='دقة التحقق', linewidth=2)
        plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'دقة الاختبار: {test_accuracy:.3f}')
        plt.title('تطور الدقة خلال التدريب', fontsize=14, fontweight='bold')
        plt.ylabel('الدقة', fontsize=12)
        plt.xlabel('الدورة', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # الخسارة
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='خسارة التدريب', linewidth=2)
        plt.plot(history.history['val_loss'], label='خسارة التحقق', linewidth=2)
        plt.title('تطور الخسارة خلال التدريب', fontsize=14, fontweight='bold')
        plt.ylabel('الخسارة', fontsize=12)
        plt.xlabel('الدورة', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("📊 تم حفظ رسم التدريب في: models/training_results.png")
        
    except Exception as e:
        print(f"⚠️ خطأ في رسم النتائج: {e}")

def test_prediction(model, X_test, y_test, class_names):
    """اختبار التنبؤ على عينات من الاختبار"""
    print("\n🔍 اختبار التنبؤ على عينات عشوائية...")
    
    if len(X_test) == 0:
        print("❌ لا توجد بيانات اختبار!")
        return
        
    # اختيار 5 عينات عشوائية
    indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    correct_predictions = 0
    total_predictions = len(indices)
    
    for i, idx in enumerate(indices):
        test_image = X_test[idx]
        true_label = np.argmax(y_test[idx])
        
        # التنبؤ
        prediction = model.predict(np.expand_dims(test_image, axis=0), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        is_correct = true_label == predicted_class
        if is_correct:
            correct_predictions += 1
        
        print(f"📸 العينة {i+1}:")
        print(f"   ✅ الحقيقة: {class_names[true_label]}")
        print(f"   🤖 التنبؤ: {class_names[predicted_class]}")
        print(f"   📊 الثقة: {confidence:.3f}")
        print(f"   {'🎉 صحيح' if is_correct else '❌ خطأ'}")
        print()
    
    accuracy = correct_predictions / total_predictions
    print(f"📈 دقة العينات العشوائية: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

if __name__ == "__main__":
    # تنظيف الذاكرة
    tf.keras.backend.clear_session()
    
    # تشغيل التدريب
    main()
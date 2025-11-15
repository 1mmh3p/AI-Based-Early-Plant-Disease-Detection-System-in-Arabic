import cv2
import numpy as np

def enhance_image(image):
    """
    تحسين جودة الصورة قبل المعالجة
    """
    # تحويل إلى مساحة لونية مناسبة
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # تحسين التباين باستخدام CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = list(cv2.split(lab))
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    lab = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # إزالة الضوضاء
    enhanced = cv2.medianBlur(enhanced, 3)
    
    return enhanced

def segment_leaf(image):
    """
    تقسيم الصورة لعزل الورقة
    """
    # تحويل إلى HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # إنشاء قناع للون الأخضر
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # عمليات مورفولوجية لتحسين القناع
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # تطبيق القناع
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented, mask

def extract_features(image):
    """
    استخراج خصائص من الصورة
    """
    features = {}
    
    # تحويل إلى تدرج الرمادي
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # الخصائص الأساسية
    features['mean_intensity'] = np.mean(gray)
    features['std_intensity'] = np.std(gray)
    
    # إحصائيات اللون
    for i, color in enumerate(['red', 'green', 'blue']):
        features[f'mean_{color}'] = np.mean(image[:,:,i])
        features[f'std_{color}'] = np.std(image[:,:,i])
    
    return features
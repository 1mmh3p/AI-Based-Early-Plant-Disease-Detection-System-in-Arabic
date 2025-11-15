class PlantDiseaseDetector {
    constructor() {
        this.API_BASE_URL = window.location.origin;
        this.checkAuthentication();
        this.initializeApp();
    }

    checkAuthentication() {
        if (localStorage.getItem('loggedIn') !== 'true') {
            window.location.href = 'login.html';
        }
    }

    initializeApp() {
        this.setupEventListeners();
        this.showWelcomeMessage();
    }

    setupEventListeners() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.previewSection = document.getElementById('previewSection');
        this.previewImage = document.getElementById('previewImage');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.resultsSection = document.getElementById('results');
        this.loadingModal = document.getElementById('loadingModal');
        this.analysisTime = document.getElementById('analysisTime');

        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.imageInput.addEventListener('change', (e) => this.handleImageSelection(e));
        
       
        
        this.setupDragAndDrop();
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        this.setupNavigation();
    }

    setupDragAndDrop() {
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleImageSelection({ target: { files } });
            }
        });
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                
                const targetId = link.getAttribute('href').substring(1);
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
    }

    handleImageSelection(event) {
        const file = event.target.files[0];
        if (!file) {
            this.showAlert('لم يتم اختيار أي ملف', 'error');
            return;
        }

        console.log('📁 الملف المختار:', file.name, 'الحجم:', file.size, 'النوع:', file.type);
        this.showFileInfo(file);

        if (!file.type.startsWith('image/')) {
            this.showAlert('الرجاء اختيار ملف صورة فقط', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            this.showAlert('حجم الصورة يجب أن يكون أقل من 10MB', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.previewSection.style.display = 'block';
            this.analyzeBtn.disabled = false;
            this.previewSection.classList.add('fade-in');
        };
        reader.onerror = () => {
            this.showAlert('تعذر قراءة الملف', 'error');
        };
        reader.readAsDataURL(file);
    }

    showFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const fileType = document.getElementById('fileType');
        
        if (fileInfo && fileName && fileSize && fileType) {
            fileName.textContent = `الاسم: ${file.name}`;
            fileSize.textContent = `الحجم: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
            fileType.textContent = `النوع: ${file.type}`;
            fileInfo.style.display = 'block';
        }
    }

    async analyzeImage() {
        const file = this.imageInput.files[0];
        if (!file) {
            this.showAlert('لم يتم اختيار أي ملف', 'error');
            return;
        }

        const startTime = Date.now();
        this.showLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', file);

            console.log('📤 جاري إرسال الملف:', file.name);

            const response = await fetch(`${this.API_BASE_URL}/predict`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`خطأ في الخادم: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ نتيجة التحليل:', result);
            
            const endTime = Date.now();
            const analysisDuration = ((endTime - startTime) / 1000).toFixed(2);
            
            if (this.analysisTime) {
                this.analysisTime.textContent = `${analysisDuration} ثانية`;
            }
            
            this.displayResults(result);

        } catch (error) {
            console.error('❌ Error:', error);
            this.showAlert(`حدث خطأ أثناء تحليل الصورة: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        if (!this.resultsSection) return;

        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('fade-in');

        const isHealthy = result.disease_class && result.disease_class.includes('سليم');
        const statusElement = document.getElementById('diseaseStatus');
        
        if (statusElement) {
            statusElement.innerHTML = `
                <div class="status-icon ${isHealthy ? 'healthy' : 'diseased'}">
                    <i class="fas ${isHealthy ? 'fa-check' : 'fa-exclamation-triangle'}"></i>
                </div>
                <span>${result.disease_class || 'غير معروف'}</span>
            `;
        }

        const confidencePercent = Math.round((result.confidence || 0) * 100);
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceText = document.getElementById('confidenceText');
        
        if (confidenceFill) {
            confidenceFill.style.width = `${confidencePercent}%`;
        }
        if (confidenceText) {
            confidenceText.textContent = `${confidencePercent}%`;
        }

        const diseaseInfo = result.disease_info || {};
        this.updateElementText('symptomsText', diseaseInfo.symptoms || 'لا توجد معلومات متاحة');
        this.updateElementText('causesText', diseaseInfo.causes || 'لا توجد معلومات متاحة');
        this.updateElementText('preventionText', diseaseInfo.prevention || 'لا توجد معلومات متاحة');
        this.updateElementText('treatmentText', diseaseInfo.treatment || 'لا توجد معلومات متاحة');

        if (result.saved_filename) {
            this.displaySavedFilename(result.saved_filename);
        }

        if (result.top_predictions && result.top_predictions.length > 1) {
            this.displayTopPredictions(result.top_predictions);
        }
        
        this.showAlert('تم تحليل الصورة بنجاح', 'success');
    }

    displaySavedFilename(filename) {
        const container = document.getElementById('savedFileInfo');
        if (!container) return;
        
        container.innerHTML = `
            <div class="saved-file-info">
                <h4><i class="fas fa-save"></i> تم حفظ الصورة</h4>
                <p><strong>اسم الملف:</strong> ${filename}</p>
                <p class="text-muted">تم حفظ الصورة في مجلد uploads باسم مرتبط بنتيجة التحليل</p>
            </div>
        `;
        container.style.display = 'block';
    }

    displayTopPredictions(predictions) {
        const container = document.getElementById('allPredictions');
        if (!container) return;
        
        let html = '<h4>التوقعات الإضافية:</h4><div class="predictions-list">';
        
        predictions.forEach((pred, index) => {
            const percent = Math.round(pred.confidence * 100);
            html += `
                <div class="prediction-item">
                    <span class="disease-name">${pred.class}</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${percent}%"></div>
                    </div>
                    <span class="confidence-text">${percent}%</span>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
        container.style.display = 'block';
    }

    updateElementText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
    }

    showLoading(show) {
        if (this.loadingModal) {
            this.loadingModal.style.display = show ? 'flex' : 'none';
        }
    }

    showAlert(message, type = 'info') {
        const alert = document.createElement('div');
        alert.className = `alert ${type} show`;
        alert.innerHTML = `
            <i class="fas fa-${this.getAlertIcon(type)} alert-icon"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(alert);
        
        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => alert.remove(), 300);
        }, 4000);
    }

    getAlertIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    showWelcomeMessage() {
        setTimeout(() => {
            this.showAlert('مرحباً بك في نظام الذكاء الاصطناعي للكشف عن الأمراض النباتية', 'info');
        }, 1000);
    }

    analyzeNewImage() {
        this.imageInput.value = '';
        this.previewSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.analyzeBtn.disabled = true;
        
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.style.display = 'none';
        }
        
        const allPredictions = document.getElementById('allPredictions');
        if (allPredictions) {
            allPredictions.style.display = 'none';
        }
        
        const savedFileInfo = document.getElementById('savedFileInfo');
        if (savedFileInfo) {
            savedFileInfo.style.display = 'none';
        }
        
        const uploadSection = document.getElementById('upload');
        if (uploadSection) {
            uploadSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    logout() {
        localStorage.removeItem('loggedIn');
        window.location.href = 'login.html';
    }
}

let plantDetector;

document.addEventListener('DOMContentLoaded', () => {
    plantDetector = new PlantDiseaseDetector();
    
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    document.querySelectorAll('section').forEach(section => {
        observer.observe(section);
    });
});

window.analyzeNewImage = () => plantDetector.analyzeNewImage();
window.logout = () => plantDetector.logout();

// وظائف إضافية للتصميم المحسن

function downloadReport() {
    const diseaseName = document.getElementById('diseaseStatus').querySelector('span').textContent;
    const confidence = document.getElementById('confidenceText').textContent;
    const analysisTime = document.getElementById('analysisTime').textContent;
    
    const report = `
        تقرير تشخيص الأمراض النباتية
        =============================
        
        الحالة الصحية: ${diseaseName}
        مستوى الثقة: ${confidence}
        وقت التحليل: ${analysisTime}
        
        الأعراض: ${document.getElementById('symptomsText').textContent}
        الأسباب: ${document.getElementById('causesText').textContent}
        الوقاية: ${document.getElementById('preventionText').textContent}
        العلاج: ${document.getElementById('treatmentText').textContent}
        
        تم إنشاء هذا التقرير بواسطة نظام الذكاء الاصطناعي للكشف عن الأمراض النباتية
        التاريخ: ${new Date().toLocaleDateString('ar-EG')}
    `;
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `تقرير_تشخيص_${diseaseName}_${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    plantDetector.showAlert('تم تحميل التقرير بنجاح', 'success');
}

// تحسينات للتصميم التفاعلي
document.addEventListener('DOMContentLoaded', function() {
    // إضافة تأثيرات للبطاقات عند التمرير
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                if (entry.target.classList.contains('step')) {
                    entry.target.style.animationDelay = `${Array.from(entry.target.parentElement.children).indexOf(entry.target) * 0.2}s`;
                }
            }
        });
    }, observerOptions);

    // مراقبة جميع الأقسام والعناصر
    document.querySelectorAll('section, .feature-card, .step, .disease-card, .component-card').forEach(element => {
        observer.observe(element);
    });

    // إضافة تأثيرات للزر CTA
    const ctaButton = document.querySelector('.cta-button');
    if (ctaButton) {
        ctaButton.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px)';
        });
        
        ctaButton.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    }
});

// تحسين التنقل النشط
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').substring(1) === current) {
            link.classList.add('active');
        }
    });
});
"""
Django Integration Examples for DeepSeek-OCR Service

This file shows how to integrate the OCR service with your Django app.
Copy relevant code into your Django views or services.
"""

# ============================================================================
# 1. Django Settings Configuration
# ============================================================================

# Add to invoices/settings/base.py or your settings file:
"""
# DeepSeek-OCR Service Configuration
OCR_SERVICE_URL = env('OCR_SERVICE_URL', default='http://localhost:8765')
OCR_SERVICE_API_KEY = env('OCR_SERVICE_API_KEY', default='')
OCR_SERVICE_TIMEOUT = env.int('OCR_SERVICE_TIMEOUT', default=60)

# Feature flag to enable/disable DeepSeek-OCR (fallback to pytesseract if disabled)
USE_DEEPSEEK_OCR = env.bool('USE_DEEPSEEK_OCR', default=True)
"""

# Add to .env file:
"""
OCR_SERVICE_URL=http://localhost:8765
OCR_SERVICE_API_KEY=
USE_DEEPSEEK_OCR=true
"""


# ============================================================================
# 2. Updated Prescription Upload Service (prescription_upload_service.py)
# ============================================================================

"""
Replace the extract_pdf_text function in:
/Users/mehdi/workspace/clients/inur-sur.lu/nuno/inur.django/invoices/services/prescription_upload_service.py
"""

def extract_pdf_text_with_deepseek(pdf_file) -> tuple[str, bool]:
    """
    Extract text from PDF file using DeepSeek-OCR with PyPDF2/pytesseract fallback.

    Args:
        pdf_file: The uploaded PDF file

    Returns:
        Tuple of (extracted_text, ocr_used)
    """
    from django.conf import settings
    from nunoocr_client import DeepSeekOCRClient, DeepSeekOCRError

    ocr_used = False
    pdf_text = ''

    # Try PyPDF2 first (for text-based PDFs)
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")

    # If PyPDF2 couldn't extract enough text, use OCR
    if not pdf_text or len(pdf_text.strip()) < 50:
        # Try DeepSeek-OCR if enabled
        if getattr(settings, 'USE_DEEPSEEK_OCR', True):
            try:
                ocr_client = DeepSeekOCRClient(
                    base_url=getattr(settings, 'OCR_SERVICE_URL', 'http://localhost:8765'),
                    api_key=getattr(settings, 'OCR_SERVICE_API_KEY', None),
                    timeout=getattr(settings, 'OCR_SERVICE_TIMEOUT', 60)
                )

                # Check if service is available
                if ocr_client.health_check():
                    pdf_file.seek(0)
                    result = ocr_client.extract_text(pdf_file, file_type='pdf')
                    pdf_text = result['text']
                    ocr_used = True
                    print("DeepSeek-OCR extraction successful")
                else:
                    print("DeepSeek-OCR service not available, falling back to pytesseract")
                    # Fall through to pytesseract fallback below

            except DeepSeekOCRError as e:
                print(f"DeepSeek-OCR extraction failed: {e}, falling back to pytesseract")
                # Fall through to pytesseract fallback below
            except Exception as e:
                print(f"Unexpected DeepSeek-OCR error: {e}, falling back to pytesseract")
                # Fall through to pytesseract fallback below

        # Fallback to pytesseract if DeepSeek-OCR failed or disabled
        if not pdf_text or len(pdf_text.strip()) < 50:
            try:
                from pdf2image import convert_from_bytes
                import pytesseract

                pdf_file.seek(0)
                images = convert_from_bytes(pdf_file.read(), dpi=300)

                pdf_text = ''
                for image in images:
                    page_text = pytesseract.image_to_string(image, lang='fra+eng')
                    pdf_text += page_text + '\n'

                if pdf_text:
                    ocr_used = True
                    print("pytesseract OCR extraction successful")
            except Exception as ocr_error:
                print(f"pytesseract extraction failed: {ocr_error}")

    return pdf_text, ocr_used


# ============================================================================
# 3. Django View Example - Using Plain Text Extraction
# ============================================================================

def upload_prescription_view_example(request):
    """
    Example view showing how to use OCR in prescription upload.
    """
    from django.shortcuts import render, redirect
    from django.contrib import messages
    from nunoocr_client import DjangoOCRService

    if request.method == 'POST' and request.FILES.get('prescription_file'):
        prescription_file = request.FILES['prescription_file']

        # Initialize OCR service
        ocr_service = DjangoOCRService()

        # Check if service is available
        if not ocr_service.is_available():
            messages.warning(
                request,
                "Le service OCR n'est pas disponible. Utilisation de l'extraction basique."
            )
            # Fall back to existing extraction method
            # ... your existing code ...

        else:
            try:
                # Extract text using DeepSeek-OCR
                result = ocr_service.extract_from_uploaded_file(
                    prescription_file,
                    extract_structured=False  # Plain text extraction
                )

                extracted_text = result['text']
                tokens_used = result['metadata']['tokens_used']

                # Now use your existing extraction functions on the text
                from invoices.services.prescription_upload_service import (
                    extract_doctor_code,
                    extract_patient_matricule,
                    extract_date,
                    # ... other functions
                )

                doctor_code = extract_doctor_code(extracted_text)
                patient_matricule = extract_patient_matricule(extracted_text)
                prescription_date = extract_date(extracted_text)
                # ... etc

                # Store in session for confirmation step
                request.session['extracted_data'] = {
                    'text': extracted_text,
                    'doctor_code': doctor_code,
                    'patient_matricule': patient_matricule,
                    'prescription_date': prescription_date,
                    # ... other fields
                }

                messages.success(request, "OCR extraction réussie!")
                return redirect('prescription_confirm')

            except Exception as e:
                messages.error(request, f"Erreur d'extraction OCR: {str(e)}")
                # Fall back to existing method
                # ... your existing code ...

    return render(request, 'prescription_upload.html')


# ============================================================================
# 4. Django View Example - Using Structured Data Extraction
# ============================================================================

def upload_prescription_structured_view_example(request):
    """
    Example using structured data extraction directly from OCR.
    This bypasses regex extraction and gets structured JSON from the model.
    """
    from django.shortcuts import render, redirect
    from django.contrib import messages
    from nunoocr_client import DjangoOCRService
    from invoices.models import Physician, Patient

    if request.method == 'POST' and request.FILES.get('prescription_file'):
        prescription_file = request.FILES['prescription_file']

        ocr_service = DjangoOCRService()

        if not ocr_service.is_available():
            messages.warning(request, "Service OCR non disponible.")
            # Fall back...
        else:
            try:
                # Extract structured data directly
                data = ocr_service.extract_from_uploaded_file(
                    prescription_file,
                    extract_structured=True  # Structured extraction
                )

                # Find matching physician
                physician = None
                if data.get('doctor_code'):
                    physician = Physician.objects.filter(
                        provider_code=data['doctor_code']
                    ).first()

                # Find matching patient
                patient = None
                if data.get('patient_matricule'):
                    patient = Patient.objects.filter(
                        code_sn=data['patient_matricule']
                    ).first()

                # Store in session
                request.session['extracted_prescription'] = {
                    'doctor_code': data.get('doctor_code'),
                    'doctor_name': data.get('doctor_name'),
                    'patient_matricule': data.get('patient_matricule'),
                    'patient_name': data.get('patient_name'),
                    'prescription_date': data.get('prescription_date'),
                    'medications': data.get('medications', []),
                    'notes': data.get('notes', ''),
                    'physician_id': physician.id if physician else None,
                    'patient_id': patient.id if patient else None,
                }

                messages.success(
                    request,
                    f"Données extraites: {len(data.get('medications', []))} médicament(s)"
                )

                return redirect('prescription_confirm')

            except Exception as e:
                messages.error(request, f"Erreur: {str(e)}")

    return render(request, 'prescription_upload.html')


# ============================================================================
# 5. Background Job Example (for RQ/Celery)
# ============================================================================

def process_prescription_ocr_job(prescription_id: int):
    """
    Background job to process prescription OCR.
    Can be used with django-rq or Celery.

    Usage with django-rq:
        import django_rq
        queue = django_rq.get_queue('default')
        queue.enqueue(process_prescription_ocr_job, prescription_id=123)
    """
    from django.conf import settings
    from invoices.models import MedicalPrescription
    from nunoocr_client import DeepSeekOCRClient

    try:
        prescription = MedicalPrescription.objects.get(id=prescription_id)

        # Initialize OCR client
        ocr_client = DeepSeekOCRClient(
            base_url=settings.OCR_SERVICE_URL,
            api_key=settings.OCR_SERVICE_API_KEY
        )

        # Extract text from the uploaded file
        with prescription.file_upload.open('rb') as f:
            result = ocr_client.extract_text(f, file_type='pdf')

        # Store extracted text in notes if empty
        if not prescription.notes:
            prescription.notes = result['text']
            prescription.save()

        print(f"OCR processed successfully for prescription {prescription_id}")
        return True

    except Exception as e:
        print(f"OCR processing failed for prescription {prescription_id}: {e}")
        return False


# ============================================================================
# 6. Health Check Management Command
# ============================================================================

"""
Create: invoices/management/commands/check_ocr_service.py
"""

from django.core.management.base import BaseCommand
from django.conf import settings
from nunoocr_client import DeepSeekOCRClient


class Command(BaseCommand):
    help = 'Check if DeepSeek-OCR service is available and healthy'

    def handle(self, *args, **options):
        ocr_client = DeepSeekOCRClient(
            base_url=settings.OCR_SERVICE_URL,
            api_key=settings.OCR_SERVICE_API_KEY
        )

        self.stdout.write(f"Checking OCR service at: {settings.OCR_SERVICE_URL}")

        if ocr_client.health_check():
            self.stdout.write(self.style.SUCCESS('✅ OCR service is healthy'))
        else:
            self.stdout.write(self.style.ERROR('❌ OCR service is not responding'))
            self.stdout.write('Check if the service is running:')
            self.stdout.write('  docker-compose -f /path/to/nunoOcr/docker-compose.yml ps')


# ============================================================================
# 7. Admin Action Example
# ============================================================================

"""
Add to invoices/admin.py for MedicalPrescriptionAdmin
"""

from django.contrib import admin, messages
from nunoocr_client import DjangoOCRService


def reprocess_with_ocr(modeladmin, request, queryset):
    """
    Admin action to reprocess selected prescriptions with DeepSeek-OCR.
    """
    ocr_service = DjangoOCRService()

    if not ocr_service.is_available():
        messages.error(request, "Service OCR non disponible")
        return

    success_count = 0
    for prescription in queryset:
        try:
            if prescription.file_upload:
                with prescription.file_upload.open('rb') as f:
                    result = ocr_service.extract_from_uploaded_file(f)

                # Update notes with extracted text
                prescription.notes = result['text']
                prescription.save()
                success_count += 1

        except Exception as e:
            messages.warning(
                request,
                f"Échec pour prescription {prescription.id}: {str(e)}"
            )

    messages.success(
        request,
        f"{success_count} prescription(s) retraitée(s) avec succès"
    )


reprocess_with_ocr.short_description = "Retraiter avec DeepSeek-OCR"


# Add to MedicalPrescriptionAdmin:
# actions = [reprocess_with_ocr]

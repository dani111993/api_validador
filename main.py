import io
import json
import os
import re
from datetime import datetime, timedelta

import cv2
import easyocr
import numpy as np
import ollama
from PIL import Image
import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import uvicorn
from transformers import ViTForImageClassification, ViTImageProcessor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

print('Inicializando EasyOCR...')
ocr_reader = easyocr.Reader(['es'], gpu=False)

print('Cargando el pre-procesador y modelo Vision Transformer (ViT)...')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k', output_hidden_states=True
).to(device)


class AnalisisINEPersonalizado(BaseModel):
    tipo_documento_detectado: str = Field(description="Debe ser estrictamente 'INE'.")
    nombre_completo_titular: str = Field(description='Nombre completo del titular unido en un solo texto fluido.')
    curp: str = Field(description='CURP de 18 caracteres encontrada. Si no existe, "No válido".')
    vigencia_estandarizada: str = Field(description='Fecha exacta en formato YYYY-12-31 basada en el último año detectado.')
    es_valido: bool = Field(description='True si el año de vigencia es mayor o igual al actual, False en caso contrario.')
    motivo_validez: str = Field(description='Breve explicación del cálculo de vigencia.')

def parse_spanish_date(date_str: str):
    if not date_str:
        return None

    months = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12,
    }

    cleaned = re.sub(r'\bde\b|\ba\b', '', date_str.lower()).strip()
    parts = cleaned.split()

    if len(parts) >= 3 and parts[1] in months:
        try:
            day = int(parts[0])
            month = months[parts[1]]
            year = int(parts[2])
            if year < 100:
                year += 2000
            return datetime(year, month, day)
        except ValueError:
            pass

    formats = (
        '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d',
        '%d %b %y', '%d %B %y', '%d %b %Y', '%d %B %Y',
    )
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def preprocesar_imagen_recibo(img_cv: np.ndarray) -> np.ndarray:
    """Aplica escala de grises, desenfoque gaussiano y binarización adaptativa para optimizar el OCR en recibos."""
    if len(img_cv.shape) == 3:
        img_gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        img_gris = img_cv

    blurred = cv2.GaussianBlur(img_gris, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )
    return thresh

def extraer_texto_y_ocr(file_path: str):
    file_path_lower = file_path.lower()
    results = []
    full_text = ""

    is_pdf = file_path_lower.endswith('.pdf')

    es_posible_ine = False
    if is_pdf:
        try:
            doc_check = pymupdf.open(file_path)
            texto_inicial = ""
            for pagina in doc_check:
                texto_inicial += pagina.get_text('text') + "\n"
            doc_check.close()
            
            if any(k in texto_inicial.upper() for k in ['INSTITUTO NACIONAL ELECTORAL', 'CREDENCIAL PARA VOTAR', 'CLAVE DE ELECTOR']):
                es_posible_ine = True
        except Exception:
            pass
    else:
        temp_ocr_check = ocr_reader.readtext(file_path, detail=0)
        texto_inicial = ' '.join(temp_ocr_check)
        if any(k in texto_inicial.upper() for k in ['INSTITUTO NACIONAL ELECTORAL', 'CREDENCIAL PARA VOTAR', 'CLAVE DE ELECTOR']):
            es_posible_ine = True

    if es_posible_ine:
        print("  -> Detectado perfil INE: Leyendo directo de la fuente original sin binarización agresiva.")
        if is_pdf:
            try:
                doc = pymupdf.open(file_path)
                for pagina in doc:
                    texto_pagina = pagina.get_text('text')
                    lineas = [l.strip() for l in texto_pagina.split('\n') if l.strip()]
                    results.extend(lineas)
                    
                    if len(texto_pagina.strip()) < 15:
                        pix = pagina.get_pixmap(dpi=300)
                        resultados_ocr = ocr_reader.readtext(np.array(Image.open(io.BytesIO(pix.tobytes('png')))), detail=0)
                        results.extend(resultados_ocr)
                        texto_pagina = ' '.join(resultados_ocr)
                        
                    full_text += texto_pagina + '\n'
                doc.close()
            except Exception as e:
                print(f'Error procesando PDF de INE {file_path}: {e}')
        else:
            results = ocr_reader.readtext(file_path, detail=0)
            full_text = ' '.join(results)
    else:
        print("  -> Detectado perfil Recibo/Servicio (Telmex): Forzando rasterización a imagen y OCR robusto.")
        if is_pdf:
            try:
                doc = pymupdf.open(file_path)
                zoom = 300 / 72 
                matriz = pymupdf.Matrix(zoom, zoom)
                
                for pagina in doc:
                    pix = pagina.get_pixmap(matrix=matriz)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

                    img_procesada = preprocesar_imagen_recibo(img_cv)

                    resultados_ocr = ocr_reader.readtext(img_procesada, detail=0)
                    results.extend(resultados_ocr)
                    full_text += ' '.join(resultados_ocr) + '\n'
                doc.close()
            except Exception as e:
                print(f'Error procesando PDF de recibo {file_path}: {e}')
        else:
            img_cv = cv2.imread(file_path)
            if img_cv is not None:
                img_procesada = preprocesar_imagen_recibo(img_cv)
                results = ocr_reader.readtext(img_procesada, detail=0)
                full_text = ' '.join(results)
            else:
                results = ocr_reader.readtext(file_path, detail=0)
                full_text = ' '.join(results)

    return full_text, results

def analizar_ine(texto_documento):
    texto_limpio = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', texto_documento.replace('"', "'"))
    if len(texto_limpio.strip()) < 10:
        return None

    prompt = f"""
    Eres un experto analista de Credenciales para Votar de México (INE). Extrae con precisión:
    1. **nombre_completo_titular**: Nombres y apellidos unidos en un solo texto fluido.
    2. **curp**: CURP de 18 caracteres. Solo el CURP, no extraigas información de más. Si no existe, 'No válido'.
    3. **vigencia_estandarizada**: Toma el último año encontrado y formatéalo estrictamente a `YYYY-12-31`.

    Regla de seguridad JSON: No uses comillas dobles (") dentro de los valores y devuelve exclusivamente JSON válido.
    
    Texto extraído:
    {texto_limpio}
    """
    try:
        response = ollama.chat(
            model='mistral',
            messages=[
                {'role': 'system', 'content': 'Extractor documental de INEs. Responde únicamente con el JSON solicitado.'},
                {'role': 'user', 'content': prompt}
            ],
            format=AnalisisINEPersonalizado.model_json_schema(),
            options={'temperature': 0.0},
        )
        texto_resp = response.message.content.strip()
        if not texto_resp.endswith('}'):
            texto_resp += '"}' if texto_resp.count('"') % 2 != 0 else '}'
        
        datos = json.loads(texto_resp)
        if anios := re.findall(r'20\d{2}', str(datos.get('vigencia_estandarizada', ''))):
            ultimo_anio = int(anios[-1])
            datos['vigencia_estandarizada'] = f'{ultimo_anio}-12-31'
            datos['es_valido'] = ultimo_anio >= datetime.now().year
            datos['motivo_validez'] = f'Vigente hasta el 31/12/{ultimo_anio}.' if datos['es_valido'] else f'Vencido en el año {ultimo_anio}.'
        else:
            datos['es_valido'], datos['motivo_validez'] = False, 'No se pudo identificar un año de vigencia válido.'
        
        return {
            'tipo_doc': 'INE',
            'nombre': datos.get('nombre_completo_titular'),
            'identificador_principal': datos.get('curp'),
            'fecha_detectada': datos.get('vigencia_estandarizada'),
            'es_valido_ia': datos.get('es_valido'),
            'motivo_validez': datos.get('motivo_validez')
        }
    except Exception as e:
        print(f'Error en IA INE: {e}')
        return None

def extract_telmex_data(full_text: str, results: list) -> dict:
    extracted_data = {'tipo_doc': 'telmex'}

    codigos_largos = re.findall(r'\b[A-Z0-9]{10,18}\b', full_text.upper())
    if codigos_largos:
        for codigo in codigos_largos:
            if len(codigo) == 18:
                extracted_data['identificador_principal'] = codigo
                break
        if 'identificador_principal' not in extracted_data:
            extracted_data['identificador_principal'] = codigos_largos[0]

    date_regex_pattern = r'(\d{1,2}[\s/-]*(?:de[\s/-]*)?[A-Za-zÁÉÍÓÚÑa-záéíóúñ]{3,10}[\s/-]*(?:de[\s/-]*)?\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'

    fecha_detectada = None
    keywords_fecha = [
        'pago oportuno', 'pagar antes', 'límite de pago', 'limite de pago', 
        'límite', 'limite', 'vencimiento', 'vigencia', 'pagar antes de',
    ]

    for i, res in enumerate(results):
        res_lower = res.lower()
        if any(kw in res_lower for kw in keywords_fecha):
            for j in range(i, min(len(results), i + 4)):
                match_fecha = re.search(date_regex_pattern, results[j])
                if match_fecha:
                    candidata = match_fecha.group(0).strip()
                    if parse_spanish_date(candidata):
                        fecha_detectada = candidata
                        break
        if fecha_detectada:
            break

    if not fecha_detectada:
        fechas_encontradas = re.findall(date_regex_pattern, full_text)
        for f in reversed(fechas_encontradas):
            candidata = f[0] if isinstance(f, tuple) else f
            if parse_spanish_date(candidata.strip()):
                fecha_detectada = candidata.strip()
                break

    if fecha_detectada:
        candidata_limpia = fecha_detectada.strip()
        if re.match(r'^[A-Z0-9]{10,}$', candidata_limpia, re.IGNORECASE):
            fecha_detectada = None
        elif not parse_spanish_date(candidata_limpia):
            fecha_detectada = None
        else:
            extracted_data['fecha_detectada'] = candidata_limpia

    amounts = re.findall(r'\b\d{1,3}(?:,\d{3})*\.\d{2}\b', full_text)
    if amounts:
        extracted_data['monto_o_total'] = amounts[-1]

    palabras_prohibidas = [
        'domicilio', 'clave', 'curp', 'estado', 'municipio', 'régimen', 'regimen', 
        'fiscal', 'rfc', 'lugar', 'fecha', 'elector', 'sexo', 's.a.', 'de c.v', 
        'comisión', 'federal', 'instituto', 'nacional', 'electoral', 'credencial', 
        'servicio', 'mexico', 'méxico', 'telmex', 'recibo', 'factura', 'total'
    ]

    nombre_encontrado = None
    etiquetas_nombre = ['nombre', 'cliente', 'titular', 'usuario', 'contribuyente', 'nombre:']
    
    for i, res in enumerate(results):
        res_clean = res.lower().strip()
        if any(etq in res_clean for etq in etiquetas_nombre):
            for j in range(i + 1, min(len(results), i + 4)):
                candidate = results[j].strip()
                if len(candidate) > 4 and not re.search(r'\d', candidate) and not any(kw in candidate.lower() for kw in palabras_prohibidas):
                    nombre_encontrado = candidate
                    break
        if nombre_encontrado:
            break

    if not nombre_encontrado:
        for res in results:
            texto = res.strip()
            texto_lower = texto.lower()
            palabras = texto.split()
            if 2 <= len(palabras) <= 4 and not re.search(r'\d', texto):
                if texto.isupper() and not any(kw in texto_lower for kw in palabras_prohibidas):
                    nombre_encontrado = texto
                    break

    if nombre_encontrado:
        extracted_data['nombre'] = nombre_encontrado

    return extracted_data


def extract_real_data_from_file(file_path: str) -> dict:
    full_text, results = extraer_texto_y_ocr(file_path)
    
    if any(k in full_text.upper() for k in ['INSTITUTO NACIONAL ELECTORAL', 'CREDENCIAL PARA VOTAR', 'CLAVE DE ELECTOR']):
        if res_ine := analizar_ine(full_text):
            return res_ine
            
    return extract_telmex_data(full_text, results)


def evaluate_document_business_rules(extracted_data: dict) -> bool:
    current_date = datetime(2026, 9, 9)

    if extracted_data.get('tipo_doc') == 'INE':
        return extracted_data.get('es_valido_ia', False)

    fecha_str = extracted_data.get('fecha_detectada')
    if fecha_str:
        parsed_date = parse_spanish_date(fecha_str)
        if parsed_date:
            three_months_ago = current_date - timedelta(days=90)
            if parsed_date < three_months_ago:
                return False

    return True

def get_vit_embeddings(image: Image.Image) -> torch.Tensor:
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.hidden_states[-1][:, 0, :].cpu()


def calculate_document_reward(
    extracted_data: dict,
    required_fields: list,
    vit_embedding: torch.Tensor,
    classifier_head_validity_prob: float,
) -> tuple:
    fields_reward = 0.0
    if required_fields:
        present_fields = sum(1 for field in required_fields if extracted_data.get(field))
        fields_reward = present_fields / len(required_fields)

    validez_score = (fields_reward * 0.5) + (classifier_head_validity_prob * 0.5)
    is_business_rule_valid = evaluate_document_business_rules(extracted_data)
    extraction_score = 1.0 if len(extracted_data) >= 2 else 0.4

    embedding_norm = torch.norm(vit_embedding).item()
    visual_score = min(max(embedding_norm / 50.0, 0.0), 1.0)

    total_reward = (0.4 * validez_score) + (0.4 * extraction_score) + (0.2 * visual_score)
    is_valid = (total_reward >= 0.55) and is_business_rule_valid

    return round(total_reward, 4), is_valid


class DocumentValidityClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


classifier_model = DocumentValidityClassifier().to(device)
if os.path.exists('document_validity_classifier.pth'):
    classifier_model.load_state_dict(torch.load('document_validity_classifier.pth', map_location=device))
classifier_model.eval()


app = FastAPI(title="API Analizador Documental Híbrido (INE + Telmex + ViT)", version="4.3")

@app.post("/procesar-documento/")
async def procesar_documento_api(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")
    
    temp_filename = f"temp_{file.filename}"
    try:
        content = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(content)
            
        extracted_data = extract_real_data_from_file(temp_filename)

        if temp_filename.lower().endswith('.pdf'):
            doc = pymupdf.open(temp_filename)
            pix = doc[0].get_pixmap(dpi=150)
            image = Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
            doc.close()
        else:
            image = Image.open(temp_filename).convert('RGB')

        vit_embedding = get_vit_embeddings(image)

        with torch.no_grad():
            prob_tensor = classifier_model(vit_embedding.to(device))
            classifier_prob = prob_tensor.item()

        campos_necesarios = ['nombre', 'fecha_detectada']
        reward, is_valid = calculate_document_reward(
            extracted_data=extracted_data,
            required_fields=campos_necesarios,
            vit_embedding=vit_embedding,
            classifier_head_validity_prob=classifier_prob
        )
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        if extracted_data:
            return {
                "archivo": file.filename, 
                "tipo_detectado": extracted_data.get('tipo_doc'), 
                "resultado": extracted_data,
                "recompensa_calculada": reward,
                "probabilidad_modelo_vit": round(classifier_prob, 4),
                "es_valido_por_reglas": is_valid
            }
        raise HTTPException(status_code=422, detail="No se pudo procesar el contenido del documento.")
    
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    print("Sube tus documentos (PDFs o imágenes) mediante la petición POST en el endpoint interactivo.")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
